// The scalar lookup is the training oracle. For decoding we traverse each
// matching sorted follower list once for totals and once for all targets.
enum ActiveSource<'a> {
    V2 {
        table: &'a CompactOrder<'a>,
        remaining: &'a [u8],
    },
    V3 {
        table: &'a CompactV3Order<'a>,
        pos: usize,
        end: usize,
        esc_rank: usize,
    },
}

struct ActiveOrder<'a> {
    source: ActiveSource<'a>,
    discounts: KnDiscounts,
    adjusted: NgramContext,
}

impl CompactCountLookup<'_> {
    /// Emit all legal original target IDs in ascending order with their fit
    /// bucket and exact Q48 mass. A caller selects the lowest ID on a tie.
    /// The count and Pick operators use only the supplied causal prefix.
    pub fn visit_next_masses(
        &self,
        prefix: &[u16],
        fit: &FrozenPickMixture,
        picked: Option<(u16, u32)>,
        mut visit: impl FnMut(u16, u16, u64),
    ) -> Result<()> {
        if picked.is_some_and(|(id, _)| id == NGRAM_BOS || u32::from(id) >= NGRAM_VOCABULARY) {
            return Err(invalid("invalid A1 successor"));
        }
        let class = fit.rule().classify(picked.map(|(_, distance)| distance));
        const WIDTH: usize = NGRAM_MAX_ORDER as usize + 1;
        let mut active: [Option<ActiveOrder<'_>>; WIDTH] = std::array::from_fn(|_| None);
        let mut totals = [0; WIDTH];
        let width = self.order_count();
        match &self.orders {
            LookupOrders::V2(orders) => {
                for (k, table) in orders
                    .iter()
                    .enumerate()
                    .take(prefix.len().saturating_add(1))
                {
                    let key = PrimeRouteKey::from_context(&prefix[prefix.len() - k..])?;
                    let body = key_body(key.word(), k as u32);
                    let Some(followers) = table.matched_rows(&body[16 - table.key_width..]) else {
                        continue;
                    };
                    let mut context = NgramContext {
                        key: key.word(),
                        total: 0,
                        followers: 0,
                        n1: 0,
                        n2: 0,
                        n3_plus: 0,
                    };
                    for row in followers.chunks_exact(follower_width(table.shared)) {
                        let (_, raw, adjusted) = table.read_follower(row);
                        totals[k] += u64::from(raw);
                        context.total += u64::from(adjusted);
                        context.followers += 1;
                        match adjusted {
                            1 => context.n1 += 1,
                            2 => context.n2 += 1,
                            _ => context.n3_plus += 1,
                        }
                    }
                    active[k] = Some(ActiveOrder {
                        source: ActiveSource::V2 {
                            table,
                            remaining: followers,
                        },
                        discounts: table.discounts,
                        adjusted: context,
                    });
                }
            }
            LookupOrders::V3(orders) => {
                for &tok in &prefix[prefix.len().saturating_sub(orders.len().saturating_sub(1))..] {
                    if u32::from(tok) >= NGRAM_VOCABULARY {
                        return Err(invalid("token id is outside the 12-bit vocabulary"));
                    }
                }
                let mut ctx_idx = 0usize;
                for (k, table) in orders
                    .iter()
                    .enumerate()
                    .take(prefix.len().saturating_add(1))
                {
                    if k == 0 {
                        if table.contexts == 0 {
                            break;
                        }
                        ctx_idx = 0;
                    } else {
                        let child_tok = prefix[prefix.len() - k];
                        let Some(next_ctx) = table.find_child(ctx_idx, child_tok) else {
                            break;
                        };
                        ctx_idx = next_ctx;
                    }
                    let (start, end) = table
                        .follower_span(ctx_idx)
                        .ok_or_else(|| invalid("invalid v3 follower span"))?;
                    let init_rank = table.esc_rank_at(start);
                    let mut esc_rank = init_rank;
                    let mut context = NgramContext {
                        key: 0,
                        total: 0,
                        followers: 0,
                        n1: 0,
                        n2: 0,
                        n3_plus: 0,
                    };
                    for f in start..end {
                        let (_, raw, adjusted) = table
                            .read_follower_counts(f, &mut esc_rank)
                            .ok_or_else(|| invalid("invalid v3 follower"))?;
                        totals[k] += u64::from(raw);
                        context.total += u64::from(adjusted);
                        context.followers += 1;
                        match adjusted {
                            1 => context.n1 += 1,
                            2 => context.n2 += 1,
                            _ => context.n3_plus += 1,
                        }
                    }
                    active[k] = Some(ActiveOrder {
                        source: ActiveSource::V3 {
                            table,
                            pos: start,
                            end,
                            esc_rank: init_rank,
                        },
                        discounts: table.discounts,
                        adjusted: context,
                    });
                }
            }
        }
        for id in 1..NGRAM_VOCABULARY as u16 {
            let mut counts = [0; WIDTH];
            let mut mass = u128::from(KN_BASE_MASS);
            for (order, slot) in active[..width].iter_mut().enumerate() {
                let Some(level) = slot else { continue };
                let mut adjusted_target = 0;
                match &mut level.source {
                    ActiveSource::V2 { table, remaining } => {
                        let stride = follower_width(table.shared);
                        if let Some(row) = remaining.get(..stride) {
                            let (candidate, raw, adjusted) = table.read_follower(row);
                            if candidate == id {
                                counts[order] = raw;
                                adjusted_target = adjusted;
                                *remaining = &remaining[stride..];
                            }
                        }
                    }
                    ActiveSource::V3 {
                        table,
                        pos,
                        end,
                        esc_rank,
                    } => {
                        if *pos < *end {
                            let w = read_u16_le(table.fol_words, *pos);
                            let candidate = w >> 4;
                            if candidate == id {
                                let (_, raw, adjusted) = table
                                    .read_follower_counts(*pos, esc_rank)
                                    .ok_or_else(|| invalid("invalid v3 follower"))?;
                                counts[order] = raw;
                                adjusted_target = adjusted;
                                *pos += 1;
                            }
                        }
                    }
                }
                mass = interpolate_order(mass, adjusted_target, level.adjusted, level.discounts);
            }
            let pick = PickTarget {
                class,
                hit: picked.is_some_and(|(token, _)| token == id),
            };
            let kn =
                u64::try_from(mass.max(1)).map_err(|_| invalid("compact KN mass exceeded u64"))?;
            let (bucket, q48) = fit.predict_mass(kn, &counts[..width], &totals[..width], pick)?;
            visit(id, bucket, q48);
        }
        Ok(())
    }
}
