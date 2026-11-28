// Reuse the sealed capture, P4 controls, ABI, and checkpoint scorer.
include!("../../../codex-4354/runner/src/device.rs");

const ELECTION_CUBIN: &[u8] = include_bytes!(env!("FULL_ELECTION_CUBIN"));

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[repr(u32)]
pub enum ElectionPolicy {
    GoldOnly,
    Full,
}

const ALREADY_STRICT: u32 = 0;
const UPDATED: u32 = 1;
const INFEASIBLE: u32 = 2;
const WORD_SPAN: u32 = u16::MAX as u32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ElectionVisit {
    pub before: Decision,
    pub after: Decision,
    pub radius: u32,
    pub changed_words: u32,
    pub l1: u32,
    pub saturated_words: u32,
    pub outcome: u32,
    #[serde(skip)]
    reserved: u32,
}

// SAFETY: ElectionVisit has the CUDA layout and every field accepts zero,
// including the final reserved word that accounts for alignment.
unsafe impl DeviceRepr for ElectionVisit {}
unsafe impl ValidAsZeroBits for ElectionVisit {}
const _: [(); 72] = [(); size_of::<ElectionVisit>()];

#[derive(Debug, Default, Serialize)]
pub struct ElectionStats {
    pub rows: u32,
    pub already_strict: u32,
    pub updates: u32,
    pub infeasible: u32,
    pub changed_words: u64,
    pub l1: u64,
    pub saturated_words: u64,
}

#[derive(Serialize)]
pub struct ElectionEpoch {
    pub stats: ElectionStats,
    #[serde(serialize_with = "serialize_weights")]
    pub weights: [i16; WEIGHT_WORDS],
    pub visits: Vec<ElectionVisit>,
    pub checkpoint: Vec<Decision>,
}

#[derive(Serialize)]
pub struct ElectionFit {
    #[serde(serialize_with = "serialize_weights")]
    pub weights: [i16; WEIGHT_WORDS],
    pub epochs: Vec<ElectionEpoch>,
}

fn summarize_visits(
    visits: &[ElectionVisit],
    targets: &[u16],
    policy: ElectionPolicy,
    support: &Support,
) -> Result<ElectionStats> {
    let active_words = match policy {
        ElectionPolicy::GoldOnly => 4,
        ElectionPolicy::Full => support.geometry.iter().map(|mask| mask.count_ones()).sum(),
    };
    if visits.len() != targets.len() {
        return Err(fail("full-election visit count differs from target count"));
    }
    let mut stats = ElectionStats {
        rows: u32::try_from(visits.len())?,
        ..ElectionStats::default()
    };
    for (&visit, &gold) in visits.iter().zip(targets) {
        validate_decision(visit.before, gold)?;
        validate_decision(visit.after, gold)?;
        let unchanged = visit.before == visit.after
            && visit.radius == 0
            && visit.changed_words == 0
            && visit.l1 == 0
            && visit.saturated_words == 0;
        if visit.reserved != 0 {
            return Err(fail("full-election visit has nonzero reserved bits"));
        }
        match visit.outcome {
            ALREADY_STRICT if visit.before.margin_q48 > 0 && unchanged => stats.already_strict += 1,
            INFEASIBLE if visit.before.margin_q48 <= 0 && unchanged => stats.infeasible += 1,
            UPDATED
                if visit.before.margin_q48 <= 0
                    && visit.after.margin_q48 > 0
                    && (1..=WORD_SPAN).contains(&visit.radius)
                    && (1..=active_words).contains(&visit.changed_words)
                    && visit.saturated_words <= active_words
                    && (visit.changed_words..=visit.changed_words * visit.radius)
                        .contains(&visit.l1) =>
            {
                stats.updates += 1;
                stats.changed_words += u64::from(visit.changed_words);
                stats.l1 += u64::from(visit.l1);
                stats.saturated_words += u64::from(visit.saturated_words);
            }
            _ => {
                return Err(fail(format!(
                    "invalid full-election visit for token {gold}: {visit:?}"
                )))
            }
        }
    }
    Ok(stats)
}

impl Gpu {
    pub fn election_digest(&self) -> String {
        hex::encode(Sha256::digest(ELECTION_CUBIN))
    }

    pub fn fit_all(
        &self,
        capture: &Capture,
        targets: &[u16],
        orders: &[Vec<u32>],
        mode: Mode,
        support: &Support,
        policy: ElectionPolicy,
    ) -> Result<ElectionFit> {
        self.fit_all_from(
            capture,
            targets,
            orders,
            mode,
            support,
            policy,
            &[0; WEIGHT_WORDS],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn fit_all_from(
        &self,
        capture: &Capture,
        targets: &[u16],
        orders: &[Vec<u32>],
        mode: Mode,
        support: &Support,
        policy: ElectionPolicy,
        starting_weights: &[i16; WEIGHT_WORDS],
    ) -> Result<ElectionFit> {
        if orders.is_empty() {
            return Err(fail("empty full-election fit orders"));
        }
        for order in orders {
            let mut seen = vec![false; targets.len()];
            if order.len() != targets.len()
                || order.iter().any(|&row| {
                    !seen
                        .get_mut(row as usize)
                        .is_some_and(|slot| !std::mem::replace(slot, true))
                })
            {
                return Err(fail("full-election epoch must visit each row once"));
            }
        }
        let n = u32::try_from(targets.len())?;
        let device_targets = self.validated_targets(capture, targets)?;
        let map = self.map(support)?;
        let module = self
            .stream
            .context()
            .load_module(Ptx::from_binary(ELECTION_CUBIN.to_vec()))?;
        let fit = module.load_function("full_election_fit")?;
        let mut weights = self.stream.clone_htod(starting_weights)?;
        let mut visits = self.stream.alloc_zeros::<ElectionVisit>(targets.len())?;
        let mode = mode as u32;
        let device_policy = policy as u32;
        let mut epochs = Vec::with_capacity(orders.len());
        for order in orders {
            let device_order = self.stream.clone_htod(order)?;
            let mut launch = self.stream.launch_builder(&fit);
            launch
                .arg(&n)
                .arg(&device_order)
                .arg(&device_targets)
                .arg(&capture.device_features)
                .arg(&capture.priors)
                .arg(&self.trunk)
                .arg(&mode)
                .arg(&map)
                .arg(&device_policy)
                .arg(&mut weights)
                .arg(&mut visits);
            // SAFETY: validated row indices and targets fit their corresponding
            // buffers. One block exclusively owns the sequential weight updates.
            unsafe { launch.launch(grid(1))? };
            let epoch_words = self.read_weights(&weights)?;
            if epoch_words.iter().enumerate().any(|(word, &value)| {
                value != starting_weights[word]
                    && support.geometry[word / 64] & (1u64 << (word % 64)) == 0
            }) {
                return Err(fail("full-election fit wrote outside the matched support"));
            }
            let visits = self.stream.clone_dtoh(&visits)?;
            let stats = summarize_visits(&visits, targets, policy, support)?;
            let (checkpoint, _) = self.scan_checkpoint(
                capture,
                targets,
                &device_targets,
                mode,
                &map,
                &weights,
                false,
            )?;
            if checkpoint[order[order.len() - 1] as usize]
                != visits[order[order.len() - 1] as usize].after
            {
                return Err(fail(
                    "full-election final visit disagreed with the frozen checkpoint",
                ));
            }
            epochs.push(ElectionEpoch {
                stats,
                weights: epoch_words,
                visits,
                checkpoint,
            });
        }
        let weights = epochs
            .last()
            .ok_or_else(|| fail("missing full-election epochs"))?
            .weights;
        Ok(ElectionFit { weights, epochs })
    }
}

// This fixture has two varying banks, with the strongest rival in each
// of the three non-gold address classes and one lower-prior gold alias.
// Common bank displacements cancel. Exhaust every remaining integer delta
// independently, without relying on the production extremizer.
fn finite_replacement_radius(policy: ElectionPolicy) -> Option<u32> {
    (0..=6).find(|&radius| {
        let down = if policy == ElectionPolicy::Full {
            radius
        } else {
            0
        };
        (0..=radius).any(|g0| {
            (0..=radius).any(|g1| {
                (0..=down).any(|r0| {
                    (0..=down).any(|r1| g0 + r0 + g1 + r1 > 6 && g1 + r1 > 5 && g0 + r0 + 99 > 0)
                })
            })
        })
    })
}

pub fn finite_election() -> Result<()> {
    let unit = SCORE_UNIT_Q48 as u64;
    let base = 100 * unit;
    let trunk = fixture_trunk(&[(7, 0x10), (8, 0x11), (10, 0x11), (12, 0x01)]);
    let gpu = Gpu::new(&trunk, p4_lexical_bfs())?;
    let prefixes = [vec![0, 7]];
    let geometry = Mode::Geometry;
    let replacement_priors =
        fixture_priors(&[(9, base), (8, base + 6 * unit), (7, base + 5 * unit)]);
    let replacement = gpu.capture(&prefixes, &replacement_priors)?;
    let support = gpu.support(&replacement)?;
    let carry = Carry::from_packed(replacement.features()[0].carry_code as u16);
    let gold_addresses = carry.addresses([0; 4])?.map(usize::from);
    let rival_addresses = carry.addresses([1, 1, 0, 0])?.map(usize::from);
    let saturated = gpu.capture(&prefixes, &fixture_priors(&[(9, base), (12, base + unit)]))?;
    let tied = gpu.capture(&prefixes, &vec![1; PRIOR_WORDS])?;
    let opposed = gpu.capture(
        &[prefixes[0].clone(), prefixes[0].clone()],
        &fixture_priors(&[(9, base), (8, base)]).repeat(2),
    )?;
    let mut at_limit = [0; WEIGHT_WORDS];
    at_limit[gold_addresses[0]] = i16::MAX;
    at_limit[rival_addresses[0]] = i16::MAX;
    // The candidate that differs only in bank 0 can no longer be beaten
    // by a gold-only move, but its word is still free to move downward.
    let mut at_ceiling = at_limit;
    at_ceiling[rival_addresses[0]] = i16::MIN;
    let ceiling_capture = gpu.capture(&prefixes, &fixture_priors(&[(9, 1), (12, ONE_Q48)]))?;

    for policy in [ElectionPolicy::GoldOnly, ElectionPolicy::Full] {
        let fit = gpu.fit_all(
            &replacement,
            &[9],
            &[vec![0], vec![0]],
            geometry,
            &support,
            policy,
        )?;
        let first = fit.epochs[0].visits[0];
        let repeat = fit.epochs[1].visits[0];
        let radius = finite_replacement_radius(policy)
            .ok_or_else(|| fail("finite replacement oracle found no radius"))?;
        let mut expected = [0; WEIGHT_WORDS];
        let step = i16::try_from(radius)?;
        for index in gold_addresses {
            expected[index] = step;
        }
        let changed_words = if policy == ElectionPolicy::Full {
            for index in &rival_addresses[..2] {
                expected[*index] = -step;
            }
            6
        } else {
            4
        };
        if first.before.rival != 8
            || first.before.margin_q48 != -6 * SCORE_UNIT_Q48
            || first.outcome != UPDATED
            || first.after.rival != 7
            || first.after.margin_q48 != SCORE_UNIT_Q48
            || first.radius != radius
            || first.changed_words != changed_words
            || first.l1 != changed_words * radius
            || first.saturated_words != 0
            || fit.epochs[0].weights != expected
            || repeat.outcome != ALREADY_STRICT
            || repeat.before != first.after
            || fit.weights != expected
        {
            return Err(fail(format!(
                "finite replacement disagreed with exhaustive search under {policy:?}: {first:?}"
            )));
        }

        for gold in [1, 2] {
            let alias = gpu.fit_all(&tied, &[gold], &[vec![0]], geometry, &support, policy)?;
            let visit = alias.epochs[0].visits[0];
            if visit.outcome != INFEASIBLE
                || visit.before.margin_q48 != 0
                || visit.before.rank != u32::from(gold)
                || alias.weights != [0; WEIGHT_WORDS]
            {
                return Err(fail(format!(
                    "finite original-ID tie changed for token {gold} under {policy:?}: {visit:?}"
                )));
            }
        }

        let fitted = gpu.fit_all_from(
            &saturated,
            &[9],
            &[vec![0]],
            geometry,
            &support,
            policy,
            &at_limit,
        )?;
        let visit = fitted.epochs[0].visits[0];
        let mut expected = at_limit;
        let expected_outcome = match policy {
            ElectionPolicy::GoldOnly => INFEASIBLE,
            ElectionPolicy::Full => {
                for index in &gold_addresses[1..] {
                    expected[*index] += 2;
                }
                for index in &rival_addresses[..2] {
                    expected[*index] -= 2;
                }
                if visit.radius != 2
                    || visit.changed_words != 5
                    || visit.saturated_words != 1
                    || visit.l1 != 10
                    || visit.after.margin_q48 != SCORE_UNIT_Q48
                {
                    return Err(fail(format!(
                        "finite directional saturation disagreed: {visit:?}"
                    )));
                }
                UPDATED
            }
        };
        if visit.before.rival != 12
            || visit.before.margin_q48 != -SCORE_UNIT_Q48
            || visit.outcome != expected_outcome
            || fitted.weights != expected
        {
            return Err(fail(format!(
                "finite one-sided saturation disagreed under {policy:?}: {visit:?}"
            )));
        }

        let hard = gpu.fit_all_from(
            &ceiling_capture,
            &[9],
            &[vec![0]],
            geometry,
            &support,
            policy,
            &at_ceiling,
        )?;
        let visit = hard.epochs[0].visits[0];
        if visit.outcome != INFEASIBLE || visit.before.rival != 12 || hard.weights != at_ceiling {
            return Err(fail(format!(
                "finite exhausted ceiling changed under {policy:?}: {visit:?}"
            )));
        }

        // For token 12 only bank 0 differs. The predecessor and exact deficits
        // separate strict raw-Q48 arithmetic from a rounded tie.
        let gain_per_step = if policy == ElectionPolicy::Full { 2 } else { 1 };
        for deficit in [2 * unit - 1, 2 * unit] {
            let capture = gpu.capture(
                &prefixes,
                &fixture_priors(&[(9, base), (12, base + deficit)]),
            )?;
            let fit = gpu.fit_all(&capture, &[9], &[vec![0]], geometry, &support, policy)?;
            let visit = fit.epochs[0].visits[0];
            let radius = 2 / gain_per_step + u32::from(deficit == 2 * unit);
            let margin =
                i64::from(radius * gain_per_step) * SCORE_UNIT_Q48 - i64::try_from(deficit)?;
            if visit.before.rival != 12
                || visit.before.margin_q48 != -i64::try_from(deficit)?
                || visit.outcome != UPDATED
                || visit.radius != radius
                || visit.after.rival != 12
                || visit.after.margin_q48 != margin
            {
                return Err(fail(format!(
                    "finite raw-Q48 boundary disagreed under {policy:?}: {visit:?}"
                )));
            }
        }

        // All rival words start at their lower limit, so even Full must
        // raise the differing gold word through the entire signed-word span.
        let max_deficit = u64::from(WORD_SPAN) * unit - 1;
        let capture = gpu.capture(&prefixes, &fixture_priors(&[(9, 3), (12, 3 + max_deficit)]))?;
        let minimum = [i16::MIN; WEIGHT_WORDS];
        let fit = gpu.fit_all_from(
            &capture,
            &[9],
            &[vec![0]],
            geometry,
            &support,
            policy,
            &minimum,
        )?;
        let visit = fit.epochs[0].visits[0];
        let expected_saturated = if policy == ElectionPolicy::Full { 6 } else { 4 };
        let mut expected = minimum;
        for index in gold_addresses {
            expected[index] = i16::MAX;
        }
        if visit.before.rival != 12
            || visit.before.margin_q48 != -i64::try_from(max_deficit)?
            || visit.outcome != UPDATED
            || visit.radius != WORD_SPAN
            || visit.changed_words != 4
            || visit.l1 != 4 * WORD_SPAN
            || visit.saturated_words != expected_saturated
            || visit.after.rival != 12
            || visit.after.margin_q48 != 1
            || fit.weights != expected
        {
            return Err(fail(format!(
                "finite maximum radius disagreed under {policy:?}: {visit:?}"
            )));
        }

        let fit = gpu.fit_all(&opposed, &[9, 8], &[vec![0, 1]], geometry, &support, policy)?;
        let epoch = &fit.epochs[0];
        if epoch.visits.iter().any(|visit| visit.outcome != UPDATED)
            || epoch.checkpoint[0].margin_q48 >= 0
            || epoch.checkpoint[1].margin_q48 <= 0
            || gpu
                .fit_all(&opposed, &[9, 8], &[vec![0, 0]], geometry, &support, policy)
                .is_ok()
        {
            return Err(fail(format!(
                "finite destructive subsequent visit or row order check failed under {policy:?}"
            )));
        }
    }
    Ok(())
}
