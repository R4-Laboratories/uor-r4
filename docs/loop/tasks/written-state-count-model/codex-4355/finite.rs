//! Standalone exact controls for the bounded lexical update. No model or device input.
const UNIT: i64 = 1 << 32;
const NATIVE: (i16, i16) = (i16::MIN, i16::MAX);

type Words<const BANKS: usize, const SLOTS: usize> = [[i16; SLOTS]; BANKS];

#[derive(Clone, Copy, Debug)]
struct Candidate<const BANKS: usize> {
    id: u16,
    prior: i64,
    slots: [usize; BANKS],
}

fn candidate<const BANKS: usize>(
    id: u16,
    prior_steps: i64,
    slots: [usize; BANKS],
) -> Candidate<BANKS> {
    Candidate {
        id,
        prior: prior_steps * UNIT,
        slots,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    Full,
    Sparse,
}

fn score<const B: usize, const S: usize>(word: Candidate<B>, weights: &Words<B, S>) -> i64 {
    word.prior
        + UNIT
            * word
                .slots
                .iter()
                .enumerate()
                .map(|(bank, slot)| i64::from(weights[bank][*slot]))
                .sum::<i64>()
}

fn margins<const B: usize, const S: usize, const R: usize>(
    gold: Candidate<B>,
    rivals: [Candidate<B>; R],
    weights: &Words<B, S>,
) -> [i64; R] {
    let gold_score = score(gold, weights);
    rivals.map(|rival| gold_score - score(rival, weights))
}

fn rank<const B: usize, const S: usize, const R: usize>(
    gold: Candidate<B>,
    rivals: [Candidate<B>; R],
    weights: &Words<B, S>,
) -> usize {
    let key = |word: Candidate<B>| (score(word, weights), std::cmp::Reverse(word.id));
    1 + rivals
        .iter()
        .filter(|&&rival| key(rival) > key(gold))
        .count()
}

fn extremizer<const B: usize, const S: usize, const R: usize>(
    start: &Words<B, S>,
    gold: Candidate<B>,
    rivals: [Candidate<B>; R],
    policy: Policy,
    radius: i32,
    (low, high): (i16, i16),
) -> Words<B, S> {
    let mut weights = *start;
    for (bank, words) in weights.iter_mut().enumerate() {
        let gold_slot = gold.slots[bank];
        words[gold_slot] = (i32::from(start[bank][gold_slot]) + radius).min(i32::from(high)) as i16;
        if policy == Policy::Full {
            for rival in rivals {
                let slot = rival.slots[bank];
                if slot != gold_slot {
                    words[slot] =
                        (i32::from(start[bank][slot]) - radius).max(i32::from(low)) as i16;
                }
            }
        }
    }
    weights
}

fn correction<const B: usize, const S: usize, const R: usize>(
    weights: &mut Words<B, S>,
    gold: Candidate<B>,
    rivals: [Candidate<B>; R],
    policy: Policy,
    bounds: (i16, i16),
) -> Option<i32> {
    let wins = |radius| {
        let trial = extremizer(weights, gold, rivals, policy, radius, bounds);
        margins(gold, rivals, &trial).iter().all(|&m| m > 0)
    };
    let mut low = 0;
    let mut high = i32::from(bounds.1) - i32::from(bounds.0);
    if !wins(high) {
        return None;
    }
    while low < high {
        let mid = low + (high - low) / 2;
        if wins(mid) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    *weights = extremizer(weights, gold, rivals, policy, low, bounds);
    Some(low)
}

fn cube_word(mut index: usize) -> Words<2, 2> {
    let mut weights = [[0; 2]; 2];
    for word in weights.iter_mut().flatten() {
        *word = (index % 3) as i16 - 1;
        index /= 3;
    }
    weights
}

#[test]
fn exhaustive_cube_checks_every_margin_and_least_displacement() {
    let mut attainable = [0; 2];
    let mut blocked = [0; 2];
    for start in (0..81).map(cube_word) {
        for (g0, g1) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            let gold = candidate(9, 0, [g0, g1]);
            let addresses = [[1 - g0, g1], [g0, 1 - g1], [1 - g0, 1 - g1]];
            for priors in [[UNIT, 2 * UNIT - 1, 3 * UNIT], [1; 3], [2 * UNIT, 0, 0]] {
                let rivals = std::array::from_fn(|i| Candidate {
                    id: [7, 10, 11][i],
                    prior: priors[i],
                    slots: addresses[i],
                });
                for (i, policy) in [Policy::Full, Policy::Sparse].into_iter().enumerate() {
                    let mut least = None;
                    for radius in 0..=2 {
                        let mut maxima = [i64::MIN; 3];
                        for proposal in (0..81).map(cube_word) {
                            let legal = start.iter().zip(&proposal).enumerate().all(
                                |(bank, (before, after))| {
                                    before.iter().zip(after).enumerate().all(
                                        |(slot, (&old, &new))| {
                                            let change = i32::from(new) - i32::from(old);
                                            change.abs() <= radius
                                                && (policy == Policy::Full
                                                    || if slot == gold.slots[bank] {
                                                        change >= 0
                                                    } else {
                                                        change == 0
                                                    })
                                        },
                                    )
                                },
                            );
                            if legal {
                                let observed = margins(gold, rivals, &proposal);
                                for (maximum, margin) in maxima.iter_mut().zip(observed) {
                                    *maximum = (*maximum).max(margin);
                                }
                                if observed.iter().all(|&margin| margin > 0) && least.is_none() {
                                    least = Some(radius);
                                }
                            }
                        }
                        let output = extremizer(&start, gold, rivals, policy, radius, (-1, 1));
                        assert_eq!(margins(gold, rivals, &output), maxima);
                    }
                    let mut updated = start;
                    assert_eq!(
                        correction(&mut updated, gold, rivals, policy, (-1, 1)),
                        least
                    );
                    let displacement = updated
                        .iter()
                        .flatten()
                        .zip(start.iter().flatten())
                        .map(|(&new, &old)| (i32::from(new) - i32::from(old)).abs())
                        .max();
                    if least.is_some() {
                        attainable[i] += 1;
                        assert_eq!(displacement, least);
                        assert!(margins(gold, rivals, &updated).iter().all(|&m| m > 0));
                    } else {
                        blocked[i] += 1;
                        assert_eq!(updated, start);
                    }
                }
            }
        }
    }
    assert!(attainable[0] > attainable[1] && attainable[1] > 0);
    assert!(blocked[1] > blocked[0] && blocked[0] > 0);
}

#[test]
fn obsolete_rival_only_correction_misses_a_third_address() {
    let start = [[0, 0, 0, 7], [0, 0, 0, -7]];
    let gold = candidate(9, 0, [0, 0]);
    let rivals = [candidate(8, 6, [1, 1]), candidate(7, 5, [0, 2])];
    assert_eq!(margins(gold, rivals, &start), [-6 * UNIT, -5 * UNIT]);
    let old = [[2, -2, 0, 7], [2, -2, 0, -7]];
    assert_eq!(margins(gold, rivals, &old), [2 * UNIT, -3 * UNIT]);
    assert_eq!(rank(gold, rivals, &old), 2);

    let full_short = extremizer(&start, gold, rivals, Policy::Full, 2, NATIVE);
    assert_eq!(margins(gold, rivals, &full_short), [2 * UNIT, -UNIT]);
    let mut full = start;
    assert_eq!(
        correction(&mut full, gold, rivals, Policy::Full, NATIVE),
        Some(3)
    );
    assert_eq!(full, [[3, -3, 0, 7], [3, -3, -3, -7]]);
    assert_eq!(margins(gold, rivals, &full), [6 * UNIT, UNIT]);
    assert_eq!(rank(gold, rivals, &full), 1);

    let sparse_short = extremizer(&start, gold, rivals, Policy::Sparse, 5, NATIVE);
    assert_eq!(margins(gold, rivals, &sparse_short), [4 * UNIT, 0]);
    assert_eq!(rank(gold, rivals, &sparse_short), 2);
    let mut sparse = start;
    assert_eq!(
        correction(&mut sparse, gold, rivals, Policy::Sparse, NATIVE),
        Some(6)
    );
    assert_eq!(sparse, [[6, 0, 0, 7], [6, 0, 0, -7]]);
    assert_eq!(margins(gold, rivals, &sparse), [6 * UNIT, UNIT]);
    for word in [gold, rivals[0], rivals[1]] {
        assert_eq!(score(word, &sparse) - score(word, &full), 6 * UNIT);
    }
}

#[test]
fn clipped_nongold_moves_differ_and_sparse_cannot_strictly_win() {
    let start = [[32_766, -32_768, -32_767]];
    let gold = candidate(9, 0, [0]);
    let rivals = [candidate(7, 65_534, [1]), candidate(8, 65_534, [2])];
    assert_eq!(margins(gold, rivals, &start), [0, -UNIT]);
    let mut full = start;
    assert_eq!(
        correction(&mut full, gold, rivals, Policy::Full, NATIVE),
        Some(1)
    );
    assert_eq!(full, [[32_767, -32_768, -32_768]]);
    assert_eq!(margins(gold, rivals, &full), [UNIT; 2]);
    let sparse_ceiling = extremizer(&start, gold, rivals, Policy::Sparse, 65_535, NATIVE);
    assert_eq!(margins(gold, rivals, &sparse_ceiling), [UNIT, 0]);
    assert_eq!(
        score(rivals[0], &full) - score(rivals[0], &sparse_ceiling),
        0
    );
    assert_eq!(
        score(rivals[1], &full) - score(rivals[1], &sparse_ceiling),
        -UNIT
    );
    let mut sparse = start;
    assert_eq!(
        correction(&mut sparse, gold, rivals, Policy::Sparse, NATIVE),
        None
    );
    assert_eq!(sparse, start);
}

#[test]
fn full_signed_span_is_needed_and_does_not_wrap() {
    let start = [[i16::MIN, i16::MAX]];
    let gold = candidate(9, 0, [0]);
    let rivals = [candidate(8, 65_534, [1])];
    let short = extremizer(&start, gold, rivals, Policy::Full, 65_534, NATIVE);
    assert_eq!(margins(gold, rivals, &short), [-UNIT]);
    let mut full = start;
    assert_eq!(
        correction(&mut full, gold, rivals, Policy::Full, NATIVE),
        Some(65_535)
    );
    assert_eq!(full, [[i16::MAX, i16::MIN]]);
    assert_eq!(margins(gold, rivals, &full), [UNIT]);
}

#[test]
fn raw_dyadic_unit_and_alias_ties_keep_original_id_order() {
    let gold = candidate(9, 0, [0]);
    let mut rival = candidate(8, 2, [1]);
    let mut weights = [[0, 0]];
    assert_eq!(rank(gold, [rival], &[[1, -1]]), 2);
    let higher_id = Candidate { id: 10, ..rival };
    assert_eq!(rank(gold, [higher_id], &[[1, -1]]), 1);
    assert_eq!(margins(gold, [higher_id], &[[1, -1]]), [0]);
    assert_eq!(
        correction(&mut weights, gold, [higher_id], Policy::Full, NATIVE),
        Some(2)
    );
    assert_eq!(margins(gold, [higher_id], &weights), [2 * UNIT]);

    rival.prior -= 1;
    let mut weights = [[0, 0]];
    assert_eq!(
        correction(&mut weights, gold, [rival], Policy::Full, NATIVE),
        Some(1)
    );
    assert_eq!(margins(gold, [rival], &weights), [1]);
    let start = weights;
    assert_eq!(
        correction(&mut weights, gold, [rival], Policy::Full, NATIVE),
        Some(0)
    );
    assert_eq!(weights, start);

    for (id, prior, expected_rank) in [(8, 0, 2), (10, 0, 1), (10, 1, 2)] {
        let alias = Candidate {
            id,
            prior,
            slots: gold.slots,
        };
        assert_eq!(rank(gold, [alias], &weights), expected_rank);
        for policy in [Policy::Full, Policy::Sparse] {
            assert_eq!(
                correction(&mut weights, gold, [alias], policy, NATIVE),
                None
            );
            assert_eq!(weights, start);
            assert_eq!(margins(gold, [alias], &weights), [-prior]);
        }
    }
}

#[test]
fn earlier_instant_wins_can_survive_or_be_erased_at_one_checkpoint() {
    let rows = [
        (candidate(9, 0, [0]), [candidate(8, 1, [1])]),
        (candidate(9, 0, [2]), [candidate(8, 1, [3])]),
        (candidate(9, 0, [1]), [candidate(8, 1, [0])]),
    ];
    for (policy, radii, expected_checkpoint) in [
        (Policy::Full, [1, 1, 2], [[-1, 1, 1, -1]]),
        (Policy::Sparse, [2, 2, 4], [[2, 4, 2, 0]]),
    ] {
        let mut weights = [[0; 4]];
        let mut instant = [false; 3];
        for (i, (gold, rivals)) in rows.into_iter().enumerate() {
            assert_eq!(
                correction(&mut weights, gold, rivals, policy, NATIVE),
                Some(radii[i])
            );
            let margin = margins(gold, rivals, &weights)[0];
            assert_eq!(margin, UNIT);
            instant[i] = margin > 0;
        }
        assert_eq!(weights, expected_checkpoint);
        let final_margins = rows.map(|(gold, rivals)| margins(gold, rivals, &weights)[0]);
        assert_eq!(final_margins, [-3 * UNIT, UNIT, UNIT]);
        let final_wins = final_margins.map(|m| m > 0);
        assert_eq!(instant, [true; 3]);
        assert_eq!(final_wins, [false, true, true]);
    }
}
