//! Standalone exact fixture for the lexical margin measurement, with no model load.
use std::collections::BTreeSet;

const Q48_STEP: i64 = 1 << 32;
type Words = [[i16; 2]; 2];

#[derive(Clone, Copy)]
struct Candidate {
    id: u16,
    prior: i64,
    address: [usize; 2],
}

fn candidate(id: u16, prior_steps: i64, address: [usize; 2]) -> Candidate {
    Candidate {
        id,
        prior: prior_steps * Q48_STEP + 1,
        address,
    }
}

fn score(candidate: Candidate, words: &Words) -> i64 {
    candidate.prior
        + Q48_STEP
            * (i64::from(words[0][candidate.address[0]])
                + i64::from(words[1][candidate.address[1]]))
}

fn key(candidate: Candidate, words: &Words) -> (i64, std::cmp::Reverse<u16>) {
    (score(candidate, words), std::cmp::Reverse(candidate.id))
}

fn strongest_rival(gold: Candidate, rivals: &[Candidate], words: &Words) -> Candidate {
    assert!(rivals.iter().all(|rival| rival.id != gold.id));
    *rivals
        .iter()
        .max_by_key(|&&rival| key(rival, words))
        .unwrap()
}

fn margin(gold: Candidate, rival: Candidate, words: &Words) -> i64 {
    score(gold, words) - score(rival, words)
}

fn differing_banks(gold: Candidate, rival: Candidate) -> i64 {
    gold.address
        .iter()
        .zip(rival.address)
        .filter(|&(&gold_slot, rival_slot)| gold_slot != rival_slot)
        .count() as i64
}

fn upper(gold: Candidate, rival: Candidate, low: i16, high: i16) -> i64 {
    gold.prior - rival.prior
        + differing_banks(gold, rival) * (i64::from(high) - i64::from(low)) * Q48_STEP
}

fn extremizer(gold: Candidate, low: i16, high: i16) -> Words {
    let mut words = [[low; 2]; 2];
    for (bank, slot) in gold.address.into_iter().enumerate() {
        words[bank][slot] = high;
    }
    words
}

fn apply(word: &mut i16, delta: i64) -> bool {
    let raw = i64::from(*word) + delta;
    let bounded = raw.clamp(i64::from(i16::MIN), i64::from(i16::MAX));
    *word = i16::try_from(bounded).unwrap();
    raw != bounded
}

struct Visit {
    old_rival: u16,
    new_rival: u16,
    before: i64,
    after_old: i64,
    after_all: i64,
    clipped: usize,
}

fn update(gold: Candidate, rivals: &[Candidate], words: &mut Words) -> Visit {
    let old = strongest_rival(gold, rivals, words);
    let before = margin(gold, old, words);
    let changed = differing_banks(gold, old);
    let mut clipped = 0;
    if before <= 0 && changed != 0 {
        let need = 1 - before;
        let per_step = 2 * changed * Q48_STEP;
        let delta = (need + per_step - 1) / per_step;
        for (bank, (&g, &r)) in gold.address.iter().zip(&old.address).enumerate() {
            if g != r {
                clipped += usize::from(apply(&mut words[bank][g], delta));
                clipped += usize::from(apply(&mut words[bank][r], -delta));
            }
        }
    }
    let new = strongest_rival(gold, rivals, words);
    Visit {
        old_rival: old.id,
        new_rival: new.id,
        before,
        after_old: margin(gold, old, words),
        after_all: margin(gold, new, words),
        clipped,
    }
}

#[test]
fn simultaneous_ceiling_matches_all_tiny_assignments() {
    let (low, high) = (-1, 2);
    for target in [[0, 0], [0, 1], [1, 0], [1, 1]] {
        let gold = candidate(9, 4, target);
        let rivals = [
            candidate(8, 5, target),
            candidate(4, 8, [1 - target[0], target[1]]),
            candidate(5, 7, [target[0], 1 - target[1]]),
            candidate(6, 10, [1 - target[0], 1 - target[1]]),
        ];
        let uppers = rivals.map(|rival| upper(gold, rival, low, high));
        assert_eq!(uppers, [-Q48_STEP, -Q48_STEP, 0, 0]);
        let largest_minimum = *uppers.iter().min().unwrap();
        let extremal = extremizer(gold, low, high);
        assert_eq!(rivals.map(|rival| margin(gold, rival, &extremal)), uppers);
        let mut brute_maxima = [i64::MIN; 4];
        let mut brute_best_all = i64::MIN;
        let mut count = 0;
        for a in low..=high {
            for b in low..=high {
                for c in low..=high {
                    for d in low..=high {
                        let words = [[a, b], [c, d]];
                        let observed = rivals.map(|rival| margin(gold, rival, &words));
                        for (index, value) in observed.into_iter().enumerate() {
                            assert!(value <= uppers[index]);
                            brute_maxima[index] = brute_maxima[index].max(value);
                        }
                        brute_best_all = brute_best_all.max(*observed.iter().min().unwrap());
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 256);
        assert_eq!(brute_maxima, uppers);
        assert_eq!(brute_best_all, largest_minimum);
    }
}

#[test]
fn immutable_alias_preserves_prior_and_original_id_tie_order() {
    let gold = candidate(9, 3, [1, 0]);
    let stronger = candidate(20, 4, [1, 0]);
    let same_prior_lower_id = candidate(8, 3, [1, 0]);
    let same_prior_higher_id = candidate(10, 3, [1, 0]);
    for words in [
        [[0, 0], [0, 0]],
        extremizer(gold, i16::MIN, i16::MAX),
        extremizer(stronger, -2, 3),
    ] {
        assert_eq!(margin(gold, stronger, &words), -Q48_STEP);
        assert_eq!(upper(gold, stronger, i16::MIN, i16::MAX), -Q48_STEP);
        assert_eq!(margin(gold, same_prior_lower_id, &words), 0);
        assert_eq!(margin(gold, same_prior_higher_id, &words), 0);
        assert!(key(same_prior_lower_id, &words) > key(gold, &words));
        assert!(key(gold, &words) > key(same_prior_higher_id, &words));
        assert_eq!(
            strongest_rival(gold, &[same_prior_higher_id, same_prior_lower_id], &words).id,
            8
        );
    }
    let mut words = [[0; 2]; 2];
    let observed = update(gold, &[stronger], &mut words);
    assert_eq!(
        (observed.before, observed.after_old, observed.after_all),
        (-Q48_STEP, -Q48_STEP, -Q48_STEP)
    );
    assert_eq!(words, [[0; 2]; 2]);
}

#[test]
fn old_rival_correction_does_not_hide_the_new_winner() {
    let gold = candidate(9, 0, [0, 0]);
    let old = candidate(8, 6, [1, 1]);
    let new = candidate(7, 5, [0, 1]);
    let mut words = [[0; 2]; 2];
    let observed = update(gold, &[new, old], &mut words);
    assert_eq!(words, [[2, -2], [2, -2]]);
    assert_eq!((observed.old_rival, observed.new_rival), (8, 7));
    assert_eq!(observed.before, -6 * Q48_STEP);
    assert_eq!(observed.after_old, 2 * Q48_STEP);
    assert_eq!(observed.after_all, -Q48_STEP);
    assert_eq!(observed.clipped, 0);
}

#[test]
fn a_subsequent_example_erases_a_previous_all_id_win() {
    let a = candidate(9, 0, [0, 0]);
    let a_rivals = [candidate(8, 1, [1, 1]), candidate(7, 0, [0, 1])];
    let b = candidate(10, 0, [1, 1]);
    let b_rivals = [candidate(11, 1, [0, 0]), candidate(12, 0, [1, 0])];
    let mut words = [[0; 2]; 2];
    let first = update(a, &a_rivals, &mut words);
    let immediate = words;
    assert_eq!((first.new_rival, first.after_all), (7, 2 * Q48_STEP));
    let second = update(b, &b_rivals, &mut words);
    assert_eq!((second.new_rival, second.after_all), (12, 2 * Q48_STEP));
    assert_eq!(words, [[-1, 1], [-1, 1]]);
    assert_eq!(
        margin(a, strongest_rival(a, &a_rivals, &immediate), &immediate),
        2 * Q48_STEP
    );
    assert_eq!(
        margin(a, strongest_rival(a, &a_rivals, &words), &words),
        -5 * Q48_STEP
    );
    assert_eq!(first.clipped + second.clipped, 0);
}

#[test]
fn bounded_update_reports_saturation_without_wraparound() {
    let gold = candidate(9, 0, [0, 0]);
    let rival = candidate(8, 65_535, [1, 0]);
    let mut words = [[i16::MAX, i16::MIN], [0, 0]];
    let result = update(gold, &[rival], &mut words);
    assert_eq!(words, [[32_767, -32_768], [0, 0]]);
    assert_eq!(result.clipped, 2);
    assert_eq!(
        (result.before, result.after_old, result.after_all),
        (0, 0, 0)
    );
    assert_eq!(upper(gold, rival, i16::MIN, i16::MAX), 0);
    assert!(key(rival, &words) > key(gold, &words));

    let mut high = i16::MAX - 1;
    let mut low = i16::MIN + 1;
    assert!(!apply(&mut high, 1));
    assert!(!apply(&mut low, -1));
    assert!(apply(&mut high, 2));
    assert!(apply(&mut low, -2));
    assert_eq!((high, low), (i16::MAX, i16::MIN));
}

#[test]
fn authoritative_native_root_transitions_reach_26_bins_per_bank() {
    // Include source bytes only. This fixture never compiles or loads a model.
    let source = include_str!("../../../../../crates/uor-r4-integer/src/discrete_d11.rs");
    let table = source
        .split_once("const BFS: [u8; 256] = [")
        .unwrap()
        .1
        .split_once("];")
        .unwrap()
        .0;
    let table: Vec<usize> = table
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| entry.parse().unwrap())
        .collect();
    assert_eq!(table.len(), 16 * 16);
    for root in 0..16 {
        assert_eq!(table[16 * root + root], 0);
    }
    let mut all = BTreeSet::new();
    let expected = BTreeSet::from([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 26, 27, 34, 35, 41, 42, 47,
        48,
    ]);
    for bank in 0..4 {
        let slots: BTreeSet<_> = table.iter().map(|&bin| bank * 64 + bin).collect();
        assert_eq!(slots.len(), 26);
        assert_eq!(
            slots
                .iter()
                .map(|&slot| slot - bank * 64)
                .collect::<BTreeSet<_>>(),
            expected
        );
        assert!(slots.iter().all(|&slot| all.insert(slot)));
    }
    assert_eq!(all.len(), 104);
}
