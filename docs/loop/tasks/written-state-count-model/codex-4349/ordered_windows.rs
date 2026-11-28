//! Matched Epoch 1 inputs for the frozen #4339 Butterfly-Rotor checkpoint.
//! Run the finite tests with `rustc --edition=2021 --test ordered_windows.rs`.
#![forbid(unsafe_code)]

pub const MAX_WINDOW_IDS: usize = 257;
pub const CHUNK_START_BIT: u16 = 1 << 15;
const BOS: u16 = 0;
const VOCABULARY: u16 = 4096;

/// The exact nonoverlapping windows in #4339 `build_training_chunks`.
/// A first ID conditions later predictions, even at a non-BOS chunk start.
pub fn epoch_one_windows(tokens: &[u16]) -> impl Iterator<Item = &[u16]> {
    let mut remaining = tokens;
    let mut chunks = [].chunks(MAX_WINDOW_IDS);
    std::iter::from_fn(move || loop {
        if let Some(window) = chunks.next() {
            return Some(window);
        }
        if remaining.is_empty() {
            return None;
        }
        let end = remaining[1..]
            .iter()
            .position(|&id| id == BOS)
            .map_or(remaining.len(), |i| i + 1);
        let (record, rest) = remaining.split_at(end);
        remaining = rest;
        chunks = record.chunks(MAX_WINDOW_IDS);
    })
    .filter(|window| window.len() >= 2)
}

/// Every target at an order has exactly that much history inside its window.
/// Order 0 still excludes each window's first ID and never predicts BOS.
pub fn ordered_rows(tokens: &[u16], order: usize) -> impl Iterator<Item = (&[u16], u16)> {
    epoch_one_windows(tokens).flat_map(move |window| {
        (order.max(1)..window.len()).map(move |at| (&window[at - order..at], window[at]))
    })
}

/// Mark each physical start out of band in a u16 for a prospective device
/// packer. The low 12 bits remain the original ID. Skipped one-ID windows
/// are omitted entirely. Reject invalid source IDs before emitting a tape.
pub fn mark_window_starts(tokens: &[u16]) -> Result<Vec<u16>, &'static str> {
    if tokens.iter().any(|&id| id >= VOCABULARY) {
        return Err("training stream contains an out-of-vocabulary original ID");
    }
    let mut marked = Vec::with_capacity(tokens.len());
    for window in epoch_one_windows(tokens) {
        let (&first, rest) = window.split_first().ok_or("missing window start")?;
        marked.push(first | CHUNK_START_BIT);
        marked.extend_from_slice(rest);
    }
    Ok(marked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    type GramCounts<'a> = BTreeMap<(&'a [u16], u16), usize>;

    fn counts(tokens: &[u16], order: usize) -> GramCounts<'_> {
        let mut result = BTreeMap::new();
        for gram in ordered_rows(tokens, order) {
            *result.entry(gram).or_default() += 1;
        }
        result
    }

    fn tagged_grams(marked: &[u16], order: usize) -> Vec<(Vec<u16>, u16)> {
        // Independent formulation for the CUDA selector: bounds on offsets,
        // skip marked targets, allow a mark in the oldest history slot only.
        let mut result = Vec::new();
        for at in order.max(1)..marked.len() {
            if marked[at] & CHUNK_START_BIT != 0 {
                continue;
            }
            let history = &marked[at - order..at];
            if history.iter().skip(1).any(|&id| id & CHUNK_START_BIT != 0) {
                continue;
            }
            result.push((
                history.iter().map(|&id| id & !CHUNK_START_BIT).collect(),
                marked[at],
            ));
        }
        result
    }

    #[test]
    fn causal_records_include_real_bos_and_eos_but_exclude_bos_targets() {
        let tokens = [0, 13, 1, 18, 0, 0, 17, 19, 1, 0];
        assert_eq!(
            epoch_one_windows(&tokens).collect::<Vec<_>>(),
            [&tokens[..4], &tokens[5..9]],
        );
        let order_zero = counts(&tokens, 0);
        assert_eq!(order_zero.values().sum::<usize>(), 6);
        assert_eq!(order_zero.get(&(&[], 1)), Some(&2));
        assert!(!order_zero.contains_key(&(&[], 0)));

        let order_one = counts(&tokens, 1);
        assert_eq!(order_one.get(&(&[0], 13)), Some(&1));
        assert_eq!(order_one.get(&(&[0], 17)), Some(&1));
        assert_eq!(order_one.get(&(&[1], 18)), Some(&1));
        assert!(!order_one.contains_key(&(&[18], 0)));
        assert!(!counts(&tokens, 2).contains_key(&(&[18, 0], 17)));
    }

    #[test]
    fn chunk_seam_preserves_original_context_and_removes_one_target() {
        let mut tokens = vec![0];
        tokens.extend(10..=265);
        tokens.extend([1000, 1001, 1002, 1, 0, 27, 1]);
        let windows = epoch_one_windows(&tokens).collect::<Vec<_>>();
        assert_eq!(
            windows.iter().map(|s| s.len()).collect::<Vec<_>>(),
            [257, 4, 3]
        );
        assert_eq!(counts(&tokens, 0).values().sum::<usize>(), 261);
        assert!(!counts(&tokens, 0).contains_key(&(&[], 1000)));
        assert!(!counts(&tokens, 1).contains_key(&(&[265], 1000)));
        assert_eq!(counts(&tokens, 1).get(&(&[1000], 1001)), Some(&1));
        assert_eq!(counts(&tokens, 2).get(&(&[1000, 1001], 1002)), Some(&1));
        assert!(!counts(&tokens, 2).contains_key(&(&[265, 1000], 1001)));
        assert!(!counts(&tokens, 1).contains_key(&(&[0], 1001)));
    }

    #[test]
    fn single_id_chunk_tail_provides_neither_context_nor_target() {
        let mut tokens = vec![0];
        tokens.extend([61; 256]);
        tokens.extend([1, 0, 62, 1]);
        assert_eq!(
            epoch_one_windows(&tokens).collect::<Vec<_>>(),
            [&tokens[..257], &tokens[258..]]
        );
        assert_eq!(counts(&tokens, 0).get(&(&[], 1)), Some(&1));
        assert!(!counts(&tokens, 1).contains_key(&(&[61], 1)));
        let marked = mark_window_starts(&tokens).unwrap();
        assert_eq!(marked.len(), tokens.len() - 1);
        assert_eq!(marked[257], CHUNK_START_BIT);
    }

    #[test]
    fn device_tape_preserves_every_window_gram_through_order_ten() {
        let mut tokens = vec![0];
        tokens.extend((1..=520).map(|i| 7 + i % 23));
        tokens.extend([0, 1, 0, 67, 1, 0, 0, 22, 1]);
        let marked = mark_window_starts(&tokens).unwrap();
        for order in 0..=10 {
            let expected = ordered_rows(&tokens, order)
                .map(|(prefix, target)| (prefix.to_vec(), target))
                .collect::<Vec<_>>();
            assert_eq!(tagged_grams(&marked, order), expected, "order {order}");
        }
        assert!(mark_window_starts(&[0, CHUNK_START_BIT | 55, 1]).is_err());
        assert!(mark_window_starts(&[0, 4096, 1]).is_err());
        assert!(mark_window_starts(&[]).unwrap().is_empty());
    }

    #[test]
    fn continuation_counts_use_only_observed_left_ids() {
        let mut tokens = vec![0];
        tokens.extend([61; 256]);
        tokens.extend([1000, 1001, 1]);
        tokens.extend([0, 31, 1000, 1001, 1]);
        tokens.extend([0, 32, 1000, 1001, 1]);
        tokens.extend([0, 31, 1000, 1001, 1]);
        let order_one = counts(&tokens, 1);
        let order_two = counts(&tokens, 2);
        let observed_left_types = |context, target| {
            order_two
                .keys()
                .filter(|(history, next)| history[1] == context && *next == target)
                .count()
        };
        assert_eq!(order_one.get(&(&[1000], 1001)), Some(&4));
        assert_eq!(observed_left_types(1000, 1001), 2);
        assert_eq!(order_one.get(&(&[0], 31)), Some(&2));
        assert_eq!(observed_left_types(0, 31), 0);

        let seam_only = &tokens[..260];
        assert_eq!(counts(seam_only, 1).get(&(&[1000], 1001)), Some(&1));
        assert!(!counts(seam_only, 2)
            .keys()
            .any(|(history, next)| history[1] == 1000 && *next == 1001));
    }
}
