use std::collections::BTreeMap;

pub const BANKS: usize = 16;
pub const RELATION_BANK: usize = 10;
pub type Keys = [u16; BANKS];
pub type Contrast = BTreeMap<u16, i8>;

/// The sparse coefficient vector of target minus rival. Bank address ranges
/// are disjoint, so two different banks never contribute to the same word.
pub fn difference(target: &Keys, rival: &Keys, banks: &[usize]) -> Contrast {
    let mut result = BTreeMap::new();
    for &bank in banks {
        if target[bank] != rival[bank] {
            result.insert(target[bank], 1);
            result.insert(rival[bank], -1);
        }
    }
    result
}

pub fn cancel(a: &Contrast, b: &Contrast) -> bool {
    a.len() == b.len() && a.iter().all(|(word, &sign)| b.get(word) == Some(&-sign))
}

/// Gordan's alternative: strict simultaneous positive margins exist exactly
/// when no nonzero, nonnegative combination of the signed rows is zero.
/// A support-minimal witness has positive coefficients and is affinely
/// independent. Enumerate supports of up to four rows and test the unique
/// solution of sum(coefficient * row) = 0, sum(coefficient) = 1.
pub fn obstruction(rows: &[&Contrast]) -> Option<Vec<usize>> {
    assert!(
        rows.len() <= 4,
        "structural probe supports at most four wordings"
    );
    let patterns = 1_u32..(1 << rows.len());
    for width in 1..=rows.len() {
        for pattern in patterns
            .clone()
            .filter(|p| p.count_ones() as usize == width)
        {
            let support: Vec<_> = (0..rows.len())
                .filter(|&index| pattern & (1 << index) != 0)
                .collect();
            if convex_combination(rows, &support) {
                return Some(support);
            }
        }
    }
    None
}

fn convex_combination(rows: &[&Contrast], support: &[usize]) -> bool {
    // Each coordinate touches at most four wording rows. A fifth entry is
    // the right-hand side, and the sole nonzero RHS comes from normalization.
    let mut by_word: BTreeMap<u16, [i64; 5]> = BTreeMap::new();
    for (index, &row) in support.iter().enumerate() {
        for (&word, &sign) in rows[row] {
            by_word.entry(word).or_insert([0; 5])[index] = i64::from(sign);
        }
    }
    let mut normalizing = [0; 5];
    normalizing[..support.len()].fill(1);
    normalizing[4] = 1;
    let mut pivots: [Option<[i64; 5]>; 4] = [None; 4];
    for mut eq in std::iter::once(normalizing).chain(by_word.into_values()) {
        let mut inserted = false;
        for index in 0..support.len() {
            if eq[index] == 0 {
                continue;
            }
            if let Some(pivot) = pivots[index] {
                eliminate(&mut eq, &pivot, index);
            } else {
                // Keep each accumulated equation diagonal in the columns
                // where a pivot exists, so its solution sign is visible.
                for (later, pivot) in pivots.iter().enumerate().skip(index + 1) {
                    if let Some(pivot) = pivot
                        && eq[later] != 0
                    {
                        eliminate(&mut eq, pivot, later);
                    }
                }
                for previous in pivots[..index].iter_mut().flatten() {
                    if previous[index] != 0 {
                        eliminate(previous, &eq, index);
                    }
                }
                pivots[index] = Some(eq);
                inserted = true;
                break;
            }
        }
        if !inserted && eq[4] != 0 {
            return false;
        }
    }
    pivots[..support.len()]
        .iter()
        .enumerate()
        .all(|(index, pivot)| {
            pivot.is_some_and(|p| p[4] != 0 && p[index].signum() == p[4].signum())
        })
}

fn eliminate(row: &mut [i64; 5], pivot: &[i64; 5], column: usize) {
    let (a, b) = (pivot[column], row[column]);
    for (term, &coefficient) in row.iter_mut().zip(pivot) {
        *term = a * *term - b * coefficient;
    }
    let divisor = row.iter().fold(0, |acc, &term| gcd(acc, term.abs()));
    if divisor > 1 {
        for term in row {
            *term /= divisor;
        }
    }
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(from: u16, to: u16) -> Contrast {
        [(from, 1), (to, -1)].into()
    }

    #[test]
    fn identical_single_opening_has_a_one_row_certificate() {
        assert_eq!(obstruction(&[&Contrast::new(), &edge(0, 1)]), Some(vec![0]));
    }

    #[test]
    fn opposite_only_when_signed_terms_cancel() {
        let forward = edge(0, 1);
        let reverse = edge(1, 0);
        let different = edge(1, 2);
        assert!(cancel(&forward, &reverse));
        assert!(!cancel(&forward, &different));
        assert_eq!(obstruction(&[&forward, &reverse]), Some(vec![0, 1]));
        assert_eq!(obstruction(&[&forward, &different]), None);
    }

    #[test]
    fn detects_three_and_four_row_obstructions_after_all_pairs_pass() {
        let a = edge(0, 1);
        let b = edge(1, 2);
        let c = edge(2, 3);
        let d = edge(3, 0);
        let e = edge(2, 0);
        assert_eq!(obstruction(&[&a, &b, &c]), None);
        assert_eq!(obstruction(&[&a, &b, &e]), Some(vec![0, 1, 2]));
        assert_eq!(obstruction(&[&a, &b, &c, &d]), Some(vec![0, 1, 2, 3]));
        let weighted = [(0, -1), (1, 1), (2, -1), (3, 1)].into();
        let a = [(0, 1), (1, -1), (2, 1), (3, -1), (4, 1), (5, -1)].into();
        let b = [(0, 1), (1, -1), (2, 1), (3, -1), (5, 1), (4, -1)].into();
        assert_eq!(obstruction(&[&weighted, &a, &b]), Some(vec![0, 1, 2]));
    }

    #[test]
    fn multiple_wordings_with_equal_contrasts_do_not_create_an_obstruction() {
        let a = edge(0, 1);
        let b = edge(1, 2);
        assert_eq!(obstruction(&[&a, &a, &b, &b]), None);
    }
}
