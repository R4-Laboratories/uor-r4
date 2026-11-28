use serde::Serialize;

pub const REFERENCE: usize = 16;
const VERTICES: usize = REFERENCE + 1;
const LIMIT: i32 = 4096;

/// score[to] - score[from] >= minimum. All variables are relative to REFERENCE.
#[derive(Clone, Debug, Serialize)]
pub struct Constraint {
    pub from: usize,
    pub to: usize,
    pub minimum: i32,
    pub why: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "result")]
pub enum Verdict {
    Feasible { active_words: Vec<i32> },
    Impossible { cycle: Vec<Constraint>, excess: i32 },
}

pub fn solve(required: &[Constraint]) -> Verdict {
    let mut edges = required.to_vec();
    for slot in 0..REFERENCE {
        for (from, to) in [(slot, REFERENCE), (REFERENCE, slot)] {
            edges.push(Constraint {
                from,
                to,
                minimum: -LIMIT,
                why: "signed table bound".into(),
            });
        }
    }
    let mut height = [0; VERTICES];
    let mut parents = [None; VERTICES];
    for iteration in 0..VERTICES {
        let mut changed = None;
        for (index, edge) in edges.iter().enumerate() {
            let lower = height[edge.from] + edge.minimum;
            if height[edge.to] < lower {
                height[edge.to] = lower;
                parents[edge.to] = Some(index);
                changed = Some(edge.to);
            }
        }
        let Some(mut tail) = changed else {
            let words: Vec<_> = height.iter().map(|v| v - height[REFERENCE]).collect();
            assert!(edges
                .iter()
                .all(|e| words[e.to] - words[e.from] >= e.minimum));
            return Verdict::Feasible {
                active_words: words[..REFERENCE].to_vec(),
            };
        };
        if iteration + 1 == VERTICES {
            let previous = |v: usize| &edges[parents[v].expect("updated vertex has predecessor")];
            for _ in 0..VERTICES {
                tail = previous(tail).from;
            }
            let start = tail;
            let mut cycle = Vec::new();
            loop {
                let edge = previous(tail);
                cycle.push(edge.clone());
                tail = edge.from;
                if tail == start {
                    break;
                }
            }
            cycle.reverse();
            let excess = cycle.iter().map(|e| e.minimum).sum();
            assert!(excess > 0);
            return Verdict::Impossible { cycle, excess };
        }
    }
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn edge(from: usize, to: usize, minimum: i32) -> Constraint {
        Constraint {
            from,
            to,
            minimum,
            why: "fixture".into(),
        }
    }
    #[test]
    fn two_questions_can_reverse_when_addresses_switch() {
        assert!(matches!(
            solve(&[edge(0, 1, 8), edge(0, 1, 8)]),
            Verdict::Feasible { .. }
        ));
        assert!(matches!(
            solve(&[edge(0, 1, 8), edge(1, 0, 8)]),
            Verdict::Impossible { excess: 16, .. }
        ));
    }
    #[test]
    fn honors_offsets_and_signed_bounds() {
        assert!(matches!(
            solve(&[edge(2, 2, -10)]),
            Verdict::Feasible { .. }
        ));
        assert!(matches!(
            solve(&[edge(2, 2, 1)]),
            Verdict::Impossible { excess: 1, .. }
        ));
        assert!(matches!(
            solve(&[edge(1, 3, 8192)]),
            Verdict::Feasible { .. }
        ));
        assert!(matches!(
            solve(&[edge(1, 3, 8193)]),
            Verdict::Impossible { excess: 1, .. }
        ));
        assert!(matches!(
            solve(&[edge(REFERENCE, 0, 4097)]),
            Verdict::Impossible { excess: 1, .. }
        ));
    }
}
