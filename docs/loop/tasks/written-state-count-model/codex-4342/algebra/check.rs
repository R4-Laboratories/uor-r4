// The driver prefixes the unmodified production tables to this finite test.
use std::collections::BTreeSet;

fn compose(a: usize, b: usize) -> usize {
    COMPOSE[a * 64 + b] as usize
}

fn apply(state: usize, vector: [i8; 8]) -> [i8; 8] {
    std::array::from_fn(|axis| {
        let code = AXES[8 * state + axis] as usize;
        let component = vector[code & 7];
        if code & 8 == 0 {
            component
        } else {
            -component
        }
    })
}

fn path(left: usize, right: usize) -> usize {
    BFS[left * 16 + right] as usize
}

fn main() {
    let states = 0..64;
    let inverse: [usize; 64] = std::array::from_fn(|state| {
        let inverses: Vec<_> = states
            .clone()
            .filter(|&other| compose(state, other) == 0 && compose(other, state) == 0)
            .collect();
        assert_eq!(inverses.len(), 1);
        assert_eq!(inverses[0], compose(state, compose(state, state)));
        inverses[0]
    });
    // All distinct nonzero components distinguish permutations and their signs.
    let vector = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut commuting = 0;
    for a in states.clone() {
        for b in states.clone() {
            assert_eq!(apply(compose(a, b), vector), apply(a, apply(b, vector)));
            commuting += usize::from(compose(a, b) == compose(b, a));
        }
    }
    assert!(commuting < 64 * 64);
    let mut reverse_is_inverse = 0;
    let mut rooted_paths = BTreeSet::new();
    for a in 0..16 {
        for b in 0..16 {
            assert_eq!(apply(path(a, b), ROOTS[a]), ROOTS[b]);
            reverse_is_inverse += usize::from(path(b, a) == inverse[path(a, b)]);
            rooted_paths.insert(path(a, b));
        }
    }
    let mut slots = BTreeSet::new();
    let mut source_swaps_change = 0;
    let mut query_swaps_change = 0;
    let mut composition_order_changes = 0;
    for a in 0..16 {
        for b in 0..16 {
            for c in 0..16 {
                for d in 0..16 {
                    let query = path(a, b);
                    let source = path(c, d);
                    let relative = compose(inverse[query], source);
                    assert_eq!(compose(query, relative), source);
                    slots.insert(relative + 1);
                    source_swaps_change +=
                        usize::from(relative != compose(inverse[query], path(d, c)));
                    query_swaps_change +=
                        usize::from(relative != compose(inverse[path(b, a)], source));
                    composition_order_changes +=
                        usize::from(relative != compose(source, inverse[query]));
                }
            }
        }
    }
    assert!(source_swaps_change > 0 && query_swaps_change > 0 && composition_order_changes > 0);
    println!("{{\"ordered_root_pairs\":256,\"quadruples\":65536,\"commuting_out_of_4096\":{commuting},\"distinct_bfs_states\":{},\"reverse_is_inverse_pairs\":{reverse_is_inverse},\"query_order_changed\":{query_swaps_change},\"source_order_changed\":{source_swaps_change},\"composition_order_changed\":{composition_order_changes},\"realized_slots\":{:?}}}",rooted_paths.len(),slots.into_iter().collect::<Vec<_>>());
}
