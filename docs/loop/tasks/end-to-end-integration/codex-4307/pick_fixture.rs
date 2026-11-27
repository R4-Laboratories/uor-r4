//! Finite public vectors for comparing implementations of the UOR-R4 Pick operator.
//! The expected fields are occurrence index (zero-based), distance, tie count,
//! successor index, and exact original successor token. No model weights are loaded.
#![forbid(unsafe_code)]

#[path = "../../../../../crates/uor-r4-discrete/src/kn_ceiling/runtime/code_pick.rs"]
mod reference;

use std::hint::black_box;

type ExpectedRead = (usize, u32, u32, usize, u16);

fn main() {
    let mut codes = [[0_u16; 4]; 14];
    codes[10] = [85, 170, 51, 204];
    codes[11] = [170, 85, 204, 51];
    codes[12] = [51, 204, 85, 170];
    codes[13] = [204, 51, 170, 85];

    let cases: &[(&[u16], Option<ExpectedRead>)] = &[
        (&[10, 11], None),
        (&[10, 11, 12, 13, 10, 11], Some((1, 0, 1, 2, 12))),
        (&[10, 11, 12, 10, 11, 13, 10, 11], Some((4, 0, 2, 5, 13))),
        (&[10, 11, 12, 10, 11, 12, 10, 11], Some((4, 0, 2, 5, 12))),
        (&[10, 11, 12, 13], Some((1, 32, 1, 2, 12))),
    ];
    for &(prefix, expected) in cases {
        let read = reference::code_pick(black_box(&codes), black_box(prefix));
        let observed =
            read.map(|(site, distance, ties)| (site, distance, ties, site + 1, prefix[site + 1]));
        assert_eq!(observed, expected);
    }
}
