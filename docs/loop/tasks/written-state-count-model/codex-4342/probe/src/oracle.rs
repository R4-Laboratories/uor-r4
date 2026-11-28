use std::path::Path;

use uor_r4_integer::discrete_d11::{IntegerDiscreteModel, native_whole_turn};

use crate::{Result, check, digest};

const D11: &str = include_str!("../../../../../../../crates/uor-r4-integer/src/discrete_d11.rs");
const AXES_SHA256: &str = "1e47785f5491692f3c977e805ac09126d9374a7e41b508d0b7eec304d7f91d82";
const BFS_SHA256: &str = "a25ab88a801890618d5e61c892ff8aaf778606ef4f7f43cd288a70d65243fc51";

pub struct Roots {
    pub context: usize,
    packed: Vec<u32>,
}

impl Roots {
    pub fn from_base(base: &Path, tokenizer_sha256: &str) -> Result<Self> {
        let model = IntegerDiscreteModel::load(base)?;
        check(
            model.artifact_sha256() == native_whole_turn::BASE_SHA256,
            "capture does not use the registered bare base artifact",
        )?;
        check(
            model.tokenizer_sha256() == Some(tokenizer_sha256),
            "base artifact and capture tokenizer pins differ",
        )?;
        Ok(Self {
            context: model.shape().context as usize,
            packed: model.packed_codebook_roots(),
        })
    }

    pub fn block_2(&self, id: u32) -> Option<u8> {
        self.packed
            .get(id as usize)
            .map(|&packed| ((packed >> 8) & 15) as u8)
    }
}

pub struct RelationOracle {
    bfs: [u8; 256],
    // Infer the stated relative action directly from signed permutations.
    // The independently pinned BFS chooses canonical representatives.
    relative: [[u8; 64]; 64],
}

fn table<const N: usize>(name: &str, expected: &str) -> Result<[u8; N]> {
    let declaration = format!("const {name}: [u8; {N}] = [");
    let (body, _) = D11
        .split_once(&declaration)
        .ok_or("missing canonical table declaration")?
        .1
        .split_once("];")
        .ok_or("unterminated canonical table")?;
    let values: Vec<u8> = body
        .split(',')
        .map(str::trim)
        .filter(|word| !word.is_empty())
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    check(
        digest(&values) == expected,
        "canonical D11 table digest differs",
    )?;
    values
        .try_into()
        .map_err(|_| "invalid canonical table width".into())
}

impl RelationOracle {
    pub fn new() -> Result<Self> {
        let bytes: [u8; 512] = table("AXES", AXES_SHA256)?;
        let mut axes = [[0u8; 8]; 64];
        for (axis, values) in axes.iter_mut().zip(bytes.chunks_exact(8)) {
            axis.copy_from_slice(values);
        }
        let mut relative = [[0u8; 64]; 64];
        for (query, row) in relative.iter_mut().enumerate() {
            for (source, slot) in row.iter_mut().enumerate() {
                // Solve query ◦ relative = source by matching all eight
                // signed axes. This avoids using production's inverse or
                // composition table when checking either orientation.
                let mut possibilities = axes.iter().enumerate().filter(|(_, action)| {
                    axes[query].iter().enumerate().all(|(out, &code)| {
                        action[usize::from(code & 7)] ^ (code & 8) == axes[source][out]
                    })
                });
                let (action, _) = possibilities
                    .next()
                    .ok_or("relative signed-axis state missing")?;
                check(
                    possibilities.next().is_none(),
                    "ambiguous relative signed-axis state",
                )?;
                *slot = action as u8 + 1;
            }
        }
        Ok(Self {
            bfs: table("BFS", BFS_SHA256)?,
            relative,
        })
    }

    pub fn geometry_slot(&self, [ql, qr, sl, sr]: [u8; 4], swap: bool) -> u8 {
        let query = self.bfs[usize::from(ql) * 16 + usize::from(qr)];
        let [left, right] = if swap { [sr, sl] } else { [sl, sr] };
        let source = self.bfs[usize::from(left) * 16 + usize::from(right)];
        self.relative[query as usize][source as usize]
    }

    pub fn count_slot([a, b, c, d]: [u32; 4]) -> u8 {
        let raw = [
            a.to_le_bytes(),
            b.to_le_bytes(),
            c.to_le_bytes(),
            d.to_le_bytes(),
        ]
        .concat();
        let low = raw
            .into_iter()
            .fold(0xC5_u8, |acc, byte| (acc ^ byte).wrapping_mul(0x93));
        // Only the lowest six bits of FNV-1a32 influence the slot.
        1 + (low & 63)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_axis_oracle_has_exact_range_and_noncommuting_examples() -> Result<()> {
        let oracle = RelationOracle::new()?;
        let mut changes = 0;
        let mut slots = [false; 65];
        for ql in 0..16 {
            for qr in 0..16 {
                for sl in 0..16 {
                    for sr in 0..16 {
                        let forward = oracle.geometry_slot([ql, qr, sl, sr], false);
                        let swapped = oracle.geometry_slot([ql, qr, sl, sr], true);
                        slots[forward as usize] = true;
                        changes += usize::from(forward != swapped);
                    }
                }
            }
        }
        assert!(!slots[0]);
        assert!(slots[1..].iter().all(|&used| used));
        assert!(changes > 0);
        Ok(())
    }

    #[test]
    fn count_control_has_equal_arity_and_ordered_bytes() {
        assert_eq!(RelationOracle::count_slot([1, 2, 3, 4]), 2);
        assert_eq!(RelationOracle::count_slot([1, 3, 2, 4]), 18);
        assert_eq!(RelationOracle::count_slot([1, 256, 3, 1024]), 33);
    }
}
