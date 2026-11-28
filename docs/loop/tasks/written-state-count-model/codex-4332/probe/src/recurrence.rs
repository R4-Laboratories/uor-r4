use sha2::{Digest, Sha256};
use uor_r4_integer::discrete_d11::{spiralcore_orbit_roots, spiralcore_signed_axes};

use crate::Result;

const VOCABULARY: usize = 4096;
const ROOT_OFFSET: usize = 70 + VOCABULARY;
const SECTION_SHA256: &str = "0fc117f37a9fb8e92f9b3934a5e4ee0568848df440b312f8d0e2b737b2b62d40";

/// Independent fixture oracle for the verified #4314 flat trunk. Apply the
/// signed axes directly to coordinates, without the product composition table.
pub struct Recurrence {
    section: Vec<u32>,
}

impl Recurrence {
    pub fn new(bytes: &[u8], product_roots: &[u32]) -> Result<Self> {
        if bytes.len() != (ROOT_OFFSET + VOCABULARY) * 4
            || hex::encode(Sha256::digest(bytes)) != SECTION_SHA256
        {
            return Err("unexpected #4314 flat trunk artifact".into());
        }
        let section: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if section[0] != 1
            || section[1..69].iter().any(|&word| word >= 16)
            || &section[ROOT_OFFSET..] != product_roots
        {
            return Err("flat trunk roots or gate shape differ from the product".into());
        }
        Ok(Self { section })
    }

    fn root(&self, token: u32, block: usize) -> usize {
        ((self.section[ROOT_OFFSET + token as usize] >> (4 * block)) & 15) as usize
    }

    pub fn code(&self, prefix: &[u32]) -> Result<u16> {
        let last = prefix.len().checked_sub(1).ok_or("empty oracle prefix")?;
        if prefix.iter().any(|&id| id as usize >= VOCABULARY) {
            return Err("oracle prefix contains a token outside the frozen vocabulary".into());
        }
        let roots = spiralcore_orbit_roots();
        let mut code = 0;
        for block in 0..4 {
            let coordinate = |at| roots[self.root(prefix[at], block)].map(i32::from);
            let transport = |start, vector| {
                prefix[start..].iter().fold(vector, |state, &token| {
                    let gate = self.section[1 + 16 * block + self.root(token, block)] as usize;
                    action(gate, state)
                })
            };

            let previous = last.saturating_sub(1);
            let middle = last / 2;
            let vectors = [
                coordinate(last),
                transport(previous + 1, coordinate(previous)),
                transport(middle + 1, coordinate(middle)),
                transport(0, coordinate(0)),
            ];
            let sum: [i32; 8] = std::array::from_fn(|axis| vectors.iter().map(|v| v[axis]).sum());
            let snapped = roots
                .iter()
                .map(|root| root.map(i32::from))
                .min_by_key(|root| {
                    let distances = std::array::from_fn::<_, 8, _>(|axis| {
                        (sum[axis] - 4 * root[axis]).unsigned_abs()
                    });
                    (
                        distances.iter().sum::<u32>(),
                        distances.iter().filter(|&&d| d != 0).count(),
                    )
                })
                .ok_or("missing native roots")?;
            let normalized = action(self.section[65 + block] as usize, snapped);
            let root = roots
                .iter()
                .position(|root| root.map(i32::from) == normalized)
                .ok_or("oracle normalization outside the orbit")?;
            code |= (root as u16) << (4 * block);
        }
        Ok(code)
    }
}

fn action(gate: usize, state: [i32; 8]) -> [i32; 8] {
    std::array::from_fn(|axis| {
        let signed_axis = spiralcore_signed_axes()[gate * 8 + axis];
        let coordinate = state[usize::from(signed_axis & 7)];
        if signed_axis & 8 == 0 {
            coordinate
        } else {
            -coordinate
        }
    })
}
