use std::fs;
#[cfg(not(recovery_audit))]
use std::fs::OpenOptions;
use std::io;
#[cfg(not(recovery_audit))]
use std::io::Write;
use std::ops::Range;
use std::path::Path;

pub const SLOTS: usize = 8;
pub const AXES: usize = 32;
pub const TAIL: usize = 20;
pub const SIGNS: usize = 2;
pub const STATE_BYTES: usize = 13_336;
pub const LEXICAL_BYTES: usize = 4 + 32 * 96 * 2;
const SIGN_BYTES: usize = LEXICAL_BYTES + 4 + SLOTS * AXES * 2;
const MAGIC_BYTES: usize = 8;

pub struct State(Box<[u8; STATE_BYTES]>);

pub fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

impl State {
    pub fn load(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?
            .into_boxed_slice()
            .try_into()
            .map_err(|_| invalid(format!("{} has the wrong UORSTT01 size", path.display())))?;
        let state = Self(bytes);
        if &state.0[..MAGIC_BYTES] != b"UORSTT01" {
            return Err(invalid(format!(
                "{} has the wrong native magic",
                path.display()
            )));
        }
        for sign in 0..SIGNS {
            let start = Self::sign_start(sign);
            if !(1..=32).contains(&state.count_at(start))
                || !(1..=SLOTS).contains(&state.active_slots(sign))
            {
                return Err(invalid(format!(
                    "{} has invalid center counts",
                    path.display()
                )));
            }
        }
        Ok(state)
    }

    #[cfg(not(recovery_audit))]
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let mut out = OpenOptions::new().write(true).create_new(true).open(path)?;
        out.write_all(self.0.as_ref())
    }

    pub fn copy(&self) -> Self {
        Self(self.0.clone())
    }

    fn sign_start(sign: usize) -> usize {
        MAGIC_BYTES + sign * SIGN_BYTES
    }

    fn count_at(&self, offset: usize) -> usize {
        let bytes = &self.0[offset..offset + 4];
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize
    }

    pub fn active_slots(&self, sign: usize) -> usize {
        self.count_at(Self::sign_start(sign) + LEXICAL_BYTES)
    }

    fn slot(sign: usize, slot: usize) -> Range<usize> {
        let start = Self::sign_start(sign) + LEXICAL_BYTES + 4 + slot * AXES * 2;
        start..start + AXES * 2
    }

    pub fn coordinate(&self, sign: usize, slot: usize, axis: usize) -> i16 {
        let start = Self::slot(sign, slot).start + axis * 2;
        i16::from_le_bytes([self.0[start], self.0[start + 1]])
    }

    #[cfg(recovery_audit)]
    pub fn set_coordinate(&mut self, sign: usize, slot: usize, axis: usize, coordinate: i16) {
        let start = Self::slot(sign, slot).start + axis * 2;
        self.0[start..start + 2].copy_from_slice(&coordinate.to_le_bytes());
    }

    pub fn replace_slot(&mut self, donor: &Self, sign: usize, slot: usize) -> io::Result<()> {
        if slot >= self.active_slots(sign) || slot >= donor.active_slots(sign) {
            return Err(invalid("whole-slot substitution outside active centers"));
        }
        let range = Self::slot(sign, slot);
        self.0[range.clone()].copy_from_slice(&donor.0[range]);
        Ok(())
    }

    pub fn same(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    pub fn verify_immutable(&self, candidate: &Self) -> io::Result<()> {
        let mut normalized = candidate.copy();
        for sign in 0..SIGNS {
            for slot in 0..self.active_slots(sign) {
                let range = Self::slot(sign, slot);
                let tail = range.start + TAIL * 2..range.end;
                normalized.0[tail.clone()].copy_from_slice(&self.0[tail]);
            }
        }
        if !self.same(&normalized) {
            return Err(invalid(
                "native bytes changed outside active signed tail coordinates",
            ));
        }
        Ok(())
    }

    pub fn tail_distance(&self, other: &Self) -> (usize, u32) {
        let mut changed = 0;
        let mut l1 = 0;
        for sign in 0..SIGNS {
            for slot in 0..self.active_slots(sign) {
                for axis in TAIL..AXES {
                    let distance = self
                        .coordinate(sign, slot, axis)
                        .abs_diff(other.coordinate(sign, slot, axis));
                    changed += usize::from(distance > 0);
                    l1 += u32::from(distance);
                }
            }
        }
        (changed, l1)
    }
}
