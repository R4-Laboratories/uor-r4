#![forbid(unsafe_code)]

use std::{fs, io, path::PathBuf};

const MAGIC: usize = 8;
const LEX: usize = 4 + 32 * 96 * 2;
const TRANSPORT_SLOTS: usize = 8;
const TRANSPORT_AXES: usize = 32;
const SIGN_BLOCK: usize = LEX + 4 + TRANSPORT_SLOTS * TRANSPORT_AXES * 2;
const TAIL: usize = 12;
const FIRST_AXIS: usize = TRANSPORT_AXES - TAIL;
const KINDS: usize = 2 * (TAIL + 1);
const POSITIONS: usize = 2 * TRANSPORT_SLOTS * KINDS;

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn state(path: &PathBuf) -> io::Result<Vec<u8>> {
    let data = fs::read(path)?;
    if data.len() != MAGIC + 2 * SIGN_BLOCK || !data.starts_with(b"UORSTT01") {
        return Err(invalid("incorrect native state shape"));
    }
    Ok(data)
}

fn axis_value(state: &[u8], start: usize, axis: usize) -> i16 {
    let offset = start + 2 * axis;
    i16::from_le_bytes([state[offset], state[offset + 1]])
}

fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let [seed, parent, result, id] = args.as_slice() else {
        return Err(invalid(
            "usage: projected-state SEED PARENT RESULT projection-ID",
        ));
    };
    let id = id
        .to_str()
        .and_then(|s| s.strip_prefix("projection-"))
        .and_then(|id| id.parse::<usize>().ok())
        .filter(|&id| id < POSITIONS)
        .ok_or_else(|| invalid("invalid projection position"))?;

    let seed = state(seed)?;
    let parent = state(parent)?;
    let result = state(result)?;
    let sign = id / (TRANSPORT_SLOTS * KINDS);
    let slot = (id / KINDS) % TRANSPORT_SLOTS;
    let axis = (id / 2) % (TAIL + 1);
    let bank = MAGIC + sign * SIGN_BLOCK + LEX;
    let count = u32::from_le_bytes(
        seed[bank..bank + 4]
            .try_into()
            .map_err(|_| invalid("count"))?,
    );
    if slot >= count as usize || count as usize > TRANSPORT_SLOTS {
        return Err(invalid("projection names an inactive slot"));
    }
    let mut candidate = seed.clone();
    let start = bank + 4 + slot * TRANSPORT_AXES * 2;
    let mut moved = 0;
    for coordinate in 0..TAIL {
        if axis != TAIL && axis != coordinate {
            continue;
        }
        let coordinate = FIRST_AXIS + coordinate;
        let old = axis_value(&seed, start, coordinate);
        let new = axis_value(&parent, start, coordinate);
        let value = if id % 2 == 0 {
            new
        } else {
            old + i16::from(new > old) - i16::from(new < old)
        };
        let offset = start + 2 * coordinate;
        candidate[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
        moved += usize::from(value != old);
    }
    if moved == 0 || candidate != result {
        return Err(invalid("projection differs from its parent delta"));
    }
    let selected_axis = if axis == TAIL {
        "null".to_owned()
    } else {
        (FIRST_AXIS + axis).to_string()
    };
    println!(
        "{{\"sign\":{},\"slot\":{slot},\"axis\":{selected_axis},\"unit_step\":{},\"changed_coordinates\":{moved}}}",
        if sign == 0 { 1 } else { -1 },
        id % 2 != 0,
    );
    Ok(())
}
