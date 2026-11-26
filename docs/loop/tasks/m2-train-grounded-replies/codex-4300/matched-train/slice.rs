#![forbid(unsafe_code)]

#[allow(dead_code)]
mod packet;

use std::{env, fs, io};

fn equal_occurrences(left: &[packet::Occurrence], right: &[packet::Occurrence]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.address.bytes == right.address.bytes
                && left.address.tokens == right.address.tokens
                && left.point == right.point
        })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = env::args().skip(1).collect();
    let [full, subset, indices] = arguments.as_slice() else {
        return Err("usage: slice FULL_PACKET SUBSET_PACKET ROW_INDICES".into());
    };
    let full = packet::text(&fs::read(full)?)?;
    let subset = packet::text(&fs::read(subset)?)?;
    let indices: Vec<_> = fs::read_to_string(indices)?
        .lines()
        .map(str::parse::<usize>)
        .collect::<Result<_, _>>()?;

    if subset.len() != 8
        || subset.len() != indices.len()
        || !indices.windows(2).all(|w| w[0] < w[1])
    {
        return Err("invalid canonical subset enrollment".into());
    }
    for (&index, row) in indices.iter().zip(subset.iter()) {
        let original = full
            .get(index)
            .ok_or_else(|| io::Error::other("packet subset index outside canonical source"))?;
        if !equal_occurrences(&row.words, &original.words)
            || !equal_occurrences(&row.sites, &original.sites)
            || row.edges != original.edges
            || row.tuples != original.tuples
        {
            return Err(format!("canonical row {index} changed in the packet subset").into());
        }
    }
    println!("{{\"rows\":8,\"all_packet_fields_identical\":true}}");
    Ok(())
}
