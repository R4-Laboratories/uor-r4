#![forbid(unsafe_code)]

use std::fs::{create_dir, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

const LEXICAL: usize = 96;
const TRANSPORT: usize = 32;
const WORDS: [([u32; 4], i16); 4] = [
    ([0, 4, 0, 1], -15),
    ([5, 9, 1, 2], -16),
    ([10, 14, 2, 3], 16),
    ([15, 19, 3, 4], 0),
];
const SITES: [[u32; 4]; 8] = [
    [20, 24, 4, 5],
    [25, 29, 5, 6],
    [30, 34, 6, 7],
    [35, 39, 7, 8],
    [40, 44, 8, 9],
    [45, 49, 9, 10],
    [50, 54, 10, 11],
    [10, 14, 2, 3],
];
const TRIPLES: [[u32; 3]; 5] = [[0, 3, 5], [1, 4, 5], [2, 4, 5], [1, 4, 6], [5, 7, 2]];
const TRANSPORT_COORDINATES: [[i16; 8]; 4] = [
    [0, 1, 4, 0, 0, 4, 0, 0],
    [5, 4, 0, 0, 0, 5, 0, 0],
    [0, 0, 0, -6, 1, 0, 0, 0],
    [0; 8],
];

type Output = BufWriter<File>;

fn create(root: &Path, name: &str) -> io::Result<Output> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(name))
        .map(BufWriter::new)
}

fn integers(out: &mut Output, values: &[u32]) -> io::Result<()> {
    for value in values {
        out.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn point(out: &mut Output, axis: i16, dimensions: usize) -> io::Result<()> {
    out.write_all(&axis.to_le_bytes())?;
    for _ in 1..dimensions {
        out.write_all(&0i16.to_le_bytes())?;
    }
    Ok(())
}

fn packet(root: &Path) -> io::Result<()> {
    let mut packet = create(root, "text.bin")?;
    packet.write_all(b"UORTXT01")?;
    integers(&mut packet, &[3])?;
    for (heading, triples) in [
        (0, TRIPLES.as_slice()),
        (1, TRIPLES.as_slice()),
        (0, &TRIPLES[2..3]),
    ] {
        integers(
            &mut packet,
            &[WORDS.len() as u32, SITES.len() as u32, triples.len() as u32],
        )?;
        for (address, axis) in WORDS {
            integers(&mut packet, &address)?;
            point(&mut packet, axis, LEXICAL)?;
        }
        for (address, axis) in SITES.iter().zip([0, 0, 0, 0, 0, 0, heading, 0]) {
            integers(&mut packet, address)?;
            point(&mut packet, axis, LEXICAL)?;
        }
        for axis in TRANSPORT_COORDINATES.into_iter().flatten() {
            point(&mut packet, axis, TRANSPORT)?;
        }
        for triple in triples {
            integers(&mut packet, triple)?;
        }
    }
    packet.flush()?;

    let mut requests = create(root, "requests.tsv")?;
    for (row, sign, word, requested, rival) in [
        (1, -1, 2, 3, 4),
        (0, 1, 0, 0, 1),
        (0, 1, 0, 0, 5),
        (0, 1, 3, 0, 1),
        (1, -1, 2, 7, 4),
        (1, 1, 0, 6, 1),
        (0, 1, 0, 2, 5),
        (1, 1, 0, 1, 0),
        (2, 1, 0, 2, 0),
    ] {
        writeln!(requests, "{row}\t{sign}\t{word}\t{requested}\t{rival}")?;
    }
    requests.flush()
}

fn frozen(root: &Path) -> io::Result<()> {
    let mut fit = create(root, "fixture.words")?;
    integers(&mut fit, &[1])?;
    for _ in 0..9 {
        for _ in 0..7 {
            integers(&mut fit, &[1, 0])?;
            point(&mut fit, 0, LEXICAL)?;
        }
    }
    fit.flush()?;

    let mut state = create(root, "fixture.state")?;
    state.write_all(b"UORSTT01")?;
    for axis in [-16, 16] {
        integers(&mut state, &[1])?;
        point(&mut state, axis, LEXICAL)?;
        for _ in 1..32 {
            point(&mut state, 0, LEXICAL)?;
        }
        integers(&mut state, &[2])?;
        point(&mut state, 0, TRANSPORT)?;
        point(&mut state, 8, TRANSPORT)?;
        for _ in 2..8 {
            point(&mut state, 0, TRANSPORT)?;
        }
    }
    state.flush()
}

fn main() -> io::Result<()> {
    let root = std::env::args_os()
        .nth(1)
        .ok_or_else(|| io::Error::other("missing fixture output directory"))?;
    let root = Path::new(&root);
    create_dir(root)?;
    packet(root)?;
    frozen(root)?;
    let mut prompt = create(root, "fixture-text.txt")?;
    prompt.write_all(b"kept keep drop hold fern fern moss pine pine here near\n")?;
    prompt.flush()
}
