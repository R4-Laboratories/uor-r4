#![forbid(unsafe_code)]

use std::fs::{create_dir, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

const LEXICAL: usize = 96;
const TRANSPORT: usize = 32;
const LEXICAL_SLOTS: usize = 32;
const TRANSPORT_SLOTS: usize = 8;
const WORDS: [[u32; 4]; 2] = [[0, 4, 0, 1], [24, 28, 6, 7]];
const SITES: [[u32; 4]; 5] = [
    [4, 8, 1, 2],
    [8, 12, 2, 3],
    [12, 16, 3, 4],
    [16, 20, 4, 5],
    [20, 24, 5, 6],
];
const TRIPLES: [[u32; 3]; 4] = [[0, 2, 4], [1, 3, 4], [0, 3, 4], [1, 2, 4]];
const UP: i16 = i16::MAX;
const DOWN: i16 = i16::MIN;
const PARENT: [[i16; 4]; 2] = [[0, 0, UP - 1, DOWN + 1], [0; 4]];
const STEP: [[i16; 4]; 2] = [[-1, 1, UP, DOWN], [1, 0, 0, 0]];

type Output = BufWriter<File>;

fn create(root: &Path, name: &str) -> io::Result<Output> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(name))
        .map(BufWriter::new)
}

fn words(out: &mut Output, values: &[u32]) -> io::Result<()> {
    for value in values {
        out.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn point(out: &mut Output, prefix: &[i16], dimensions: usize) -> io::Result<()> {
    for value in prefix {
        out.write_all(&value.to_le_bytes())?;
    }
    for _ in prefix.len()..dimensions {
        out.write_all(&0i16.to_le_bytes())?;
    }
    Ok(())
}

fn packet(root: &Path) -> io::Result<()> {
    let mut input = create(root, "text.bin")?;
    input.write_all(b"UORTXT01")?;
    words(&mut input, &[6])?;
    let mut labels = create(root, "labels.bin")?;
    labels.write_all(b"UORLNK01")?;
    words(&mut labels, &[6])?;

    // Each group is a two-endpoint pair. Axis 0 improves the far edges safely.
    // Axis 1 improves the far edges, loses the first near pair and gains the second.
    let first_near = [0, -1, UP - 1, DOWN + 1];
    let second_near = [0, 2, UP - 1, DOWN + 1];
    for (affirmed, excluded) in [
        ([[-2, 2, UP, DOWN], [1, -1, UP - 2, DOWN + 2]], [2, -1]),
        ([first_near, second_near], [0, 20]),
        ([second_near, first_near], [0, 20]),
    ] {
        for _ in 0..2 {
            words(&mut input, &[2, 5, TRIPLES.len() as u32])?;
            for (address, coordinate) in WORDS.iter().zip([0, 10]) {
                words(&mut input, address)?;
                point(&mut input, &[coordinate], LEXICAL)?;
            }
            for (address, coordinate) in SITES.iter().zip([0, 0, 3, 4, 5]) {
                words(&mut input, address)?;
                point(&mut input, &[coordinate], LEXICAL)?;
            }
            for edge in affirmed {
                point(&mut input, &edge, TRANSPORT)?;
            }
            for _ in 2..SITES.len() {
                point(&mut input, &[], TRANSPORT)?;
            }
            for value in [0, 0, excluded[0], excluded[1], 0] {
                point(&mut input, &[value], TRANSPORT)?;
            }
            for tuple in &TRIPLES {
                words(&mut input, tuple)?;
            }
            words(&mut labels, &[0, 0, 1, 2])?;
        }
    }
    input.flush()?;
    labels.flush()
}

fn state(root: &Path, name: &str, coordinates: [[i16; 4]; 2]) -> io::Result<()> {
    let mut out = create(root, name)?;
    out.write_all(b"UORSTT01")?;
    for (word, transport) in [0, 10].into_iter().zip(coordinates) {
        words(&mut out, &[1])?;
        point(&mut out, &[word], LEXICAL)?;
        for _ in 1..LEXICAL_SLOTS {
            point(&mut out, &[], LEXICAL)?;
        }
        words(&mut out, &[1])?;
        point(&mut out, &transport, TRANSPORT)?;
        for _ in 1..TRANSPORT_SLOTS {
            point(&mut out, &[], TRANSPORT)?;
        }
    }
    out.flush()
}

fn main() -> io::Result<()> {
    let directory = std::env::args_os()
        .nth(1)
        .ok_or_else(|| io::Error::other("usage: fixture NEW_DIRECTORY"))?;
    let root = Path::new(&directory);
    create_dir(root)?;
    packet(root)?;
    state(root, "parent.state", PARENT)?;
    state(root, "expected-step-1.state", STEP)?;
    for (id, sign, axis, value) in [
        (0, 0, 0, -1),
        (1, 0, 1, 1),
        (2, 0, 2, UP),
        (3, 0, 3, DOWN),
        (4, 1, 0, 1),
    ] {
        let mut expected = PARENT;
        expected[sign][axis] = value;
        state(root, &format!("expected-{id}.state"), expected)?;
    }
    state(root, "expected-5.state", [STEP[0], PARENT[1]])?;
    state(
        root,
        "expected-saturated-center.state",
        [[-2, 2, UP, DOWN], [0; 4]],
    )
}
