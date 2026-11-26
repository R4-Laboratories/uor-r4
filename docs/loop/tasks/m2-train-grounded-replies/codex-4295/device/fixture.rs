#![forbid(unsafe_code)]

use std::fs::{create_dir, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

const LEXICAL: usize = 96;
const TRANSPORT: usize = 32;
const WORDS: [[u32; 4]; 2] = [[0, 4, 0, 1], [5, 9, 1, 2]];
const SITES: [[u32; 4]; 7] = [
    [15, 24, 3, 4],
    [5, 9, 1, 2],
    [10, 14, 2, 3],
    [25, 28, 4, 5],
    [29, 33, 5, 6],
    [34, 38, 6, 7],
    [39, 44, 7, 8],
];
const TRIPLES: [[u32; 3]; 7] = [
    [0, 3, 6],
    [1, 3, 6],
    [2, 3, 6],
    [0, 4, 6],
    [1, 4, 6],
    [2, 4, 6],
    [0, 2, 6],
];
const RECOVERY: [[i16; 7]; 2] = [[-4, -1, -9, 0, 0, 0, 0], [0, 0, -11, -2, -8, 0, 0]];
const SECOND_RIVAL: [[i16; 7]; 2] = [[-4, -10, -9, 0, 0, 0, 0], [0, 0, -4, -8, -1, 0, 0]];
const GOLD_TIE: [[i16; 7]; 2] = [[4, -1, 4, 0, 0, 0, 0], RECOVERY[1]];
const RIVAL_TIE: [[i16; 7]; 2] = [[-4, -1, -1, 0, 0, 0, 0], RECOVERY[1]];
const CASES: [([[i16; 7]; 2], [i16; 3]); 6] = [
    (RECOVERY, [0, 1, 2]),
    (SECOND_RIVAL, [1, 4, 2]),
    (GOLD_TIE, [0, 1, 2]),
    (RECOVERY, [-1, -1, -1]),
    (RECOVERY, [0, 6, -1]),
    (RIVAL_TIE, [0, 1, -1]),
];

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

fn point(out: &mut Output, axis: i16, dimensions: usize) -> io::Result<()> {
    out.write_all(&axis.to_le_bytes())?;
    for _ in 1..dimensions {
        out.write_all(&0i16.to_le_bytes())?;
    }
    Ok(())
}

fn packet(root: &Path) -> io::Result<()> {
    let mut input = create(root, "text.bin")?;
    input.write_all(b"UORTXT01")?;
    words(&mut input, &[CASES.len() as u32])?;
    let mut labels = create(root, "labels.bin")?;
    labels.write_all(b"UORLNK01")?;
    words(&mut labels, &[CASES.len() as u32])?;
    let mut masks = create(root, "masks.tsv")?;
    for (transport, [sign, winner, control]) in CASES {
        words(
            &mut input,
            &[WORDS.len() as u32, SITES.len() as u32, TRIPLES.len() as u32],
        )?;
        for (address, axis) in WORDS.iter().zip([-16, 16]) {
            words(&mut input, address)?;
            point(&mut input, axis, LEXICAL)?;
        }
        for (address, axis) in SITES.iter().zip([0, 0, 0, 0, 0, 0, 10]) {
            words(&mut input, address)?;
            point(&mut input, axis, LEXICAL)?;
        }
        for axis in transport.into_iter().flatten() {
            point(&mut input, axis, TRANSPORT)?;
        }
        for triple in &TRIPLES {
            words(&mut input, triple)?;
        }
        words(&mut labels, &[0, 0, 1, 3])?;
        writeln!(masks, "{sign}\t{winner}\t{control}")?;
    }
    input.flush()?;
    labels.flush()?;
    masks.flush()
}

fn frozen(root: &Path) -> io::Result<()> {
    let mut fit = create(root, "fixture.words")?;
    words(&mut fit, &[1])?;
    for _ in 0..9 {
        for axis in [0, 0, 10, 0, 0, 0, 0] {
            words(&mut fit, &[1, 0])?;
            point(&mut fit, axis, LEXICAL)?;
        }
    }
    fit.flush()?;

    let mut state = create(root, "parent.state")?;
    state.write_all(b"UORSTT01")?;
    for axis in [-16, 16] {
        words(&mut state, &[1])?;
        point(&mut state, axis, LEXICAL)?;
        for _ in 1..32 {
            point(&mut state, 0, LEXICAL)?;
        }
        words(&mut state, &[2])?;
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
    prompt.write_all(b"keep drop drop medallion nut gear seal topic\n")?;
    prompt.flush()
}
