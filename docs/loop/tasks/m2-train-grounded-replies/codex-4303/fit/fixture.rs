#![forbid(unsafe_code)]

use std::fs::{create_dir, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

const LEXICAL: usize = 96;
const TRANSPORT: usize = 32;
const TAIL: usize = 20;
const LEXICAL_SLOTS: usize = 32;
const TRANSPORT_SLOTS: usize = 8;
const SIGN_BYTES: usize = 4 + 2 * LEXICAL_SLOTS * LEXICAL + 4 + 2 * TRANSPORT_SLOTS * TRANSPORT;

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

fn point(out: &mut Output, coordinates: &[i16]) -> io::Result<()> {
    for coordinate in coordinates {
        out.write_all(&coordinate.to_le_bytes())?;
    }
    Ok(())
}

fn physical_span(out: &mut Output, ordinal: u32) -> io::Result<()> {
    integers(out, &[ordinal * 5, ordinal * 5 + 4, ordinal, ordinal + 1])
}

const SITES: usize = 8;
const ROWS: u32 = 10;
const THIRDS: [[u32; 3]; 6] = [
    [0, 4, 5],
    [1, 4, 5],
    [2, 4, 5],
    [3, 4, 5],
    [3, 4, 6],
    [7, 6, 5],
];

fn packets(root: &Path) -> io::Result<()> {
    let mut packet = create(root, "text.bin")?;
    packet.write_all(b"UORTXT01")?;
    integers(&mut packet, &[ROWS])?;
    let mut requests = create(root, "requests.tsv")?;
    let mut repeated = create(root, "repeated.tsv")?;
    for row in 0..ROWS {
        let pair = row / 2;
        let side = pair % 2;
        let sign = if side == 0 { 1_i16 } else { -1 };
        let requested = row % 2;
        let rival = 1 - requested;
        let word = if pair == 4 { 1 - side } else { side };
        let triples = match pair {
            0 | 1 => &THIRDS[..],
            3 => &THIRDS[..2],
            _ => &THIRDS[..5],
        };
        writeln!(requests, "{row}\t{sign}\t{word}\t{requested}\t{rival}")?;
        let repeat_row = if row == 1 { 0 } else { row };
        writeln!(
            repeated,
            "{repeat_row}\t{sign}\t{word}\t{requested}\t{rival}"
        )?;
        integers(&mut packet, &[2, SITES as u32, triples.len() as u32])?;
        for (ordinal, lexical) in [-16_i16, 16].into_iter().enumerate() {
            physical_span(&mut packet, ordinal as u32)?;
            let mut coordinates = [0; LEXICAL];
            coordinates[0] = lexical;
            point(&mut packet, &coordinates)?;
        }
        for site in 0..SITES {
            physical_span(&mut packet, 2 + site as u32)?;
            point(&mut packet, &[0; LEXICAL])?;
        }
        for ordinal in 0..2 {
            for site in 0..SITES as u32 {
                let mut coordinates = [0; TRANSPORT];
                coordinates[0] = 7;
                if ordinal == side && pair < 2 {
                    match site {
                        x if x == requested => coordinates[TAIL] = 3 * sign,
                        2 => coordinates[1] = 1,
                        3 => {
                            coordinates[1] = 1;
                            coordinates[TAIL] = 2 * sign;
                        }
                        _ => {}
                    }
                }
                if ordinal != side && site == 6 && pair < 2 {
                    coordinates[1] = 10;
                }
                point(&mut packet, &coordinates)?;
            }
        }
        for &[focus, opposite, heading] in triples {
            let triple = if side == 0 {
                [focus, opposite, heading]
            } else {
                [opposite, focus, heading]
            };
            integers(&mut packet, &triple)?;
        }
    }
    packet.flush()?;
    requests.flush()?;
    repeated.flush()
}

fn state(root: &Path, name: &str, moved: bool) -> io::Result<()> {
    let mut state = create(root, name)?;
    state.write_all(b"UORSTT01")?;
    for (side, lexical) in [-16_i16, 16].into_iter().enumerate() {
        integers(&mut state, &[1])?;
        for slot in 0..LEXICAL_SLOTS {
            let mut coordinates = [0; LEXICAL];
            if slot == 0 {
                coordinates[0] = lexical;
            }
            point(&mut state, &coordinates)?;
        }
        integers(&mut state, &[1])?;
        for slot in 0..TRANSPORT_SLOTS {
            let mut coordinates = [0; TRANSPORT];
            coordinates[0] = 7;
            if slot == 0 && moved {
                coordinates[TAIL] = if side == 0 { 4 } else { -4 };
            }
            if slot != 0 {
                coordinates[TAIL] = 123;
            }
            point(&mut state, &coordinates)?;
        }
    }
    state.flush()
}

fn generate(root: &Path) -> io::Result<()> {
    create_dir(root)?;
    packets(root)?;
    let mut fit = create(root, "fixture.words")?;
    integers(&mut fit, &[1])?;
    for _ in 0..9 {
        for _ in 0..7 {
            integers(&mut fit, &[1, 0])?;
            point(&mut fit, &[0; LEXICAL])?;
        }
    }
    fit.flush()?;
    state(root, "start.state", false)?;
    state(root, "known.state", true)
}
fn verify_invariant(original: &Path, candidate: &Path) -> io::Result<()> {
    let a = std::fs::read(original)?;
    let mut b = std::fs::read(candidate)?;
    if a.len() != 8 + 2 * SIGN_BYTES || a.len() != b.len() {
        return Err(io::Error::other("wrong unchanged native state size"));
    }
    for side in 0..2 {
        let begin = 8 + side * SIGN_BYTES + 4 + 2 * LEXICAL_SLOTS * LEXICAL + 4;
        let range = begin + 2 * TAIL..begin + 2 * TRANSPORT;
        b[range.clone()].copy_from_slice(&a[range]);
    }
    if a != b {
        return Err(io::Error::other(
            "bytes outside active tail coordinates changed",
        ));
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [root] => generate(Path::new(root)),
        [original, candidate] => verify_invariant(Path::new(original), Path::new(candidate)),
        _ => Err(io::Error::other(
            "expected output directory or original and candidate paths",
        )),
    }
}
