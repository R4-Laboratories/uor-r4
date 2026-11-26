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
const SITES: usize = 5;
const TRIPLES: [[u32; 3]; 3] = [[0, 2, 4], [1, 2, 4], [0, 3, 4]];

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

fn packets(root: &Path) -> io::Result<()> {
    let mut packet = create(root, "text.bin")?;
    packet.write_all(b"UORTXT01")?;
    integers(&mut packet, &[4])?;
    let mut good = create(root, "requests.tsv")?;
    let mut exchanged = create(root, "exchanged.tsv")?;
    for row in 0..4 {
        let sign = row / 2;
        let selector = row % 2;
        let requested = 2 * sign + selector;
        let rival = 2 * sign + 1 - selector;
        let signed = if sign == 0 { 1 } else { -1 };
        writeln!(good, "{row}\t{signed}\t{sign}\t{requested}\t{rival}")?;
        writeln!(exchanged, "{row}\t{signed}\t{sign}\t{rival}\t{requested}")?;
        integers(&mut packet, &[2, SITES as u32, TRIPLES.len() as u32])?;
        for (word, lexical) in [-16_i16, 16].into_iter().enumerate() {
            physical_span(&mut packet, word as u32)?;
            let mut coordinates = [0; LEXICAL];
            coordinates[0] = lexical;
            point(&mut packet, &coordinates)?;
        }
        for site in 0..SITES {
            physical_span(&mut packet, 2 + site as u32)?;
            point(&mut packet, &[0; LEXICAL])?;
        }
        for word in 0..2 {
            for site in 0..SITES {
                let mut coordinates = [0; TRANSPORT];
                coordinates[0] = 7;
                coordinates[TAIL] = if word == sign {
                    match site {
                        x if x == requested => -2 * signed,
                        x if x == rival => signed,
                        _ => 25,
                    }
                } else if site == 2 * word {
                    0
                } else {
                    40
                };
                point(&mut packet, &coordinates)?;
            }
        }
        for triple in TRIPLES {
            integers(&mut packet, &triple)?;
        }
    }
    packet.flush()?;
    good.flush()?;
    exchanged.flush()?;

    let mut broken = create(root, "broken.tsv")?;
    broken.write_all(b"0\t1\t0\t0\t1\n0\t1\t0\t1\t0\n2\t-1\t1\t2\t3\n3\t-1\t1\t3\t2\n")?;
    broken.flush()?;
    let mut wrong = create(root, "wrong-sign.tsv")?;
    wrong.write_all(b"0\t-1\t0\t0\t1\n1\t-1\t0\t1\t0\n2\t1\t1\t2\t3\n3\t1\t1\t3\t2\n")?;
    wrong.flush()
}

fn state(root: &Path, name: &str, median: bool, damage: bool) -> io::Result<()> {
    let mut state = create(root, name)?;
    state.write_all(b"UORSTT01")?;
    for (side, lexical) in [-16_i16, 16].into_iter().enumerate() {
        integers(&mut state, &[1])?;
        for slot in 0..LEXICAL_SLOTS {
            let mut coordinates = [0; LEXICAL];
            if slot == 0 {
                coordinates[0] = lexical + i16::from(damage);
            }
            point(&mut state, &coordinates)?;
        }
        integers(&mut state, &[1])?;
        for slot in 0..TRANSPORT_SLOTS {
            let mut coordinates = [0; TRANSPORT];
            coordinates[0] = 7;
            if slot == 0 && median {
                coordinates[31] = if side == 0 { 1 } else { -1 };
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
    state(root, "original.state", false, false)?;
    state(root, "median.state", true, false)?;
    state(root, "damaged.state", false, true)
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
