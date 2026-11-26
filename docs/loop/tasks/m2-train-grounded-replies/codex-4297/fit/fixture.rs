#![forbid(unsafe_code)]

use std::fs::{create_dir, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

const LEXICAL: usize = 96;
const TRANSPORT: usize = 32;
const LEXICAL_SLOTS: usize = 32;
const TRANSPORT_SLOTS: usize = 8;
const ROWS: u32 = 6;
const WORDS: [[u32; 4]; 2] = [[0, 4, 0, 1], [32, 36, 8, 9]];
const SITES: [[u32; 4]; 7] = [
    [4, 8, 1, 2],
    [8, 12, 2, 3],
    [12, 16, 3, 4],
    [16, 20, 4, 5],
    [20, 24, 5, 6],
    [24, 28, 6, 7],
    [28, 32, 7, 8],
];
const FAR: [[i16; 3]; 4] = [[3, 3, 0], [1, 0, 0], [0, 1, 0], [6, 0, 0]];
const NEAR_LOST: [[i16; 3]; 4] = [[0, 0, 0], [3, 2, 0], [8, 0, 8], [0, 8, 8]];
const NEAR_GAINED: [[i16; 3]; 4] = [[3, 2, 0], [0, 0, 0], [8, 0, 8], [0, 8, 8]];

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

fn packet(root: &Path, boundary: bool) -> io::Result<()> {
    let text_name = if boundary {
        "boundary-text.bin"
    } else {
        "text.bin"
    };
    let mut input = create(root, text_name)?;
    input.write_all(b"UORTXT01")?;
    words(&mut input, &[ROWS])?;

    for affirmed in [FAR, NEAR_LOST, NEAR_GAINED] {
        for side in 0..2 {
            words(&mut input, &[2, 7, 8])?;
            for (address, coordinate) in WORDS.iter().zip([0, 10]) {
                words(&mut input, address)?;
                point(&mut input, &[coordinate], LEXICAL)?;
            }
            for address in &SITES {
                words(&mut input, address)?;
                point(&mut input, &[], LEXICAL)?;
            }
            // Shift the original site identities on the opposed row.
            // Keep the two strongest far rivals in site order on both rows.
            for edge in if side == 0 {
                affirmed
            } else {
                [affirmed[3], affirmed[1], affirmed[2], affirmed[0]]
            } {
                point(&mut input, &edge, TRANSPORT)?;
            }
            for _ in 4..SITES.len() {
                point(&mut input, &[], TRANSPORT)?;
            }
            for _ in 0..4 {
                point(&mut input, &[], TRANSPORT)?;
            }
            let (expected, other) = if boundary {
                ([i16::MAX, i16::MIN], [i16::MAX - 1, i16::MIN + 1])
            } else {
                ([0, 0], [7, 0])
            };
            for edge in if side == 0 {
                [expected, other]
            } else {
                [other, expected]
            } {
                point(&mut input, &edge, TRANSPORT)?;
            }
            point(&mut input, &[], TRANSPORT)?;
            for first in 0..4 {
                for second in 4..6 {
                    words(&mut input, &[first, second, 6])?;
                }
            }
        }
    }
    input.flush()
}

fn labels(root: &Path) -> io::Result<()> {
    let mut labels = create(root, "labels.bin")?;
    labels.write_all(b"UORLNK01")?;
    words(&mut labels, &[ROWS])?;
    let mut aid = create(root, "aid.bin")?;
    aid.write_all(b"UORAID01")?;
    words(&mut aid, &[ROWS])?;
    for row in 0..ROWS {
        let side = row % 2;
        let edges = [0, 3 * side, 1, 4 + side];
        words(&mut labels, &edges)?;
        words(&mut aid, &[0, 1])?;
        words(&mut aid, &edges)?;
    }
    labels.flush()?;
    aid.flush()
}

fn state(root: &Path, name: &str, coordinates: [[i16; 2]; 2]) -> io::Result<()> {
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
        // Nonzero inactive coordinates must remain byte-identical.
        for _ in 1..TRANSPORT_SLOTS {
            point(&mut out, &[101, -203], TRANSPORT)?;
        }
    }
    out.flush()
}

fn main() -> io::Result<()> {
    let directory = std::env::args_os()
        .nth(1)
        .ok_or_else(|| io::Error::other("usage: bridge-fixture NEW_DIRECTORY"))?;
    let root = Path::new(&directory);
    create_dir(root)?;
    packet(root, false)?;
    packet(root, true)?;
    labels(root)?;
    state(root, "parent.state", [[0, 0], [0, 0]])?;
    state(root, "expected-strongest-step2.state", [[0, 2], [0, 0]])?;
    state(root, "expected-all-step2.state", [[2, 2], [0, 0]])?;
    state(root, "expected-remine-step2.state", [[2, 4], [0, 0]])?;
    state(
        root,
        "boundary-parent.state",
        [[0, 0], [i16::MAX - 1, i16::MIN + 1]],
    )?;
    state(
        root,
        "expected-saturated.state",
        [[0, 0], [i16::MAX, i16::MIN]],
    )
}
