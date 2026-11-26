#![forbid(unsafe_code)]

use std::fs::{create_dir, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

const WORDS: [[u32; 4]; 2] = [[0, 4, 1, 2], [40, 44, 10, 11]];
const SITES: [[u32; 4]; 9] = [
    [6, 10, 2, 3],
    [12, 16, 4, 5],
    [18, 22, 6, 7],
    [24, 28, 7, 8],
    [30, 34, 8, 9],
    [50, 54, 12, 13],
    [2, 6, 1, 2],
    [55, 59, 13, 14],
    [60, 64, 1, 2],
];
const TRIPLES: [[u32; 3]; 8] = [
    [0, 3, 5],
    [0, 4, 5],
    [1, 3, 5],
    [1, 4, 5],
    [2, 3, 5],
    [6, 3, 5],
    [7, 3, 5],
    [8, 3, 5],
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

fn point(out: &mut Output, value: i16, width: usize) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())?;
    for _ in 1..width {
        out.write_all(&0i16.to_le_bytes())?;
    }
    Ok(())
}

struct Row<'a> {
    governor: i16,
    affirmed: [i16; 9],
    excluded: [i16; 9],
    tuples: &'a [[u32; 3]],
}

impl Row<'_> {
    fn write(&self, out: &mut Output) -> io::Result<()> {
        words(out, &[2, 9, self.tuples.len() as u32])?;
        for (address, lexical) in WORDS.iter().zip([self.governor, 10]) {
            words(out, address)?;
            point(out, lexical, 96)?;
        }
        // Sites 0 and 1 have identical coordinates and distinct occurrences.
        for (address, value) in SITES.iter().zip([0, 0, 2, 3, 4, 5, 6, 7, 8]) {
            words(out, address)?;
            point(out, value, 96)?;
        }
        for transport in [self.affirmed, self.excluded] {
            for coordinate in transport {
                point(out, coordinate, 32)?;
            }
        }
        for tuple in self.tuples {
            words(out, tuple)?;
        }
        Ok(())
    }
}

fn packet(root: &Path, name: &str, rows: &[Row<'_>]) -> io::Result<()> {
    let mut input = create(root, &format!("{name}.bin"))?;
    input.write_all(b"UORTXT01")?;
    words(&mut input, &[rows.len() as u32])?;
    let mut labels = create(root, &format!("{name}-labels.bin"))?;
    labels.write_all(b"UORLNK01")?;
    words(&mut labels, &[rows.len() as u32])?;
    for row in rows {
        row.write(&mut input)?;
        words(&mut labels, &[0, 0, 1, 3])?;
    }
    input.flush()?;
    labels.flush()
}

fn state(root: &Path, name: &str, affirmed: [i16; 2]) -> io::Result<()> {
    let mut out = create(root, name)?;
    out.write_all(b"UORSTT01")?;
    for (governor, transport) in [(0, affirmed.as_slice()), (10, [0].as_slice())] {
        words(&mut out, &[1])?;
        point(&mut out, governor, 96)?;
        for _ in 1..32 {
            point(&mut out, 0, 96)?;
        }
        words(&mut out, &[transport.len() as u32])?;
        for slot in 0..8 {
            point(&mut out, transport.get(slot).copied().unwrap_or(0), 32)?;
        }
    }
    out.flush()
}

fn frozen(root: &Path) -> io::Result<()> {
    let mut out = create(root, "fixture.words")?;
    words(&mut out, &[1])?;
    for _ in 0..9 {
        for first in [0, 3, 5, 0, 0, 0, 0] {
            words(&mut out, &[1, 0])?;
            point(&mut out, first, 96)?;
        }
    }
    out.flush()
}

fn generate(root: &Path) -> io::Result<()> {
    create_dir(root)?;
    let mut original = [b' '; 64];
    for address in &SITES[..2] {
        original[address[0] as usize..address[1] as usize].copy_from_slice(b"bell");
    }
    let mut source = create(root, "repeated-site-source.txt")?;
    source.write_all(&original)?;
    source.flush()?;
    let excluded_win = [0, 0, 0, 0, 5, 0, 0, 0, 0];
    let outranked = [4, 19, 2, 0, 0, 0, 0, 19, 7];
    let win = [10, -20, -20, 0, 0, 0, 0, -20, -20];
    packet(
        root,
        "fixture",
        &[
            Row {
                governor: 0,
                affirmed: outranked,
                excluded: excluded_win,
                tuples: &TRIPLES,
            },
            Row {
                governor: 0,
                affirmed: win,
                excluded: [0; 9],
                tuples: &TRIPLES,
            },
            Row {
                governor: 5,
                affirmed: outranked,
                excluded: excluded_win,
                tuples: &TRIPLES,
            },
            Row {
                governor: 0,
                affirmed: outranked,
                excluded: excluded_win,
                tuples: &TRIPLES[2..],
            },
            Row {
                governor: 0,
                affirmed: outranked,
                excluded: excluded_win,
                tuples: &TRIPLES[..1],
            },
        ],
    )?;
    packet(
        root,
        "tie-only",
        &[Row {
            governor: 0,
            affirmed: [0; 9],
            excluded: [0; 9],
            tuples: &TRIPLES,
        }],
    )?;
    let same_side_loss = [0, 0, 0, 4, 2, 0, 0, 0, 0];
    let right = [1, -1, 0, 0, 0, 0, 0, 0, 0];
    let left = [-1, 1, 0, 0, 0, 0, 0, 0, 0];
    let pocket = [right, right, left].map(|affirmed| Row {
        governor: 0,
        affirmed,
        excluded: same_side_loss,
        tuples: &TRIPLES[..3],
    });
    packet(root, "pocket", &pocket)?;
    state(root, "parent.state", [0, 20])?;
    state(root, "expect-contrastive.state", [1, 21])?;
    state(root, "expect-positive.state", [4, 20])?;
    state(root, "expect-pocket.state", [1, 20])?;
    frozen(root)?;
    Ok(())
}

fn main() -> io::Result<()> {
    let output = std::env::args_os()
        .nth(1)
        .ok_or_else(|| io::Error::other("usage: fixture NEW_DIRECTORY"))?;
    generate(Path::new(&output))
}
