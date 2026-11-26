#![forbid(unsafe_code)]

use std::cmp::Reverse;
use std::fs::{create_dir, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

const WORDS: usize = 4;
const SITES: usize = 5;
const LEXICAL: usize = 96;
const EDGE: usize = 32;
const OBSERVATIONS: [i16; 14] = [0, 100, 200, 300, 400, 500, 600, 700, 2, 2, 2, 98, 98, 98];
const WORD_ADDRESS: [[u32; 4]; WORDS] = [
    [10, 11, 20, 21],
    [40, 41, 50, 51],
    [0, 1, 10, 11],
    [80, 81, 90, 91],
];
const SITE_ADDRESS: [[u32; 4]; SITES] = [
    [24, 26, 35, 36],
    [43, 45, 53, 54],
    [7, 9, 18, 19],
    [65, 67, 72, 73],
    [69, 71, 76, 77],
];
const SITE_POINT: [i16; SITES] = [0, 4, 0, 10, 10];
const BASE_TRIPLES: [[u32; 3]; 3] = [[0, 1, 3], [1, 0, 3], [2, 1, 3]];

#[derive(Clone)]
struct Example {
    lexical: [i16; WORDS],
    transport: [[i16; SITES]; WORDS],
    heading_tie: bool,
}

impl Example {
    fn new(delta: i16) -> Self {
        let mut transport = [[3000; SITES]; WORDS];
        transport[0] = [20 + delta, 1200 + delta, -22, 3000, 3000];
        transport[1] = [-1200 - delta, -20 - delta, 22, 3000, 3000];
        Self {
            lexical: [-30, 30, 0, 0],
            transport,
            heading_tie: false,
        }
    }
}

type Output = BufWriter<File>;

fn create(root: &Path, name: &str) -> io::Result<Output> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(name))
        .map(BufWriter::new)
}

fn words(out: &mut Output, words: &[u32]) -> io::Result<()> {
    for word in words {
        out.write_all(&word.to_le_bytes())?;
    }
    Ok(())
}

fn coordinates(out: &mut Output, first: i16, width: usize) -> io::Result<()> {
    out.write_all(&first.to_le_bytes())?;
    for _ in 1..width {
        out.write_all(&0i16.to_le_bytes())?;
    }
    Ok(())
}

fn geometry(root: &Path, name: &str, rows: &[Example]) -> io::Result<()> {
    let mut out = create(root, name)?;
    out.write_all(b"UORTXT01")?;
    words(&mut out, &[rows.len() as u32])?;
    for row in rows {
        words(
            &mut out,
            &[WORDS as u32, SITES as u32, 3 + u32::from(row.heading_tie)],
        )?;
        for (address, lexical) in WORD_ADDRESS.iter().zip(row.lexical) {
            words(&mut out, address)?;
            coordinates(&mut out, lexical, LEXICAL)?;
        }
        for (address, point) in SITE_ADDRESS.iter().zip(SITE_POINT) {
            words(&mut out, address)?;
            coordinates(&mut out, point, LEXICAL)?;
        }
        for word in row.transport {
            for edge in word {
                coordinates(&mut out, edge, EDGE)?;
            }
        }
        for tuple in BASE_TRIPLES {
            words(&mut out, &tuple)?;
        }
        if row.heading_tie {
            words(&mut out, &[0, 1, 4])?;
        }
    }
    out.flush()
}

fn annotations(
    root: &Path,
    name: &str,
    magic: &[u8; 8],
    values: &[u32],
    count: usize,
) -> io::Result<()> {
    let mut out = create(root, name)?;
    out.write_all(magic)?;
    words(&mut out, &[count as u32])?;
    for _ in 0..count {
        words(&mut out, values)?;
    }
    out.flush()
}

fn frozen(root: &Path) -> io::Result<()> {
    let mut out = create(root, "fixture.words")?;
    words(&mut out, &[1])?;
    for _ in 0..9 {
        for first in [4, 0, 10, 0, 0, 0, 0] {
            words(&mut out, &[1, 0])?;
            coordinates(&mut out, first, LEXICAL)?;
        }
    }
    out.flush()
}

// The finite data have one nonzero axis, so the independent reference needs
// only scalar absolute distances, partitions, and integer up/down counts.
fn fitted(values: &[i16], capacity: usize) -> io::Result<Vec<i16>> {
    let first = values
        .iter()
        .copied()
        .min()
        .ok_or_else(|| io::Error::other("empty fixture"))?;
    let mut centers = vec![first];
    while centers.len() < capacity {
        let choice = values
            .iter()
            .copied()
            .map(|candidate| {
                let coverage_gap = centers
                    .iter()
                    .map(|center| center.abs_diff(candidate))
                    .min();
                (coverage_gap.unwrap_or(0), Reverse(candidate))
            })
            .max();
        match choice {
            Some((distance, Reverse(value))) if distance > 0 => centers.push(value),
            _ => break,
        }
    }
    let initial = centers.clone();
    for _ in 0..2 {
        let mut partitions = vec![Vec::new(); centers.len()];
        for &value in values {
            let index = (0..centers.len())
                .min_by_key(|&i| (value.abs_diff(centers[i]), i))
                .ok_or_else(|| io::Error::other("no prototype in finite reference"))?;
            partitions[index].push(value);
        }
        for (sample, center) in partitions.iter().zip(centers.iter_mut()) {
            let up = sample.iter().filter(|&&v| v > *center).count();
            let down = sample.iter().filter(|&&v| v < *center).count();
            *center += match up.cmp(&down) {
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => 1,
            };
        }
    }
    if capacity == 8 && initial == centers {
        return Err(io::Error::other(
            "finite transport did not require a coordinate update",
        ));
    }
    Ok(centers)
}

fn expected_state(
    root: &Path,
    name: &str,
    train: &[Example],
    signs: [usize; 2],
    endpoints: [usize; 2],
) -> io::Result<()> {
    let mut out = create(root, name)?;
    out.write_all(b"UORSTT01")?;
    for sign in signs {
        let lexical: Vec<_> = train.iter().map(|r| r.lexical[sign]).collect();
        let edges: Vec<_> = train
            .iter()
            .map(|r| r.transport[sign][endpoints[sign]])
            .collect();
        for (input, capacity, width) in [(&lexical, 32, LEXICAL), (&edges, 8, EDGE)] {
            let centers = fitted(input, capacity)?;
            words(&mut out, &[centers.len() as u32])?;
            for i in 0..capacity {
                coordinates(&mut out, centers.get(i).copied().unwrap_or(0), width)?;
            }
        }
    }
    out.flush()
}

fn fixture(root: &Path) -> io::Result<()> {
    create_dir(root)?;
    frozen(root)?;
    let train: Vec<_> = OBSERVATIONS.iter().copied().map(Example::new).collect();
    geometry(root, "train.bin", &train)?;
    annotations(
        root,
        "train-labels.bin",
        b"UORLNK01",
        &[0, 0, 1, 1],
        train.len(),
    )?;
    expected_state(root, "expect-trained", &train, [0, 1], [0, 1])?;
    expected_state(root, "expect-swapped", &train, [1, 0], [0, 1])?;
    expected_state(root, "expect-endpoints", &train, [0, 1], [1, 0])?;

    let ordinary = Example::new(2);
    let mut noun_tie = ordinary.clone();
    noun_tie.transport[0][2] = noun_tie.transport[0][0];
    let mut governor_tie = ordinary.clone();
    governor_tie.lexical[3] = governor_tie.lexical[0];
    governor_tie.transport[3][0] = governor_tie.transport[0][0];
    let mut legal_joint = ordinary.clone();
    legal_joint.transport[0][0] = 23;
    legal_joint.transport[0][2] = 22;
    legal_joint.transport[1][0] = -22;
    legal_joint.transport[1][1] = -25;
    let mut heading_tie = ordinary.clone();
    heading_tie.heading_tie = true;
    let score = [ordinary, noun_tie, governor_tie, legal_joint, heading_tie];
    geometry(root, "predict.bin", &score)?;
    for (name, values) in [
        ("aid.bin", [0, 1, 0, 0, 1, 1]),
        ("aid-permuted.bin", [0, 1, 1, 1, 0, 0]),
        ("aid-wrong-edge.bin", [0, 1, 3, 0, 1, 1]),
        ("aid-wrong-key.bin", [3, 1, 0, 0, 1, 1]),
        ("aid-dangling.bin", [0, 1, WORDS as u32, 0, 1, 1]),
        ("aid-collision.bin", [0, 1, 0, 0, 1, 0]),
    ] {
        annotations(root, name, b"UORAID01", &values, score.len())?;
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let root = std::env::args_os()
        .nth(1)
        .ok_or_else(|| io::Error::other("missing fixture output directory"))?;
    fixture(Path::new(&root))
}
