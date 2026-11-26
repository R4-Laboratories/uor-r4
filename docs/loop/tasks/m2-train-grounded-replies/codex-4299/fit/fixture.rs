#![forbid(unsafe_code)]

use std::{
    fs::{create_dir, File, OpenOptions},
    io::{self, BufWriter, Write},
    path::Path,
};

const LEXICAL: usize = 96;
const TRANSPORT: usize = 32;
const START_AXIS: usize = 20;
const LEXICAL_SLOTS: usize = 32;
const TRANSPORT_SLOTS: usize = 8;
const ROWS: usize = 4;
const WORDS: [[u32; 4]; 2] = [[0, 4, 0, 1], [84, 88, 21, 22]];
const SITES: [[u32; 4]; 3] = [[36, 40, 9, 10], [44, 48, 11, 12], [76, 80, 19, 20]];

// Each point supplies axes 20, 21 and 31. The signed minimum and maximum values
// are far apart without requiring any widening of the stored state.
const ENDPOINT: [[[i16; 3]; 2]; ROWS] = [
    [[-3, -7, 0], [3, 0, i16::MIN]],
    [[5, -5, 0], [9, 0, -1]],
    [[10, 1, 0], [12, 0, 5]],
    [[14, 3, 0], [15, 0, i16::MAX]],
];
const ORDERED: [[[i16; 3]; 2]; ROWS] = [
    [[24, 7, 0], [-22, 0, i16::MAX]],
    [[26, -5, 0], [-20, 0, -4]],
    [[-20, 1, 0], [-2, 0, 2]],
    [[-22, -3, 0], [6, 0, i16::MIN]],
];
const BASE: [[i16; 3]; 3] = [[0, 0, 0], [10, 0, 0], [0, 0, 0]];
const ENDPOINT_MEDIANS: [[i16; 3]; 3] = [[-3, -7, 0], [10, 1, 0], [9, 0, -1]];
const ORDERED_MEDIANS: [[i16; 3]; 3] = [[24, -5, 0], [-22, -3, 0], [-20, 0, -4]];

type Output = BufWriter<File>;

fn create(root: &Path, name: &str) -> io::Result<Output> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(name))
        .map(BufWriter::new)
}

fn words(out: &mut Output, values: &[u32]) -> io::Result<()> {
    for &value in values {
        out.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn point(out: &mut Output, values: &[i16]) -> io::Result<()> {
    for &value in values {
        out.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn transport(prefix: i16, [x, y, z]: [i16; 3]) -> [i16; TRANSPORT] {
    let mut result = [0; TRANSPORT];
    result[0] = prefix;
    result[START_AXIS] = x;
    result[START_AXIS + 1] = y;
    result[TRANSPORT - 1] = z;
    result
}

fn labels(root: &Path) -> io::Result<()> {
    let mut labels = create(root, "labels.bin")?;
    labels.write_all(b"UORLNK01")?;
    words(&mut labels, &[ROWS as u32])?;
    for row in 0..ROWS {
        // Alternate both gold site indices while keeping the same signed geometry.
        let swap = (row % 2) as u32;
        words(&mut labels, &[0, swap, 1, 1 - swap])?;
    }
    labels.flush()
}

fn packet(
    root: &Path,
    name: &str,
    points: [[[i16; 3]; 2]; ROWS],
    alter_short_path: bool,
) -> io::Result<()> {
    let mut output = create(root, name)?;
    output.write_all(b"UORTXT01")?;
    words(&mut output, &[ROWS as u32])?;
    for (row, signed) in points.into_iter().enumerate() {
        words(&mut output, &[2, 3, 2])?;
        for (word, address) in WORDS.iter().enumerate() {
            words(&mut output, address)?;
            let mut lexical = [0; LEXICAL];
            lexical[0] = 16 * word as i16;
            point(&mut output, &lexical)?;
        }
        for address in &SITES {
            words(&mut output, address)?;
            point(&mut output, &[0; LEXICAL])?;
        }
        for (word, gold) in signed.into_iter().enumerate() {
            for site in 0..SITES.len() {
                let coordinates = if site == (row + word) % 2 {
                    transport(1, gold)
                } else if alter_short_path && word == 1 && site == 2 {
                    transport(0, [1, 0, 0])
                } else {
                    [0; TRANSPORT]
                };
                point(&mut output, &coordinates)?;
            }
        }
        words(&mut output, &[0, 1, 2, 1, 0, 2])?;
    }
    output.flush()
}

fn state(root: &Path, name: &str, trained: [[i16; 3]; 3]) -> io::Result<()> {
    let mut output = create(root, name)?;
    output.write_all(b"UORSTT01")?;
    for sign in 0..2 {
        words(&mut output, &[1])?;
        let mut lexical = [0; LEXICAL];
        lexical[0] = 16 * sign;
        point(&mut output, &lexical)?;
        for _ in 1..LEXICAL_SLOTS {
            point(&mut output, &[302; LEXICAL])?;
        }
        let active = 3 - sign as u32;
        words(&mut output, &[active])?;
        for slot in 0..TRANSPORT_SLOTS as u32 {
            let coordinates = match (sign, slot) {
                (0, 0) => trained[0],
                (0, 1) => trained[1],
                (0, 2) => [300, 0, 0],
                (1, 0) => trained[2],
                (1, 1) => [-50, 0, 0],
                _ => [711, -893, 151],
            };
            point(&mut output, &transport(4, coordinates))?;
        }
    }
    output.flush()
}

fn main() -> io::Result<()> {
    let directory = std::env::args_os()
        .nth(1)
        .ok_or_else(|| io::Error::other("usage: median-fixture NEW_DIRECTORY"))?;
    let root = Path::new(&directory);
    create_dir(root)?;
    labels(root)?;
    packet(root, "endpoint.bin", ENDPOINT, false)?;
    packet(root, "ordered.bin", ORDERED, false)?;
    packet(root, "changed-short.bin", ORDERED, true)?;
    state(root, "protected.state", BASE)?;
    for (arm, estimates) in [("endpoint", ENDPOINT_MEDIANS), ("ordered", ORDERED_MEDIANS)] {
        state(root, &format!("{arm}-full.state"), estimates)?;
        for (index, slot) in [0, 1, 8].into_iter().enumerate() {
            let mut single = BASE;
            single[index] = estimates[index];
            state(root, &format!("{arm}-slot-{slot:02}.state"), single)?;
        }
    }
    Ok(())
}
