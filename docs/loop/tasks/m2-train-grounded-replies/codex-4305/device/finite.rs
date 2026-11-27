// Write small native-format input packets with hand-chosen integer distances.
use std::env;
use std::fs::{create_dir, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

const DIMENSIONS: usize = 96;
const TRANSPORT: usize = 32;
const SITES: u32 = 11;
const MASKS: &str =
    "0 +1 0 3 3 1 2 3 3 4 5 9\n1 +1 0 3 3 1 2 3 3 4 5 9\n2 -1 0 3 3 1 2 3 3 4 5 9\n";

fn word(out: &mut impl Write, value: u32) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())
}

fn index_list(out: &mut impl Write, indices: &[u32]) -> io::Result<()> {
    for &index in indices {
        word(out, index)?;
    }
    Ok(())
}

fn point(out: &mut impl Write, first: i16, length: usize) -> io::Result<()> {
    out.write_all(&first.to_le_bytes())?;
    for _ in 1..length {
        out.write_all(&0i16.to_le_bytes())?;
    }
    Ok(())
}

fn addressed_point(out: &mut impl Write, start: u32, first: i16) -> io::Result<()> {
    index_list(out, &[start * 3, start * 3 + 2, start, start + 1])?;
    point(out, first, DIMENSIONS)
}

fn bank(out: &mut impl Write) -> io::Result<()> {
    word(out, 1)?;
    for _ in 0..9 * 7 {
        index_list(out, &[1, 0])?;
        point(out, 0, DIMENSIONS)?;
    }
    Ok(())
}

fn state(out: &mut impl Write) -> io::Result<()> {
    out.write_all(b"UORSTT01")?;
    for (lexical, transport) in [(0, 100), (10, -100)] {
        word(out, 1)?;
        point(out, lexical, DIMENSIONS * 32)?;
        word(out, 1)?;
        point(out, transport, TRANSPORT * 8)?;
    }
    Ok(())
}

fn transport(row: u32, governor: u32, site: u32) -> i16 {
    let action_column = [0, 1, 2, 3, 4, 5, 9].iter().position(|&s| s == site);
    match (governor, action_column, site) {
        (0, Some(i), _) => 100 + [5, 1, 4, 3, 0, 6, 2][i],
        (1, Some(i), _) => 100 + [0, 4, 2, 5, 1, 6, 3][i],
        (3, Some(i), _) => 100 + [4, 5, 1, 3, 6, 0, 2][i],
        (4, Some(0), _) => 100,
        (2, _, 6) => -99,
        (2 | 5, _, 10) => -100,
        (0, _, 6) if row == 2 => -99,
        (0, _, 10) if row == 2 => -100,
        _ => 0,
    }
}

fn geometry(out: &mut impl Write) -> io::Result<()> {
    out.write_all(b"UORTXT01")?;
    word(out, 3)?;
    for row in 0..3 {
        let words = if row == 1 { 6 } else { 4 };
        let mirrored = row == 2;
        let action_sites = [0, 1, 2, 3, 4, 5, 9];
        let mut tuples: Vec<[u32; 3]> = action_sites.into_iter().map(|a| [a, 6, 7]).collect();
        tuples.push([0, 10, 8]);
        if row == 1 {
            tuples.push([0, 10, 7]);
        }
        index_list(out, &[words, SITES, tuples.len() as u32])?;
        for governor in 0..words {
            let lexical = [8, 0, if mirrored { 15 } else { 10 }, 3, 0, 10][governor as usize];
            let address = if mirrored && governor == 0 {
                13
            } else {
                governor + 1
            };
            addressed_point(out, address, if mirrored { 10 - lexical } else { lexical })?;
        }
        for site in 0..SITES {
            addressed_point(out, (site + 12) as u32, 0)?;
        }
        for governor in 0..words {
            for site in 0..SITES {
                let coordinate = transport(row, governor, site);
                point(
                    out,
                    if mirrored { -coordinate } else { coordinate },
                    TRANSPORT,
                )?;
            }
        }
        for [a, e, h] in tuples {
            index_list(out, &if mirrored { [e, a, h] } else { [a, e, h] })?;
        }
    }
    Ok(())
}

fn aid(out: &mut impl Write, swapped: bool) -> io::Result<()> {
    out.write_all(b"UORAID01")?;
    word(out, 3)?;
    for row in 0..3 {
        let a = if row == 2 { 1 } else { 0 };
        let b = if swapped { a + 2 } else { a + 1 };
        if row == 2 {
            index_list(out, &[2, 0, 2, 6, 0, b])?;
        } else {
            index_list(out, &[0, 2, 0, b, 2, 6])?;
        }
    }
    Ok(())
}

fn write(path: &Path, f: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>) -> io::Result<()> {
    let mut out = BufWriter::new(File::create_new(path)?);
    f(&mut out)?;
    out.flush()
}

fn main() -> io::Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [root] = args.as_slice() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: finite NEW_FIXTURE_DIRECTORY",
        ));
    };
    let root = PathBuf::from(root);
    create_dir(&root)?;
    write(&root.join("fixture.bank"), |out| bank(out))?;
    write(&root.join("fixture.state"), |out| state(out))?;
    write(&root.join("fixture.text"), |out| geometry(out))?;
    write(&root.join("masks.tsv"), |out| {
        out.write_all(MASKS.as_bytes())
    })?;
    write(&root.join("gold.aid"), |out| aid(out, false))?;
    write(&root.join("swapped.aid"), |out| aid(out, true))
}
