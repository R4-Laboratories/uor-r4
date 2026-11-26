#![forbid(unsafe_code)]

use std::{
    env, fs,
    io::{self, Write},
    path::Path,
};

const SIGN_SIZE: usize = 4 + 32 * 96 * 2 + 4 + 8 * 32 * 2;

fn ensure(test: bool, label: &str) -> io::Result<()> {
    if test {
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, label))
    }
}

fn verify(original: &Path, candidate: &Path, parser_view: &Path) -> io::Result<()> {
    let original = fs::read(original)?;
    let mut view = fs::read(candidate)?;
    ensure(
        original.len() == 8 + 2 * SIGN_SIZE
            && view.len() == original.len()
            && original.starts_with(b"UORSTT01")
            && view.starts_with(b"UORSTT01"),
        "invalid synthetic native envelope",
    )?;
    let mut expected = original.clone();
    for side in 0..2 {
        let lexical = 8 + side * SIGN_SIZE;
        let transport = lexical + 4 + 32 * 96 * 2;
        ensure(
            original[lexical..lexical + 4] == [1, 0, 0, 0]
                && original[transport..transport + 4] == [1, 0, 0, 0],
            "fixture active slots changed",
        )?;
        let tail = transport + 4 + 20 * 2..transport + 4 + 32 * 2;
        expected[tail.clone()].copy_from_slice(&view[tail]);
    }
    ensure(
        expected == view,
        "fixture changed bytes outside the two native active tails",
    )?;
    for side in 0..2 {
        let transport = 8 + side * SIGN_SIZE + 4 + 32 * 96 * 2;
        for slot in 1..8 {
            let start = transport + 4 + slot * 32 * 2;
            let mut sentinel = [0; 64];
            sentinel[..2].copy_from_slice(&7_i16.to_le_bytes());
            sentinel[40..42].copy_from_slice(&123_i16.to_le_bytes());
            ensure(
                view[start..start + 64] == sentinel,
                "unexpected inactive synthetic sentinel",
            )?;
            view[start..start + 64].fill(0);
        }
    }
    fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(parser_view)?
        .write_all(&view)
}

fn main() -> io::Result<()> {
    let paths: Vec<_> = env::args_os().skip(1).collect();
    match paths.as_slice() {
        [original, candidate, view] => {
            verify(Path::new(original), Path::new(candidate), Path::new(view))
        }
        _ => Err(io::Error::other(
            "expected original state, captured candidate and new finite parser view",
        )),
    }
}
