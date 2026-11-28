use std::{
    env, fs,
    io::{self, Write},
    path::Path,
};

const BANKS: usize = 16;
const SLOTS: usize = 256;
const WORDS: usize = BANKS * SLOTS;
const EPOCHS: usize = 32;
const WIDTH: usize = 36;
const TRAIN: &[u8; 8] = b"URELPAI1";
const REPLAY: &[u8; 8] = b"URELEVT1";

#[derive(Clone, Copy)]
struct Row {
    keys: [u16; BANKS],
    ordinary: u16,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn publish(path: &Path, data: &[u8]) -> io::Result<()> {
    let mut output = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    output.write_all(data)?;
    output.sync_all()
}

fn fixture_rows() -> [Row; 10] {
    fn row(slot: u16, ordinary: u16) -> Row {
        Row {
            keys: std::array::from_fn(|bank| (bank * SLOTS) as u16 + slot),
            ordinary,
        }
    }

    let first = row(1, 31);
    let mut rival = row(1, 701);
    rival.keys[0] = 2;
    let mut order_rival = row(80, 103);
    order_rival.keys[2] += 1;
    [
        first,
        rival,
        row(22, 222),
        row(22, 223),
        row(80, 100),
        row(80, 101),
        row(80, 102),
        order_rival,
        row(251, 4095),
        row(252, 1),
    ]
}

fn encode_rows(rows: &[Row], train: bool) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(12 + rows.len() * WIDTH);
    bytes.extend_from_slice(if train { TRAIN } else { REPLAY });
    bytes.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    for (index, row) in rows.iter().enumerate() {
        for key in row.keys {
            bytes.extend_from_slice(&key.to_le_bytes());
        }
        bytes.extend_from_slice(&row.ordinary.to_le_bytes());
        bytes.extend_from_slice(&[u8::from(train && index % 2 == 0), 0]);
    }
    bytes
}

fn decode_replay(path: &Path) -> io::Result<Vec<Row>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 12 || bytes[..8] != *REPLAY {
        return Err(invalid("expected target-blind URELEVT1 tape"));
    }
    let count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if !(1..=16_384).contains(&count) || bytes.len() != 12 + count * WIDTH {
        return Err(invalid("invalid tape size"));
    }
    bytes[12..]
        .chunks_exact(WIDTH)
        .map(|record| {
            let keys = std::array::from_fn(|bank| {
                u16::from_le_bytes([record[2 * bank], record[2 * bank + 1]])
            });
            let ordinary = u16::from_le_bytes([record[32], record[33]]);
            if ordinary >= 4096
                || record[34..] != [0, 0]
                || keys
                    .iter()
                    .enumerate()
                    .any(|(bank, &key)| usize::from(key) / SLOTS != bank)
            {
                return Err(invalid("invalid ordinary ID, reserved field, or bank key"));
            }
            Ok(Row { keys, ordinary })
        })
        .collect()
}

fn read_words(path: &Path) -> io::Result<[i16; WORDS]> {
    let bytes = fs::read(path)?;
    if bytes.len() != 2 * WORDS {
        return Err(invalid("expected exactly 8,192 word bytes"));
    }
    Ok(std::array::from_fn(|i| {
        i16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])
    }))
}

fn encode_words(values: &[i16; WORDS]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn score(row: &Row, words: &[i16; WORDS]) -> i32 {
    row.keys
        .iter()
        .map(|&key| i32::from(words[usize::from(key)]))
        .sum()
}

fn prediction_bytes(rows: &[Row], words: &[i16; WORDS]) -> Vec<u8> {
    let mut result = Vec::with_capacity(rows.len() * 8);
    for row in rows {
        let value = score(row, words);
        let emitted = if value > 0 { 1 } else { row.ordinary };
        result.extend_from_slice(&value.to_le_bytes());
        result.extend_from_slice(&emitted.to_le_bytes());
        result.extend_from_slice(&[0, 0]);
    }
    result
}

fn make_fixture(dir: &Path) -> io::Result<()> {
    let rows = fixture_rows();
    publish(&dir.join("finite.train"), &encode_rows(&rows[..8], true))?;
    let tape = encode_rows(&rows, false);
    publish(&dir.join("finite.replay"), &tape)?;

    let mut wrong = tape.clone();
    wrong[12 + 34] = 1;
    publish(&dir.join("label.replay"), &wrong)?;
    wrong = tape.clone();
    wrong[13] = 1;
    publish(&dir.join("bank.replay"), &wrong)?;
    publish(&dir.join("truncated.replay"), &tape[..tape.len() - 1])?;

    publish(&dir.join("zero.words"), &encode_words(&[0; WORDS]))?;
    let mut extremes = [0; WORDS];
    for bank in 0..BANKS {
        extremes[bank * SLOTS + 251] = i16::MAX;
        extremes[bank * SLOTS + 252] = i16::MIN;
    }
    publish(&dir.join("extreme.words"), &encode_words(&extremes))
}

fn next_random(counter: &mut u64) -> u64 {
    *counter = counter.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mixed = (*counter ^ (*counter >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

// Independent finite oracle. This always obtains examples from the literals
// above. There is no CPU fitting entry point for external tapes.
fn training_oracle(seed: u64, flipped: bool) -> ([i16; WORDS], [usize; 3]) {
    let rows = fixture_rows();
    let mut counter = seed;
    let mut totals = [0; 3];
    let mut words = [0_i16; WORDS];
    for _ in 0..EPOCHS {
        let mut order = [0, 1, 2, 3];
        for top in (1..order.len()).rev() {
            let chosen = (next_random(&mut counter) % (top + 1) as u64) as usize;
            order.swap(top, chosen);
        }
        for pair in order {
            let stop = &rows[2 * pair + usize::from(flipped)];
            let cont = &rows[2 * pair + usize::from(!flipped)];
            let violations = [score(stop, &words) <= 0, score(cont, &words) >= 0];
            let mut delta = [0_i16; WORDS];
            for (side, row) in [stop, cont].into_iter().enumerate() {
                if violations[side] {
                    totals[side] += 1;
                    for key in row.keys {
                        delta[usize::from(key)] += if side == 0 { 1 } else { -1 };
                    }
                }
            }
            for (parameter, change) in words.iter_mut().zip(delta) {
                let next = parameter.saturating_add(change);
                totals[2] += usize::from(next != *parameter);
                *parameter = next;
            }
        }
    }
    (words, totals)
}

fn verify(tape: &Path, words: &Path, result: &Path) -> io::Result<()> {
    let rows = decode_replay(tape)?;
    let expected = prediction_bytes(&rows, &read_words(words)?);
    let actual = fs::read(result)?;
    if expected != actual {
        let offset = expected.iter().zip(&actual).position(|(a, b)| a != b);
        return Err(invalid(&format!(
            "raw predictions diverged at byte {offset:?}, expected {} bytes, observed {}",
            expected.len(),
            actual.len()
        )));
    }
    println!(
        "verified {} original rows and {} exact result bytes",
        rows.len(),
        expected.len()
    );
    Ok(())
}

fn verify_fixture(dir: &Path) -> io::Result<()> {
    let rows = fixture_rows();
    let tape = dir.join("finite.replay");
    let mut results = Vec::new();
    for seed in [1, 2] {
        for (control, flipped) in [("actual", false), ("flipped", true)] {
            let stem = format!("{seed}.{control}");
            let words_path = dir.join(format!("{stem}.words"));
            let observed = read_words(&words_path)?;
            let (expected, [stop, cont, changed]) = training_oracle(seed, flipped);
            if observed != expected
                || score(&rows[0], &observed).signum() != if flipped { -1 } else { 1 }
                || score(&rows[1], &observed) != -score(&rows[0], &observed)
                || score(&rows[2], &observed) != 0
                || score(&rows[3], &observed) != 0
            {
                return Err(invalid(
                    "fitted table failed oriented and collision controls",
                ));
            }
            let receipt = format!(
                "{{\"seed\":{seed},\"epochs\":32,\"pairs\":4,\"flipped\":{flipped},\"opportunities\":128,\"stop_violations\":{stop},\"continue_violations\":{cont},\"changed_words\":{changed}}}\n"
            );
            if fs::read(dir.join(format!("{stem}.json")))? != receipt.as_bytes() {
                return Err(invalid("device update counters diverged"));
            }
            verify(&tape, &words_path, &dir.join(format!("{stem}.scores")))?;
            results.push(observed);
        }
    }
    let extreme = read_words(&dir.join("extreme.words"))?;
    if results[0] == results[2]
        || score(&rows[8], &extreme) != 524_272
        || score(&rows[9], &extreme) != -524_288
    {
        return Err(invalid("seed order control or signed i32 bounds failed"));
    }
    for fixture in ["zero", "extreme"] {
        verify(
            &tape,
            &dir.join(format!("{fixture}.words")),
            &dir.join(format!("{fixture}.scores")),
        )?;
    }
    let mut bounded_update = [0_i16; WORDS];
    for bank in 0..BANKS {
        bounded_update[bank * SLOTS + 249] = 1;
        bounded_update[bank * SLOTS + 250] = -1;
    }
    bounded_update[249] = i16::MAX;
    bounded_update[250] = i16::MIN;
    bounded_update[SLOTS + 249] = -32_766;
    bounded_update[SLOTS + 250] = 32_766;
    bounded_update[2 * SLOTS + 250] = 0;
    if read_words(&dir.join("saturation.words"))? != bounded_update
        || fs::read(dir.join("saturation.csv"))? != b"1,1,30\n"
    {
        return Err(invalid(
            "saturating update failed at one or both i16 bounds",
        ));
    }
    println!(
        "verified fitted word reopens, shuffled orders, opposite labels, ties, and both saturating bounds"
    );
    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    match args.as_slice() {
        [command, dir] if command == "fixture" => make_fixture(Path::new(dir)),
        [command, dir] if command == "verify-fixture" => verify_fixture(Path::new(dir)),
        [command, tape, words, result] if command == "verify" => {
            verify(Path::new(tape), Path::new(words), Path::new(result))
        }
        _ => Err(invalid(
            "usage: termination-oracle {fixture|verify-fixture} DIR or verify TAPE WORDS SCORES",
        )),
    }
}
