use std::collections::BTreeSet;
use std::io::{self, Cursor, Read};

pub const POINT: usize = 96;
pub const EDGE: usize = 32;
const WORD_SLOTS: usize = 32;
const EDGE_SLOTS: usize = 8;

#[derive(Clone, Copy)]
pub struct Address {
    pub bytes: [u32; 2],
    pub tokens: [u32; 2],
}

impl Address {
    pub fn overlaps(self, other: Self) -> bool {
        self.bytes[0] < other.bytes[1] && other.bytes[0] < self.bytes[1]
    }

    pub fn gap(self, other: Self) -> u64 {
        u64::from(
            self.tokens[0]
                .saturating_sub(other.tokens[1])
                .max(other.tokens[0].saturating_sub(self.tokens[1])),
        )
    }
}

pub struct Occurrence {
    pub address: Address,
    pub point: [i16; POINT],
}

pub struct Row {
    pub words: Vec<Occurrence>,
    pub sites: Vec<Occurrence>,
    pub edges: Vec<[i16; EDGE]>,
    pub tuples: Vec<[usize; 3]>,
}

pub struct Sign {
    pub lexicon: Vec<[i16; POINT]>,
    pub transport: Vec<[i16; EDGE]>,
}

pub type Bank = Vec<[i16; POINT]>;

fn check(value: bool, reason: &str) -> io::Result<()> {
    if value {
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, reason))
    }
}

struct Reader<'a>(Cursor<&'a [u8]>);

impl<'a> Reader<'a> {
    fn new(blob: &'a [u8]) -> Self {
        Self(Cursor::new(blob))
    }

    fn bytes<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let mut bytes = [0; N];
        self.0.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn word(&mut self) -> io::Result<u32> {
        self.bytes().map(u32::from_le_bytes)
    }

    fn count(&mut self, max: usize) -> io::Result<usize> {
        let n = self.word()? as usize;
        check(n <= max, "integer count or ordinal exceeds packet bounds")?;
        Ok(n)
    }

    fn point<const N: usize>(&mut self) -> io::Result<[i16; N]> {
        let mut values = [0; N];
        for value in &mut values {
            *value = i16::from_le_bytes(self.bytes()?);
        }
        Ok(values)
    }

    fn points<const N: usize>(&mut self, count: usize) -> io::Result<Vec<[i16; N]>> {
        (0..count).map(|_| self.point()).collect()
    }

    fn slots<const N: usize>(&mut self, capacity: usize) -> io::Result<Vec<[i16; N]>> {
        let active = self.count(capacity)?;
        let mut slots = self.points(capacity)?;
        check(
            slots[active..]
                .iter()
                .flatten()
                .all(|&coordinate| coordinate == 0),
            "nonzero inactive coordinate in restored state",
        )?;
        slots.truncate(active);
        Ok(slots)
    }

    fn occurrences(&mut self, count: usize, frozen: bool) -> io::Result<Vec<Occurrence>> {
        let mut seen = BTreeSet::new();
        (0..count)
            .map(|_| {
                let bytes = [self.word()?, self.word()?];
                let tokens = [self.word()?, self.word()?];
                check(
                    bytes[0] < bytes[1] && tokens[0] < tokens[1],
                    "empty occurrence",
                )?;
                check(seen.insert(bytes), "duplicate occurrence address")?;
                let point = self.point()?;
                check(
                    !frozen || point.iter().all(|value| (-24..=24).contains(value)),
                    "site coordinate outside D11 point bound",
                )?;
                Ok(Occurrence {
                    address: Address { bytes, tokens },
                    point,
                })
            })
            .collect()
    }

    fn end(&self) -> io::Result<()> {
        check(
            self.0.position() == self.0.get_ref().len() as u64,
            "trailing packet bytes",
        )
    }
}

pub fn text(blob: &[u8]) -> io::Result<Vec<Row>> {
    let mut reader = Reader::new(blob);
    check(
        reader.bytes::<8>()? == *b"UORTXT01",
        "wrong blind text magic",
    )?;
    let rows = reader.count(512)?;
    check(rows > 0, "no blind rows")?;
    let mut remaining = 4_194_304;
    let mut parsed = Vec::with_capacity(rows);
    for _ in 0..rows {
        let word_count = reader.count(128)?;
        let site_count = reader.count(128)?;
        let tuple_count = reader.count(remaining)?;
        remaining -= tuple_count;
        let words = reader.occurrences(word_count, false)?;
        let sites = reader.occurrences(site_count, true)?;
        let edges = reader.points(word_count * site_count)?;
        let mut seen = BTreeSet::new();
        let mut tuples = Vec::new();
        for _ in 0..tuple_count {
            let tuple = [
                reader.count(site_count.saturating_sub(1))?,
                reader.count(site_count.saturating_sub(1))?,
                reader.count(site_count.saturating_sub(1))?,
            ];
            let [a, e, h] = tuple;
            check(a != e && a != h && h != e, "repeated site in native tuple")?;
            check(seen.insert(tuple), "duplicate native tuple")?;
            tuples.push(tuple);
        }
        parsed.push(Row {
            words,
            sites,
            edges,
            tuples,
        });
    }
    reader.end()?;
    Ok(parsed)
}

pub fn state(blob: &[u8]) -> io::Result<[Sign; 2]> {
    let mut reader = Reader::new(blob);
    check(
        reader.bytes::<8>()? == *b"UORSTT01",
        "wrong learned state magic",
    )?;
    let mut sign = || {
        Ok::<_, io::Error>(Sign {
            lexicon: reader.slots(WORD_SLOTS)?,
            transport: reader.slots(EDGE_SLOTS)?,
        })
    };
    let signs = [sign()?, sign()?];
    reader.end()?;
    Ok(signs)
}

pub fn pooled(blob: &[u8]) -> io::Result<[Bank; 3]> {
    let mut reader = Reader::new(blob);
    check(reader.word()? == 1, "unknown frozen fit schema")?;
    let mut scope = [Vec::new(), Vec::new(), Vec::new()];
    // Historical schema 1 has eight folds and then the pooled seven-bank fit.
    // Every prototype includes a training origin ID before its i16 coordinates.
    for fold in 0..9 {
        for bank in 0..7 {
            let count = reader.count(blob.len() / (4 + 2 * POINT))?;
            check(count > 0, "empty historical bank")?;
            for _ in 0..count {
                reader.word()?;
                let point = reader.point::<POINT>()?;
                check(
                    point
                        .iter()
                        .all(|coordinate| (-24..=24).contains(coordinate)),
                    "frozen coordinate outside D11 point bound",
                )?;
                if fold == 8 && bank < scope.len() {
                    scope[bank].push(point);
                }
            }
        }
    }
    reader.end()?;
    Ok(scope)
}
