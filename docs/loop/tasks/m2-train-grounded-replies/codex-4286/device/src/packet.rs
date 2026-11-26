use crate::{require, save, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
    io::{Read, Write},
    path::Path,
};
use uor_r4_integer::role_action::instruction::learned::Point;

pub const VERSION: u32 = 2;
pub const MISSING: u32 = u32::MAX;
const MAX_ROWS: usize = 512;
const MAX_POINTS: usize = 128;
const MAX_TRIPLES: usize = 1 << 22;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn from_pair([start, end]: [u32; 2]) -> Self {
        Self { start, end }
    }

    pub fn frame(self) -> Result<Self> {
        Ok(Self {
            start: self.start.checked_add(1).ok_or("invalid start byte")?,
            end: self.end.checked_add(1).ok_or("invalid end byte")?,
        })
    }

    pub fn valid(self) -> bool {
        self.start < self.end
    }

    pub fn text(self, source: &str) -> Result<&str> {
        require(self.valid(), "empty source span")?;
        source
            .get(self.start as usize..self.end as usize)
            .ok_or_else(|| "span is outside the exact source text".into())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Address {
    pub bytes: Span,
    pub tokens: Span,
}

#[derive(Clone, Copy)]
pub struct Site {
    pub address: Address,
    pub point: Point,
}

pub type Tuple = [usize; 3];

pub struct Row {
    pub sites: Vec<Site>,
    pub tuples: Vec<Tuple>,
}

impl Row {
    pub fn addresses(&self, tuple: Tuple) -> [Address; 3] {
        tuple.map(|i| self.sites[i].address)
    }

    pub fn contains(&self, target: [Address; 3]) -> bool {
        self.tuples
            .iter()
            .any(|&tuple| self.addresses(tuple) == target)
    }

    fn validate(&self, budget: &mut usize) -> Result<()> {
        require(
            (3..=MAX_POINTS).contains(&self.sites.len()),
            "invalid site count",
        )?;
        require(
            (2..=*budget).contains(&self.tuples.len()),
            "invalid proposal count",
        )?;
        *budget -= self.tuples.len();
        let mut addresses = BTreeSet::new();
        for s in &self.sites {
            require(
                s.address.bytes.valid()
                    && s.address.bytes.end <= 4096
                    && s.address.tokens.valid()
                    && s.address.tokens.end <= 1 << 20
                    && addresses.insert(s.address.bytes)
                    && s.point.iter().flatten().all(|&x| (-24..=24).contains(&x)),
                "invalid or duplicate proposer site",
            )?;
        }
        let mut tuples = BTreeSet::new();
        for &[a, e, h] in &self.tuples {
            require(
                a < self.sites.len()
                    && e < self.sites.len()
                    && h < self.sites.len()
                    && a != e
                    && a != h
                    && e != h
                    && tuples.insert([a, e, h]),
                "invalid or duplicate original proposal",
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Edge {
    pub governor: Span,
    pub noun: Span,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Input {
    Native,
    Governors([Span; 2]),
    Nouns([Span; 2]),
    Links {
        governors: [Span; 2],
        edges: [Edge; 2],
    },
}

pub const ARMS: [&str; 5] = [
    "native",
    "governor_only",
    "nouns_only",
    "correct_link",
    "exchanged_link",
];

impl Input {
    fn input_type(self) -> u32 {
        match self {
            Self::Native => 0,
            Self::Governors(_) => 1,
            Self::Nouns(_) => 2,
            Self::Links { .. } => 3,
        }
    }

    fn validate(self) -> Result<()> {
        let pair = |[x, y]: [Span; 2]| {
            x.valid() && y.valid() && x.end <= 1 << 20 && y.end <= 1 << 20 && x != y
        };
        require(
            match self {
                Self::Native => true,
                Self::Governors(g) | Self::Nouns(g) => pair(g),
                Self::Links { governors, edges } => {
                    pair(governors)
                        && pair(edges.map(|edge| edge.governor))
                        && pair(edges.map(|edge| edge.noun))
                }
            },
            "invalid or ambiguous addressed intervention",
        )
    }

    pub fn permute(self) -> Self {
        match self {
            Self::Nouns([a, b]) => Self::Nouns([b, a]),
            Self::Links {
                governors,
                edges: [a, b],
            } => Self::Links {
                governors,
                edges: [b, a],
            },
            other => other,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Label {
    pub case_id: String,
    pub pair_id: String,
    pub family: String,
    pub distance: String,
    pub admission: String,
    pub prompt_sha256: String,
    pub gold: [Address; 3],
    pub neutral: Address,
}

impl Label {
    pub fn exchanged(&self) -> [Address; 3] {
        let [a, e, h] = self.gold;
        [e, a, h]
    }
}

pub fn word(input: &mut impl Read) -> Result<u32> {
    let mut buf = [0; 4];
    input.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub fn write_word(out: &mut impl Write, value: u32) -> Result<()> {
    out.write_all(&value.to_le_bytes())?;
    Ok(())
}

pub fn end(input: &mut impl Read) -> Result<()> {
    require(
        input.read(&mut [0])? == 0,
        "unexpected trailing packet data",
    )
}

pub fn count(input: &mut impl Read) -> Result<usize> {
    Ok(usize::try_from(word(input)?)?)
}

pub fn start(input: &mut impl Read) -> Result<usize> {
    require(word(input)? == VERSION, "unrecognized packet version")?;
    let n = count(input)?;
    require((1..=MAX_ROWS).contains(&n), "invalid packet row count")?;
    Ok(n)
}

pub fn read_point(input: &mut impl Read) -> Result<Point> {
    let mut point = [[0; 32]; 3];
    for x in point.iter_mut().flatten() {
        let mut buf = [0; 2];
        input.read_exact(&mut buf)?;
        *x = i16::from_le_bytes(buf);
    }
    Ok(point)
}

pub fn write_point(out: &mut impl Write, point: Point) -> Result<()> {
    for x in point.into_iter().flatten() {
        out.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

pub fn read_span(input: &mut impl Read) -> Result<Span> {
    Ok(Span {
        start: word(input)?,
        end: word(input)?,
    })
}

fn write_span(out: &mut impl Write, span: Span) -> Result<()> {
    write_word(out, span.start)?;
    write_word(out, span.end)
}

fn write_count(out: &mut impl Write, len: usize) -> Result<()> {
    write_word(out, u32::try_from(len)?)
}

fn header(n: usize) -> Result<Vec<u8>> {
    require((1..=MAX_ROWS).contains(&n), "invalid output row count")?;
    let mut out = Vec::new();
    write_word(&mut out, VERSION)?;
    write_count(&mut out, n)?;
    Ok(out)
}

pub fn store(path: &Path, rows: &[Row], notes: &[Label], arms: &[[Input; 5]]) -> Result<()> {
    require(
        rows.len() == notes.len() && rows.len() == arms.len(),
        "row census changed before export",
    )?;
    fs::create_dir(path)?;
    let mut geometry = header(rows.len())?;
    let mut budget = MAX_TRIPLES;
    for row in rows {
        row.validate(&mut budget)?;
        write_count(&mut geometry, row.sites.len())?;
        for site in &row.sites {
            write_span(&mut geometry, site.address.bytes)?;
            write_span(&mut geometry, site.address.tokens)?;
            write_point(&mut geometry, site.point)?;
        }
        write_count(&mut geometry, row.tuples.len())?;
        for tuple in &row.tuples {
            for &i in tuple {
                write_count(&mut geometry, i)?;
            }
        }
    }
    save(&path.join("geometry.bin"), &geometry)?;
    save(
        &path.join("labels.json"),
        &serde_json::to_vec_pretty(notes)?,
    )?;
    for (a, name) in ARMS.iter().enumerate() {
        write_arm(path, name, arms.iter().map(|row| row[a]), false)?;
        if a >= 2 {
            write_arm(path, name, arms.iter().map(|row| row[a]), true)?;
        }
    }
    Ok(())
}

pub fn write_arm(
    path: &Path,
    name: &str,
    rows: impl ExactSizeIterator<Item = Input>,
    permute: bool,
) -> Result<()> {
    let mut out = header(rows.len())?;
    for row in rows {
        let row = if permute { row.permute() } else { row };
        row.validate()?;
        write_word(&mut out, row.input_type())?;
        match row {
            Input::Native => {}
            Input::Governors(spans) | Input::Nouns(spans) => {
                for span in spans {
                    write_span(&mut out, span)?;
                }
            }
            Input::Links { governors, edges } => {
                for span in governors {
                    write_span(&mut out, span)?;
                }
                for edge in edges {
                    write_span(&mut out, edge.governor)?;
                    write_span(&mut out, edge.noun)?;
                }
            }
        }
    }
    let suffix = if permute { ".permuted" } else { "" };
    save(&path.join(format!("{name}{suffix}.bin")), &out)
}

pub fn read_arm(input: &mut impl Read, ordinal: usize, n: usize) -> Result<Vec<Input>> {
    require(start(input)? == n, "arm row count changed")?;
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let row = match (ordinal, word(input)?) {
            (0, 0) => Input::Native,
            (1, 1) => Input::Governors([read_span(input)?, read_span(input)?]),
            (2, 2) => Input::Nouns([read_span(input)?, read_span(input)?]),
            (3 | 4, 3) => {
                let governors = [read_span(input)?, read_span(input)?];
                let mut edge = || -> Result<Edge> {
                    Ok(Edge {
                        governor: read_span(input)?,
                        noun: read_span(input)?,
                    })
                };
                Input::Links {
                    governors,
                    edges: [edge()?, edge()?],
                }
            }
            _ => return Err("arm input contract changed".into()),
        };
        row.validate()?;
        rows.push(row);
    }
    end(input)?;
    Ok(rows)
}

pub fn read_geometry(input: &mut impl Read) -> Result<Vec<Row>> {
    let n = start(input)?;
    let mut budget = MAX_TRIPLES;
    let mut output = Vec::with_capacity(n);
    for _ in 0..n {
        let num = count(input)?;
        require((3..=MAX_POINTS).contains(&num), "invalid site count")?;
        let mut sites = Vec::with_capacity(num);
        for _ in 0..num {
            let address = Address {
                bytes: read_span(input)?,
                tokens: read_span(input)?,
            };
            sites.push(Site {
                address,
                point: read_point(input)?,
            });
        }
        let num = count(input)?;
        require((2..=budget).contains(&num), "invalid proposal count")?;
        let mut tuples = Vec::with_capacity(num);
        for _ in 0..num {
            tuples.push([count(input)?, count(input)?, count(input)?]);
        }
        let row = Row { sites, tuples };
        row.validate(&mut budget)?;
        output.push(row);
    }
    end(input)?;
    Ok(output)
}
