#![forbid(unsafe_code)]
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

const ROWS: usize = 3886;
const IDS: usize = 4096;
const RECORD: usize = 12 + 8 * IDS;
const UNIT: i64 = 1 << 32;
const SUPPORT: [u8; 16] = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1, 0];
type Form = BTreeMap<(usize, u8), i8>;

macro_rules! require {
    ($condition:expr, $($message:tt)*) => {
        if !$condition { return Err(format!($($message)*).into()); }
    };
}

struct Frozen {
    tables: Vec<u8>,
    roots: Vec<u8>,
    rows: Vec<u8>,
    ordinals: BTreeMap<u32, usize>,
}
impl Frozen {
    fn read(table: &str, roots: &str, rows: &str) -> Result<Self> {
        let tables = fs::read(table)?;
        let roots = fs::read(roots)?;
        let rows = fs::read(rows)?;
        require!(
            tables.len() == 4352 && tables.iter().all(|&x| x < 64),
            "table format"
        );
        require!(
            roots.len() == 2 * IDS && rows.len() == ROWS * RECORD,
            "frozen tape format"
        );
        let mut ordinals = BTreeMap::new();
        let mut previous = None;
        for (position, row) in rows.chunks_exact(RECORD).enumerate() {
            let ordinal = u32::from_le_bytes(row[..4].try_into()?);
            let gold = u16::from_le_bytes(row[8..10].try_into()?);
            require!(
                previous.is_none_or(|prev| prev < ordinal) && (1..4096).contains(&gold),
                "invalid row {position}"
            );
            require!(row[12..20] == [0; 8], "nonzero reserved ID");
            previous = Some(ordinal);
            ordinals.insert(ordinal, position);
        }
        Ok(Self {
            tables,
            roots,
            rows,
            ordinals,
        })
    }

    fn root(&self, id: u16) -> Result<u16> {
        require!((1..4096).contains(&id), "invalid candidate ID {id}");
        let start = 2 * usize::from(id);
        Ok(u16::from_le_bytes(self.roots[start..start + 2].try_into()?))
    }

    fn slots(&self, carry: u16, root: u16, arm: &str) -> [u8; 5] {
        let mut words = [0; 5];
        for (bank, word) in words[..4].iter_mut().enumerate() {
            let source = usize::from((carry >> (4 * bank)) & 15);
            let target = usize::from((root >> (4 * bank)) & 15);
            *word = self.tables[16 * source + target];
        }
        words[4] = match arm {
            "product" => self.tables[256 + 64 * usize::from(words[0]) + usize::from(words[1])],
            "unary" => 4 * (carry & 15) as u8 + SUPPORT[usize::from(root & 15)],
            _ => 0,
        };
        words
    }

    fn site(&self, ordinal: u32, rival: u16, arm: &str) -> Result<Site> {
        let index = *self
            .ordinals
            .get(&ordinal)
            .ok_or_else(|| format!("unknown ordinal {ordinal}"))?;
        let row = &self.rows[index * RECORD..(index + 1) * RECORD];
        let group = u32::from_le_bytes(row[4..8].try_into()?);
        let gold = u16::from_le_bytes(row[8..10].try_into()?);
        let carry = u16::from_le_bytes(row[10..12].try_into()?);
        require!(gold != rival, "self comparison at {ordinal}");
        let roots = [self.root(gold)?, self.root(rival)?];
        let slots = roots.map(|root| self.slots(carry, root, arm));
        let mass = |id| -> Result<u64> {
            let start = 12 + 8 * usize::from(id);
            let value = u64::from_le_bytes(row[start..start + 8].try_into()?);
            require!(
                (1..=1 << 48).contains(&value),
                "invalid mass at {ordinal}/{id}"
            );
            Ok(value)
        };
        let priors = [mass(gold)?, mass(rival)?];
        let bound = ((priors[1] as i64 - priors[0] as i64).div_euclid(UNIT) + 1) as i32;
        let mut form = Form::new();
        for bank in 0..5 {
            if slots[0][bank] != slots[1][bank] {
                form.insert((bank, slots[0][bank]), 1);
                form.insert((bank, slots[1][bank]), -1);
            }
        }
        Ok(Site {
            index,
            group,
            gold,
            carry,
            roots,
            slots,
            priors,
            bound,
            form,
        })
    }
}

struct Site {
    index: usize,
    group: u32,
    gold: u16,
    carry: u16,
    roots: [u16; 2],
    slots: [[u8; 5]; 2],
    priors: [u64; 2],
    bound: i32,
    form: Form,
}
impl Site {
    fn key(&self) -> Result<u64> {
        let mut words = [0u64; 5];
        let mut inverse = [0u64; 5];
        for (&(bank, address), &sign) in &self.form {
            let address = u64::from(address);
            let (g, r) = if sign == 1 {
                (address * 64, address)
            } else {
                (address, address * 64)
            };
            words[bank] += g;
            inverse[bank] += r;
        }
        words.reverse();
        inverse.reverse();
        let reverse = inverse < words;
        Ok(if reverse { inverse } else { words }
            .iter()
            .fold(0, |sum, word| 4096 * sum + word)
            * 2
            + u64::from(reverse))
    }

    fn check_edge(&self, tape: &mut File, rival: u16) -> Result<()> {
        let offset = 16 * (self.index * (IDS - 1) + usize::from(rival - 1));
        tape.seek(SeekFrom::Start(offset as u64))?;
        let mut entry = [0; 16];
        tape.read_exact(&mut entry)?;
        require!(
            u64::from_le_bytes(entry[..8].try_into()?) == self.key()?
                && usize::from(u16::from_le_bytes(entry[8..10].try_into()?)) == self.index
                && u16::from_le_bytes(entry[10..12].try_into()?) == rival
                && i32::from_le_bytes(entry[12..].try_into()?) == self.bound,
            "GPU edge mismatch at row {} rival {rival}",
            self.index
        );
        Ok(())
    }
}

fn edge_file(path: &str) -> Result<File> {
    let file = File::open(path)?;
    require!(
        file.metadata()?.len() == (ROWS * (IDS - 1) * 16) as u64,
        "wrong GPU edge tape length"
    );
    Ok(file)
}

fn cancelling(left: &Site, right: &Site) -> bool {
    left.group != right.group
        && left.bound + right.bound > 0
        && left.form.len() == right.form.len()
        && left
            .form
            .iter()
            .all(|(word, sign)| right.form.get(word) == Some(&-*sign))
}

fn protected(frozen: &Frozen, path: &str, raw: &str) -> Result<()> {
    let source = fs::read_to_string(path)?;
    let mut lines = source.lines();
    require!(lines.next() == Some("pair_id\tordinal\tgroup\tgold_id\trival_id\tgold_packed\trival_packed\tgold_prior_q48\trival_prior_q48\tcarry"), "protected schema");
    let mut tape = edge_file(raw)?;
    let mut pending = None;
    let mut seen = BTreeSet::new();
    for (member, line) in lines.enumerate() {
        let values = line
            .split('\t')
            .map(str::parse::<u64>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let &[pair, ordinal, group, gold, rival, gp, rp, gm, rm, carry] = values.as_slice() else {
            return Err("protected field count".into());
        };
        require!(
            pair == (member / 2) as u64 && seen.insert(ordinal),
            "protected ordering"
        );
        let rival = u16::try_from(rival)?;
        let site = frozen.site(u32::try_from(ordinal)?, rival, "baseline")?;
        let pack = |slots: [u8; 5]| {
            slots[..4]
                .iter()
                .enumerate()
                .map(|(b, &a)| u64::from(a) << (6 * b))
                .sum::<u64>()
        };
        require!(
            u64::from(site.group) == group
                && u64::from(site.gold) == gold
                && site.slots.map(pack) == [gp, rp]
                && site.priors == [gm, rm]
                && u64::from(site.carry) == carry,
            "protected frozen row mismatch at {ordinal}"
        );
        site.check_edge(&mut tape, rival)?;
        if let Some(left) = pending.take() {
            require!(
                cancelling(&left, &site),
                "protected contradiction failed at pair {pair}"
            );
        } else {
            pending = Some(site);
        }
    }
    require!(
        seen.len() == 2668 && pending.is_none(),
        "protected coverage"
    );
    println!("baseline: all 2668 historical sites and 1334 disjoint proofs reproduced in GPU constraint tape");
    Ok(())
}

fn certificates(frozen: &Frozen, arm: &str, raw: &str, path: &str) -> Result<()> {
    require!(
        ["baseline", "product", "unary"].contains(&arm),
        "unknown arm"
    );
    let source = fs::read_to_string(path)?;
    let mut lines = source.lines();
    require!(lines.next() == Some("arm\tkind\tpair\tordinal\tgroup\tcarry\tgold\trival\tgold_root\trival_root\tgold_slots\trival_slots\tgold_prior_q48\trival_prior_q48\tstrict_min\tkey"), "certificate schema");
    let mut tape = edge_file(raw)?;
    let mut matching = BTreeMap::<u32, Vec<Site>>::new();
    let mut ordinals = BTreeSet::new();
    let mut singletons = 0;
    for line in lines {
        let cols: Vec<_> = line.split('\t').collect();
        let &[seen_arm, kind, pair, ordinal, group, carry, gold, rival, gold_root, rival_root, gold_slots, rival_slots, gold_prior, rival_prior, bound, key] =
            cols.as_slice()
        else {
            return Err("certificate field count".into());
        };
        let ordinal = ordinal.parse::<u32>()?;
        let pair = pair.parse::<u32>()?;
        let rival = rival.parse::<u16>()?;
        require!(
            arm == seen_arm && ordinals.insert(ordinal),
            "certificate arm or duplicate ordinal {ordinal}"
        );
        let site = frozen.site(ordinal, rival, arm)?;
        require!(
            site.group == group.parse::<u32>()?
                && site.carry == carry.parse::<u16>()?
                && site.gold == gold.parse::<u16>()?
                && site.roots == [gold_root.parse()?, rival_root.parse()?]
                && format!("{:?}", site.slots[0]) == gold_slots
                && format!("{:?}", site.slots[1]) == rival_slots
                && site.priors == [gold_prior.parse()?, rival_prior.parse()?]
                && site.bound == bound.parse::<i32>()?
                && format!("{:016x}", site.key()?) == key,
            "certificate mismatch at {ordinal}/{rival}"
        );
        site.check_edge(&mut tape, rival)?;
        match kind {
            "singleton" => {
                require!(
                    pair == 0 && site.form.is_empty() && site.bound > 0,
                    "invalid singleton at {ordinal}"
                );
                singletons += 1;
            }
            "pair" => {
                require!(pair > 0, "invalid pair sequence at {ordinal}");
                matching.entry(pair).or_default().push(site);
            }
            _ => return Err("unknown certificate kind".into()),
        }
    }
    for (i, (&pair, sides)) in matching.iter().enumerate() {
        require!(pair as usize == i + 1, "invalid pair number {pair}");
        require!(
            sides.len() == 2 && cancelling(&sides[0], &sides[1]),
            "invalid pair proof {pair}"
        );
    }
    let pairs = matching.len();
    println!("{arm}: verified {singletons} singleton losses and {pairs} disjoint pairs, strict win upper bound {}", ROWS - singletons - pairs);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let [_, tables, roots, rows, history, baseline, receipts @ ..] = args.as_slice() else {
        return Err("usage: census_audit TABLE ROOTS ROWS PROTECTED BASELINE_EDGES [ARM EDGES CERTIFICATES]...".into());
    };
    require!(receipts.len() % 3 == 0, "incomplete receipt triple");
    let frozen = Frozen::read(tables, roots, rows)?;
    protected(&frozen, history, baseline)?;
    for chunk in receipts.chunks_exact(3) {
        certificates(&frozen, &chunk[0], &chunk[1], &chunk[2])?;
    }
    Ok(())
}
