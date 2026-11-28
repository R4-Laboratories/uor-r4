#![forbid(unsafe_code)]
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::{env, fs};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const ROWS: usize = 3886;
const BAR: usize = 3109;

#[derive(Debug)]
struct Side {
    opening: u32,
    group: u32,
    carry: u16,
    ids: [u16; 2],
    roots: [u16; 2],
    addresses: [[u8; 5]; 2],
    delta: i64,
    bound: i32,
}

fn slots(value: &str) -> Result<[u8; 5]> {
    let fields = value
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or("slots syntax")?;
    let values = fields
        .split(", ")
        .map(str::parse::<u8>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if values.iter().any(|&x| x >= 64) {
        return Err("slot out of range".into());
    }
    values.try_into().map_err(|_| "slot count".into())
}

type Pair = [Side; 2];
type Witness = (Vec<Pair>, Vec<Side>);

fn witness(path: &str, arm: &str) -> Result<Witness> {
    let content = fs::read_to_string(path)?;
    let mut lines = content.lines();
    if lines.next() != Some("arm\tkind\tpair\tordinal\tgroup\tcarry\tgold\trival\tgold_root\trival_root\tgold_slots\trival_slots\tgold_prior_q48\trival_prior_q48\tstrict_min\tkey") {
        return Err("witness header mismatch".into());
    }
    let mut records = BTreeMap::<u32, Vec<Side>>::new();
    let mut aliases = Vec::new();
    let mut seen = BTreeSet::new();
    for line in lines {
        let columns = line.split('\t').collect::<Vec<_>>();
        let &[actual_arm, kind, number, opening, group, carry, gold, rival, gold_root, rival_root, gs, rs, gp, rp, bound, _] =
            columns.as_slice()
        else {
            return Err("bad witness row".into());
        };
        let delta = rp.parse::<i64>()? - gp.parse::<i64>()?;
        let s = Side {
            opening: opening.parse()?,
            group: group.parse()?,
            carry: carry.parse()?,
            ids: [gold.parse()?, rival.parse()?],
            roots: [gold_root.parse()?, rival_root.parse()?],
            addresses: [slots(gs)?, slots(rs)?],
            delta,
            bound: bound.parse()?,
        };
        let number: u32 = number.parse()?;
        if actual_arm != arm
            || !seen.insert(s.opening)
            || i64::from(s.bound) != delta.div_euclid(1 << 32) + 1
        {
            return Err("wrong arm, repeated opening or wrong bound".into());
        }
        match (kind, number) {
            ("singleton", 0) if s.addresses[0] == s.addresses[1] && s.bound > 0 => aliases.push(s),
            ("pair", n) if n > 0 => records.entry(n).or_default().push(s),
            _ => return Err("invalid witness kind".into()),
        }
    }
    let mut pairs = Vec::new();
    for (number, sides) in records {
        if number as usize != pairs.len() + 1 {
            return Err("nonconsecutive proof IDs".into());
        }
        let pair: Pair = sides.try_into().map_err(|_| "pair cardinality")?;
        let [l, r] = &pair;
        if l.group == r.group
            || l.bound + r.bound <= 0
            || !(0..5).all(|b| {
                let (lg, lr, rg, rr) = (
                    l.addresses[0][b],
                    l.addresses[1][b],
                    r.addresses[0][b],
                    r.addresses[1][b],
                );
                (lg == lr && rg == rr) || (lg == rr && lr == rg)
            })
        {
            return Err("pair is not a verified opposite contrast".into());
        }
        pairs.push(pair);
    }
    Ok((pairs, aliases))
}

fn normalized<T: Ord>(mut values: [T; 2]) -> [T; 2] {
    values.sort_unstable();
    values
}

fn row_ids(pair: &Pair) -> [u32; 2] {
    normalized(pair.each_ref().map(|s| s.opening))
}

fn full_ids(pair: &Pair) -> [(u32, [u16; 2]); 2] {
    normalized(pair.each_ref().map(|s| (s.opening, s.ids)))
}

fn main() -> Result<()> {
    let args = env::args().collect::<Vec<_>>();
    let [_, baseline, product, unary] = args.as_slice() else {
        return Err("usage: attribution BASELINE PRODUCT UNARY".into());
    };
    let (base, base_aliases) = witness(baseline, "baseline")?;
    let (prod, aliases) = witness(product, "product")?;
    let (unary, unary_aliases) = witness(unary, "unary")?;
    let base_rows: BTreeSet<_> = base.iter().map(row_ids).collect();
    let base_full: BTreeSet<_> = base.iter().map(full_ids).collect();
    let unary_full: BTreeSet<_> = unary.iter().map(full_ids).collect();
    let mut classes = [[0usize; 2]; 2];
    let mut tuple_real = 0;
    let mut same_carry = 0;
    let mut same_group = 0;
    let mut crossed_roots = [0usize; 3];
    let mut crossed_ids = [0usize; 3];
    let mut contrast_banks = [0usize; 5];
    let mut tuple_by_contrast = [0usize; 5];
    let mut root_aliases = 0;
    let mut baseline_exact = 0;
    let mut baseline_rows = 0;
    let mut unary_exact = 0;
    let mut examples = [None, None, None];
    for (i, pair) in prod.iter().enumerate() {
        let [l, r] = pair;
        let direct = l.addresses[0][..4] == r.addresses[1][..4]
            && l.addresses[1][..4] == r.addresses[0][..4];
        let real = l.delta + r.delta >= 0;
        let fifth_changed = l.addresses[0][4] != l.addresses[1][4];
        classes[usize::from(direct)][usize::from(fifth_changed)] += 1;
        tuple_real += usize::from(direct && real);
        same_carry += usize::from(l.carry == r.carry);
        same_group += usize::from(l.group == r.group);
        let crossed =
            |a: &[u16; 2], b: &[u16; 2]| usize::from(a[0] == b[1]) + usize::from(a[1] == b[0]);
        crossed_roots[crossed(&l.roots, &r.roots)] += 1;
        crossed_ids[crossed(&l.ids, &r.ids)] += 1;
        root_aliases += pair.iter().filter(|s| s.roots[0] == s.roots[1]).count();
        let differing = (0..4)
            .filter(|&b| l.addresses[0][b] != l.addresses[1][b])
            .count();
        contrast_banks[differing] += 1;
        tuple_by_contrast[differing] += usize::from(direct);
        baseline_exact += usize::from(base_full.contains(&full_ids(pair)));
        baseline_rows += usize::from(base_rows.contains(&row_ids(pair)));
        unary_exact += usize::from(unary_full.contains(&full_ids(pair)));
        let class = if direct { usize::from(!real) } else { 2 };
        if examples[class].is_none() {
            examples[class] = Some((i + 1, pair));
        }
    }
    let tuple_integer = classes[1].iter().sum::<usize>();
    let alias_real = aliases.iter().filter(|s| s.delta >= 0).count();
    println!("selected pairs baseline={} product={} unary={}; singletons baseline={} product={} unary={}", base.len(),prod.len(),unary.len(),base_aliases.len(),aliases.len(),unary_aliases.len());
    println!(
        "four-tuple irreducible integer={} real={}; integer upper={} real upper={} (bar {BAR})",
        tuple_integer + aliases.len(),
        tuple_real + alias_real,
        ROWS - tuple_integer - aliases.len(),
        ROWS - tuple_real - alias_real
    );
    println!("row classes [[non-reversal constant fifth, non-reversal changed fifth], [reversal constant fifth, reversal changed fifth]] = {classes:?}");
    println!("same snapped carry {same_carry}; same group {same_group}; crossed equal candidate roots {crossed_roots:?}; crossed equal candidate IDs {crossed_ids:?}; within-side root aliases {root_aliases}");
    println!("differing-bank histogram {contrast_banks:?}; exact tuple reversals by differing banks {tuple_by_contrast:?}");
    println!("baseline row overlaps {baseline_rows}; baseline exact witness overlaps {baseline_exact}; unary exact overlaps {unary_exact}");
    for (name, example) in [
        "real irreducible",
        "integer-only irreducible",
        "no direct tuple bound",
    ]
    .into_iter()
    .zip(examples)
    {
        println!("{name}: {example:?}");
    }
    Ok(())
}
