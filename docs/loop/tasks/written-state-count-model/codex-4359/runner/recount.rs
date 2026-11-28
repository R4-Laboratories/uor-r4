#![forbid(unsafe_code)]
// Independent host audit of already-collected GPU evidence.
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::{env, fs, io};

const PAIRS: usize = 1334;
const OPENINGS: usize = 3886;
const UNIT: i64 = 1 << 32;
const INPUT_HEADER: &str = "pair_id\tordinal\tgroup\tgold_id\trival_id\tgold_packed\trival_packed\tgold_prior_q48\trival_prior_q48\tcarry";
const RAW_HEADER: &str = "pair_id\tordinal_l\tstrict_l\ttie_aware_l\tgold_joint_l\trival_joint_l\tgold_unary_l\trival_unary_l\tordinal_r\tstrict_r\ttie_aware_r\tgold_joint_r\trival_joint_r\tgold_unary_r\trival_unary_r\topposite_mask\tstrict_mask\ttie_aware_mask";

fn require(ok: bool, message: impl Into<String>) -> io::Result<()> {
    if ok {
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, message.into()))
    }
}

fn records<T, const N: usize>(path: &str, header: &str) -> Result<Vec<[T; N]>, Box<dyn Error>>
where
    T: std::str::FromStr + ToString,
    T::Err: Error + 'static,
{
    let input = fs::read_to_string(path)?;
    let mut lines = input.lines();
    require(
        lines.next() == Some(header),
        format!("unexpected header in {path}"),
    )?;
    lines
        .enumerate()
        .map(|(number, line)| {
            let mut fields = Vec::new();
            for field in line.split('\t') {
                let value: T = field.parse()?;
                require(
                    value.to_string() == field,
                    format!("noncanonical field in {path}:{}", number + 2),
                )?;
                fields.push(value);
            }
            fields
                .try_into()
                .map_err(|_| format!("wrong column count in {path}:{}", number + 2).into())
        })
        .collect()
}

type Address = [u8; 4];
type Incidence = BTreeMap<(usize, u32), i8>;

struct Tables {
    roots: [BTreeMap<u8, u8>; 16],
    composition: [u8; 4096],
}

impl Tables {
    fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = fs::read(path)?;
        let [bfs, composition] = file
            .split_at_checked(256)
            .map(|(a, b)| [a, b])
            .ok_or("truncated tables")?;
        require(
            composition.len() == 4096 && file.iter().all(|&x| x < 64),
            "invalid native-tables.bin",
        )?;
        let roots = std::array::from_fn(|source| {
            bfs[16 * source..16 * (source + 1)]
                .iter()
                .copied()
                .zip(0..16)
                .collect()
        });
        require(
            roots.iter().all(|row: &BTreeMap<_, _>| row.len() == 16),
            "ambiguous BFS inverse",
        )?;
        Ok(Self {
            roots,
            composition: composition.try_into()?,
        })
    }

    fn product(&self, a: u8, b: u8) -> u8 {
        self.composition[64 * usize::from(a) + usize::from(b)]
    }

    fn read(&self, address: Address, carry: i64) -> Result<[u8; 2], Box<dyn Error>> {
        // Support classes in native root order.
        const CLASS: [u8; 16] = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1, 0];
        let mut unary = 0;
        for (bank, action) in address.into_iter().enumerate() {
            let source = ((carry >> (4 * bank)) & 15) as usize;
            let root = self.roots[source].get(&action).ok_or_else(|| {
                format!("unreachable action {action} at carry {carry} bank {bank}")
            })?;
            if bank == 0 {
                unary = 4 * source as u8 + CLASS[usize::from(*root)];
            }
        }
        Ok([self.product(address[0], address[1]), unary])
    }
}

struct Side {
    gold: Address,
    rival: Address,
    gold_extra: [u8; 2],
    rival_extra: [u8; 2],
    packed: [u32; 2],
    ordinal: i64,
    group: u64,
    ids: [i64; 2],
    delta: i64,
}

fn unpack(packed: u32) -> Address {
    std::array::from_fn(|bank| ((packed >> (6 * bank)) % 64) as u8)
}

impl Side {
    fn read(
        tables: &Tables,
        mut row: [i128; 10],
        pair: usize,
        ordinals: &mut BTreeSet<i64>,
    ) -> Result<Self, Box<dyn Error>> {
        let group = u64::try_from(row[2])?;
        row[2] = 0;
        require(
            row.iter().all(|value| i64::try_from(*value).is_ok()),
            "protected field exceeds range",
        )?;
        let [index, ordinal, _, gold_id, rival_id, gold_packed, rival_packed, gold_prior, rival_prior, carry] =
            row.map(|value| value as i64);
        require(
            index == pair as i64
                && (0..=i64::from(u32::MAX)).contains(&ordinal)
                && ordinals.insert(ordinal)
                && (1..4096).contains(&gold_id)
                && (1..4096).contains(&rival_id)
                && gold_id != rival_id
                && (0..1 << 24).contains(&gold_packed)
                && (0..1 << 24).contains(&rival_packed)
                && (0..=1 << 48).contains(&gold_prior)
                && (0..=1 << 48).contains(&rival_prior)
                && (0..1 << 16).contains(&carry),
            format!("invalid protected member for pair {pair}, ordinal {ordinal}"),
        )?;
        let packed = [gold_packed as u32, rival_packed as u32];
        let [gold, rival] = packed.map(unpack);
        Ok(Self {
            gold,
            rival,
            gold_extra: tables.read(gold, carry)?,
            rival_extra: tables.read(rival, carry)?,
            packed,
            ordinal,
            group,
            ids: [gold_id, rival_id],
            delta: rival_prior - gold_prior,
        })
    }

    fn strict(&self) -> i64 {
        self.delta.div_euclid(UNIT) + 1
    }

    fn tie_aware(&self) -> i64 {
        if self.ids[0] < self.ids[1] {
            -(-self.delta).div_euclid(UNIT)
        } else {
            self.strict()
        }
    }

    fn observed(&self) -> [i64; 7] {
        [
            self.ordinal,
            self.strict(),
            self.tie_aware(),
            i64::from(self.gold_extra[0]),
            i64::from(self.rival_extra[0]),
            i64::from(self.gold_extra[1]),
            i64::from(self.rival_extra[1]),
        ]
    }

    fn incidence(&self, family: usize) -> Incidence {
        let mut form = BTreeMap::new();
        let mut term = |bank, gold, rival| {
            *form.entry((bank, gold)).or_insert(0) += 1;
            *form.entry((bank, rival)).or_insert(0) -= 1;
        };
        if family == 2 {
            term(0, self.packed[0], self.packed[1]);
        } else {
            for bank in 0..4 {
                let relabel = |action| {
                    if family == 4 {
                        (5 * u32::from(action) + 1 + 7 * bank as u32) % 64
                    } else {
                        u32::from(action)
                    }
                };
                term(bank, relabel(self.gold[bank]), relabel(self.rival[bank]));
            }
            if family == 1 || family == 3 {
                let fifth = usize::from(family == 3);
                term(
                    4,
                    u32::from(self.gold_extra[fifth]),
                    u32::from(self.rival_extra[fifth]),
                );
            }
        }
        form.retain(|_, coefficient| *coefficient != 0);
        form
    }
}

fn opposite(left: &Incidence, right: &Incidence) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .all(|(key, value)| right.get(key) == Some(&-*value))
}

fn zero_fifth(form: &Incidence) -> Incidence {
    form.iter()
        .filter(|((bank, _), _)| *bank < 4)
        .map(|(k, v)| (*k, *v))
        .collect()
}

fn mask(flags: [bool; 4]) -> i64 {
    flags
        .into_iter()
        .enumerate()
        .filter(|(_, set)| *set)
        .map(|(bit, _)| 1 << bit)
        .sum()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    let [_, protected, raw, table_file] = args.as_slice() else {
        return Err("usage: recount protected.tsv raw.tsv native-tables.bin".into());
    };
    let tables = Tables::load(table_file)?;
    let protected = records::<i128, 10>(protected, INPUT_HEADER)?;
    let raw = records::<i64, 18>(raw, RAW_HEADER)?;
    require(
        protected.len() == 2 * PAIRS && raw.len() == PAIRS,
        "wrong proof domain",
    )?;
    let mut ordinals = BTreeSet::new();
    let mut all_pairs = [0usize; 5];
    let mut opposed = [0usize; 4];
    let mut strict = [0usize; 4];
    let mut tie_aware = [0usize; 4];
    let mut joint_persisting = [0usize; 5];
    let mut joint_cleared = [0usize; 5];
    let mut unary_persisting = [0usize; 5];
    let mut unary_cleared = [0usize; 5];
    let mut tuple_real = 0;
    let mut tuple_real_tie_aware = 0;
    let mut states = [BTreeSet::new(), BTreeSet::new()];
    let mut witness = None;
    for (pair, gpu) in raw.iter().enumerate() {
        let l = Side::read(&tables, protected[2 * pair], pair, &mut ordinals)?;
        let r = Side::read(&tables, protected[2 * pair + 1], pair, &mut ordinals)?;
        require(
            l.group != r.group,
            format!("shared conversation in pair {pair}"),
        )?;
        let [base_l, joint_l, tuple_l, unary_l, renamed_l] =
            std::array::from_fn(|family| l.incidence(family));
        let [base_r, joint_r, tuple_r, unary_r, renamed_r] =
            std::array::from_fn(|family| r.incidence(family));
        let cancelling = [
            opposite(&base_l, &base_r),
            opposite(&joint_l, &joint_r),
            opposite(&tuple_l, &tuple_r),
            opposite(&unary_l, &unary_r),
        ];
        require(
            cancelling[0] && l.strict() + r.strict() > 0,
            format!("baseline proof lost at pair {pair}"),
        )?;
        require(
            opposite(&renamed_l, &renamed_r) == cancelling[0],
            format!("relabel failed at pair {pair}"),
        )?;
        require(
            zero_fifth(&joint_l) == base_l
                && zero_fifth(&joint_r) == base_r
                && zero_fifth(&unary_l) == base_l
                && zero_fifth(&unary_r) == base_r,
            format!("zero fifth failed at pair {pair}"),
        )?;
        let strict_flags = cancelling.map(|flag| flag && l.strict() + r.strict() > 0);
        let tie_flags = cancelling.map(|flag| flag && l.tie_aware() + r.tie_aware() > 0);
        require(
            gpu[0] == pair as i64
                && gpu[1..8] == l.observed()
                && gpu[8..15] == r.observed()
                && gpu[15..] == [mask(cancelling), mask(strict_flags), mask(tie_flags)],
            format!("GPU observation mismatch at pair {pair}"),
        )?;
        for family in 0..4 {
            opposed[family] += usize::from(cancelling[family]);
            strict[family] += usize::from(strict_flags[family]);
            tie_aware[family] += usize::from(tie_flags[family]);
        }
        let delta_sum = l.delta + r.delta;
        tuple_real += usize::from(cancelling[2] && delta_sum >= 0);
        tuple_real_tie_aware += usize::from(
            cancelling[2]
                && (delta_sum > 0
                    || (delta_sum == 0 && (l.ids[0] > l.ids[1] || r.ids[0] > r.ids[1]))),
        );
        let differing = l
            .gold
            .iter()
            .zip(l.rival)
            .filter(|(gold, rival)| **gold != *rival)
            .count();
        all_pairs[differing] += 1;
        if strict_flags[1] {
            joint_persisting[differing] += 1;
        } else {
            joint_cleared[differing] += 1;
        }
        if strict_flags[3] {
            unary_persisting[differing] += 1;
        } else {
            unary_cleared[differing] += 1;
        }
        for side in [&l, &r] {
            for (extra, active) in states.iter_mut().enumerate() {
                active.extend([side.gold_extra[extra], side.rival_extra[extra]]);
            }
        }
        if !strict_flags[1] && witness.is_none() {
            witness = Some(format!(
                "{{\"pair\":{pair},\"differing_banks\":{differing},\"ordinals\":{:?},\"ids_left\":{:?},\"ids_right\":{:?},\"gold_left\":{:?},\"rival_left\":{:?},\"gold_right\":{:?},\"rival_right\":{:?},\"joint_left_right\":{:?},\"unary_left_right\":{:?},\"strict_thresholds\":{:?},\"id_tie_thresholds\":{:?}}}",
                [l.ordinal, r.ordinal], l.ids, r.ids, l.gold, l.rival, r.gold, r.rival,
                [l.gold_extra[0], l.rival_extra[0], r.gold_extra[0], r.rival_extra[0]],
                [l.gold_extra[1], l.rival_extra[1], r.gold_extra[1], r.rival_extra[1]],
                [l.strict(), r.strict()], [l.tie_aware(), r.tie_aware()],
            ));
        }
    }
    require(
        ordinals.len() == 2 * PAIRS && strict[0] == PAIRS,
        "baseline totals changed",
    )?;
    println!(
        "{{\n  \"pairs_verified\": {PAIRS},\n  \"ordinals_verified\": {},\n  \"family_order\": [\"four_bank\", \"joint\", \"direct_tuple\", \"unary\"],\n  \"opposed\": {opposed:?},\n  \"strict_integer_losses\": {strict:?},\n  \"original_id_tie_integer_losses\": {tie_aware:?},\n  \"strict_integer_upper_bounds\": {:?},\n  \"direct_tuple_real_strict_losses\": {tuple_real},\n  \"direct_tuple_real_id_tie_losses\": {tuple_real_tie_aware},\n  \"within_bank_relabel_checks\": {PAIRS},\n  \"zero_fifth_checks\": {PAIRS},\n  \"pairs_by_differing_banks\": {all_pairs:?},\n  \"joint_persisting_by_differing_banks\": {joint_persisting:?},\n  \"joint_cleared_by_differing_banks\": {joint_cleared:?},\n  \"unary_persisting_by_differing_banks\": {unary_persisting:?},\n  \"unary_cleared_by_differing_banks\": {unary_cleared:?},\n  \"joint_observed_states\": {:?},\n  \"unary_observed_states\": {:?},\n  \"joint_cleared_example\": {}\n}}",
        ordinals.len(), strict.map(|losses| OPENINGS - losses),
        states[0].iter().collect::<Vec<_>>(), states[1].iter().collect::<Vec<_>>(),
        witness.as_deref().unwrap_or("null"),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn side(delta: i64, ids: [i64; 2], gold: Address, rival: Address) -> Side {
        Side {
            gold,
            rival,
            gold_extra: [1, 2],
            rival_extra: [3, 4],
            packed: [1, 2],
            ordinal: 0,
            group: 0,
            ids,
            delta,
        }
    }

    #[test]
    fn signed_thresholds_at_q48_boundaries() {
        for (delta, strict, tie) in [
            (-UNIT - 1, -1, -1),
            (-UNIT, 0, -1),
            (-1, 0, 0),
            (0, 1, 0),
            (1, 1, 1),
            (UNIT, 2, 1),
            (UNIT + 1, 2, 2),
        ] {
            let favored = side(delta, [1, 2], [0; 4], [0; 4]);
            let unfavored = side(delta, [2, 1], [0; 4], [0; 4]);
            assert_eq!(favored.strict(), strict);
            assert_eq!(favored.tie_aware(), tie);
            assert_eq!(unfavored.tie_aware(), strict);
        }
    }

    #[test]
    fn signed_banks_cancel_even_with_different_constant_addresses() {
        let left = side(0, [1, 2], [0, 7, 0, 0], [1, 7, 0, 0]);
        let right = side(0, [2, 1], [1, 8, 0, 0], [0, 8, 0, 0]);
        assert!(opposite(&left.incidence(0), &right.incidence(0)));
        assert!(opposite(&left.incidence(4), &right.incidence(4)));
        assert!(!opposite(&left.incidence(1), &right.incidence(1)));
        assert_eq!(zero_fifth(&left.incidence(1)), left.incidence(0));
    }
}
