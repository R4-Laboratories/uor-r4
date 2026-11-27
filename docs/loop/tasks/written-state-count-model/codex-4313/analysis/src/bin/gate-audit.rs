//! Independent integer recount of saved observations. No model forward call.
//! Stream the raw count triples once per checkpoint, accumulating every token
//! at the observed keys, then sort all original IDs to compare ranks.
#![forbid(unsafe_code)]

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SPACE: usize = 223;
const NEWLINE: usize = 201;
const LOWERCASE_G: usize = 73;
const TABLES: [&str; 6] = [
    "query_table",
    "query_read_table",
    "read_table",
    "query_rec_table",
    "read_rec_table",
    "rec_table",
];
const WEIGHTS: [u64; 6] = [1, 64, 2, 32, 8, 1];

#[derive(Deserialize, PartialEq)]
struct Input {
    pair_id: String,
    side: usize,
    cohort: String,
    target: usize,
    prefix: Vec<usize>,
}

#[derive(Deserialize)]
struct Frozen {
    #[serde(flatten)]
    input: Input,
    group_id: String,
}

#[derive(Deserialize)]
struct Row {
    #[serde(flatten)]
    input: Input,
    rival: usize,
    space: usize,
    artifact_sha256: String,
    query_token: usize,
    read_code: u32,
    recurrence_code: u32,
    source_position: Option<usize>,
    source_token: Option<usize>,
    gate_selected: usize,
    gate_mass: u64,
    active_candidates: usize,
    positive_candidates: usize,
    prior_selected: usize,
    emitted: usize,
    writer_vector: [i32; 32],
    admitted_cells: Vec<u32>,
    writer_candidates: usize,
    top8: Vec<usize>,
    duplicate_replay: bool,
    read_vector: [i32; 32],
    recurrent_vector: [i32; 32],
    recurrent_sum: [i32; 32],
    recurrent_frames: [i32; 4],
    routes: Vec<Value>,
    probes: Probes,
}

impl Row {
    fn keys(&self) -> Result<[u32; 6]> {
        let q = u32::from(u16::try_from(self.query_token)?);
        let r = u32::from(u16::try_from(self.read_code)?);
        let c = u32::from(u16::try_from(self.recurrence_code)?);
        Ok([q, (q << 16) | r, r, (q << 16) | c, (r << 16) | c, c])
    }
}

#[derive(Deserialize)]
struct Probes {
    gold: Probe,
    rival: Probe,
    space: Probe,
}

#[derive(Deserialize)]
struct Probe {
    id: usize,
    mass: u64,
    gate_score_terms: Terms,
    pool_member: bool,
    positive_mass: bool,
    eligible_positive: bool,
    gate_rank: Option<usize>,
    gate_full_rank: usize,
    mass_ties: usize,
    eligible_mass_ties: usize,
    prior_count: u32,
    prior_rank: usize,
    prior_ties: usize,
    writer_cell: u32,
    writer_distance: u32,
    writer_admitted: bool,
    writer_rank: Option<usize>,
    writer_full_rank: usize,
    writer_distance_ties: usize,
    writer_admitted_ties: usize,
}

#[derive(Deserialize)]
struct Terms {
    c_q: u32,
    c_qr: u32,
    c_r: u32,
    c_qc: u32,
    c_rr: u32,
    pointer_bonus: u32,
}

impl Terms {
    fn words(&self) -> [u32; 6] {
        [
            self.c_q,
            self.c_qr,
            self.c_r,
            self.c_qc,
            self.c_rr,
            self.pointer_bonus,
        ]
    }
}

#[derive(Deserialize)]
struct Header {
    schema: String,
    shape: Shape,
    tables: Vec<Section>,
}

#[derive(Deserialize)]
struct Shape {
    vocabulary: usize,
    blocks: usize,
}

#[derive(Deserialize)]
struct Section {
    name: String,
    kind: String,
    len: u64,
    span: Span,
}

#[derive(Deserialize)]
struct Span {
    offset: u64,
    bytes: u64,
}

fn require(condition: bool, label: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(label.to_owned().into())
    }
}

fn decode(bytes: &[u8]) -> Result<u32> {
    Ok(u32::from_le_bytes(bytes.try_into()?))
}

type SparseSlice = HashMap<u32, Vec<(usize, u32)>>;

struct Artifact {
    counts: Vec<u32>,
    fallback: usize,
    prior: Vec<usize>,
    cells: Vec<u32>,
    vectors: Vec<i32>,
    slices: [SparseSlice; 6],
    entries: BTreeMap<String, u64>,
}

impl Artifact {
    fn open(path: &str, keys: &[[u32; 6]]) -> Result<Self> {
        let mut file = BufReader::with_capacity(256 * 1024, File::open(path)?);
        let file_len = file.get_ref().metadata()?.len();
        let mut lead = [0u8; 16];
        file.read_exact(&mut lead)?;
        let json_len = u64::from_le_bytes(lead[8..16].try_into()?);
        require(
            &lead[..8] == b"UORLUT01" && json_len <= 64 << 20,
            "D11 magic/length",
        )?;
        let mut header = vec![0; usize::try_from(json_len)?];
        file.read_exact(&mut header)?;
        let data_start = (16 + json_len).div_ceil(64) * 64;
        let header: Header = serde_json::from_slice(&header)?;
        let vocab = header.shape.vocabulary;
        require(
            header.schema == "uor-r4.lut-discrete-trunk/1"
                && vocab == 4096
                && header.shape.blocks == 4
                && header.tables.len() == 19,
            "D11 layout",
        )?;
        let section = |name: &str, kind: &str| -> Result<&Section> {
            let mut matches = header.tables.iter().filter(|part| part.name == name);
            let part = matches.next().ok_or_else(|| format!("missing {name}"))?;
            require(
                matches.next().is_none()
                    && part.kind == kind
                    && part.span.offset % 64 == 0
                    && part.len.checked_mul(4) == Some(part.span.bytes)
                    && data_start
                        .saturating_add(part.span.offset)
                        .saturating_add(part.span.bytes)
                        <= file_len,
                name,
            )?;
            Ok(part)
        };
        let mut words = |name, kind| -> Result<Vec<u32>> {
            let part = section(name, kind)?;
            require(part.len <= (vocab * 32) as u64, name)?;
            file.seek(SeekFrom::Start(data_start + part.span.offset))?;
            let mut data = vec![0; usize::try_from(part.span.bytes)?];
            file.read_exact(&mut data)?;
            data.chunks_exact(4).map(decode).collect()
        };
        let mut counts = words("discrete.trunk.unigram", "u32")?;
        require(counts.len() == vocab + 1, "unigram shape")?;
        let fallback = counts.remove(0) as usize;
        let mut prior: Vec<_> = (0..vocab).collect();
        prior.sort_unstable_by_key(|&id| (Reverse(counts[id]), id));
        require(
            fallback == prior[0],
            "serialized fallback and sorted prior differ",
        )?;
        let cells = words("discrete.cells", "u32")?;
        let vectors = words("discrete.table", "i32")?
            .into_iter()
            .map(|v| v as i32)
            .collect::<Vec<_>>();
        require(
            cells.len() == vocab
                && cells.iter().all(|&cell| cell < 256)
                && vectors.len() == vocab * 32,
            "writer layout",
        )?;

        let mut slices: [SparseSlice; 6] =
            std::array::from_fn(|i| keys.iter().map(|key| (key[i], Vec::new())).collect());
        let mut entries = BTreeMap::new();
        for ((name, keep), index) in TABLES.into_iter().zip(&mut slices).zip(0..) {
            let name = format!("discrete.trunk.{name}");
            let part = section(&name, "u32")?;
            require(part.len % 3 == 0, &name)?;
            file.seek(SeekFrom::Start(data_start + part.span.offset))?;
            let mut previous = None;
            let mut byte = [0; 12];
            for _ in 0..part.len / 3 {
                file.read_exact(&mut byte)?;
                let address = decode(&byte[..4])?;
                let id = decode(&byte[4..8])? as usize;
                let count = decode(&byte[8..])?;
                require(
                    id < vocab && count > 0 && previous.is_none_or(|last| last < (address, id)),
                    &name,
                )?;
                previous = Some((address, id));
                if let Some(found) = keep.get_mut(&address) {
                    found.push((id, count));
                }
            }
            entries.insert(TABLES[index].into(), part.len / 3);
        }
        Ok(Self {
            counts,
            fallback,
            prior,
            cells,
            vectors,
            slices,
            entries,
        })
    }

    fn codebook_entry(&self, id: usize) -> Value {
        let vector = &self.vectors[id * 32..(id + 1) * 32];
        let aliases: Vec<_> = self
            .vectors
            .chunks_exact(32)
            .enumerate()
            .filter(|(_, candidate)| *candidate == vector)
            .map(|(id, _)| id)
            .collect();
        json!({"id":id, "cell":self.cells[id], "vector":vector, "exact_vector_ids":aliases,
            "exact_vector_cells":aliases.iter().map(|&id| self.cells[id]).collect::<Vec<_>>()})
    }

    fn audit(&self, row: &Row, key: [u32; 6], targets: [usize; 3]) -> Result<Audited> {
        let n = self.counts.len();
        let mut raw = vec![[0; 6]; n];
        let mut pool = vec![false; n];
        for (i, slice) in self.slices.iter().enumerate() {
            for &(id, count) in &slice[&key[i]] {
                if matches!(i, 0 | 2 | 5) {
                    pool[id] = true;
                }
                if i != 5 {
                    raw[id][i] = count;
                }
            }
        }
        if let Some(id) = row.source_token {
            pool[id] = true;
            raw[id][5] = 256;
        }
        let mass: Vec<u64> = raw
            .iter()
            .map(|terms| {
                terms
                    .iter()
                    .zip(WEIGHTS)
                    .map(|(&c, w)| u64::from(c) * w)
                    .sum()
            })
            .collect();
        let effective: Vec<_> = (0..n)
            .map(|id| if pool[id] { mass[id] } else { 0 })
            .collect();
        let mut gate = self.prior.clone();
        gate.sort_by_key(|&id| Reverse(effective[id]));
        let positive = effective.iter().filter(|&&value| value > 0).count();
        let winner = gate[0];
        require(
            row.gate_selected == winner
                && row.gate_mass == effective[winner]
                && row.prior_selected == self.fallback
                && row.positive_candidates == positive
                && row.active_candidates == pool.iter().filter(|&&value| value).count(),
            "gate winner, prior or pool size",
        )?;
        let mut admitted = [false; 256];
        for &cell in &row.admitted_cells {
            *admitted
                .get_mut(cell as usize)
                .ok_or("invalid admitted cell")? = true;
        }
        let distance: Vec<u32> = self
            .vectors
            .chunks_exact(32)
            .map(|vector| {
                vector
                    .iter()
                    .zip(row.writer_vector)
                    .map(|(&a, b)| a.abs_diff(b))
                    .sum()
            })
            .collect();
        let admitted_token: Vec<_> = self
            .cells
            .iter()
            .map(|&cell| admitted[cell as usize])
            .collect();
        let mut writer: Vec<_> = (0..n).collect();
        writer.sort_unstable_by_key(|&id| (!admitted_token[id], distance[id], id));
        let writer_candidates = admitted_token.iter().filter(|&&value| value).count();
        require(
            row.writer_candidates == writer_candidates
                && row.emitted == writer[0]
                && row.top8 == writer[..writer_candidates.min(8)],
            "writer top eight, winner or candidate count",
        )?;
        let mut probes = [Endpoint::default(); 3];
        for ((endpoint, claim), id) in probes
            .iter_mut()
            .zip([&row.probes.gold, &row.probes.rival, &row.probes.space])
            .zip(targets)
        {
            let prior_rank = self
                .prior
                .iter()
                .position(|&other| other == id)
                .ok_or("prior id")?
                + 1;
            let gate_rank = gate
                .iter()
                .position(|&other| other == id)
                .ok_or("gate id")?
                + 1;
            let writer_rank = writer
                .iter()
                .position(|&other| other == id)
                .ok_or("writer id")?
                + 1;
            let native_gate_rank =
                (effective[id] > 0 || (positive == 0 && id == self.fallback)).then_some(gate_rank);
            let native_writer_rank = admitted_token[id].then_some(writer_rank);
            require(
                claim.id == id
                    && claim.mass == mass[id]
                    && claim.gate_score_terms.words() == raw[id]
                    && claim.pool_member == pool[id]
                    && claim.positive_mass == (mass[id] > 0)
                    && claim.eligible_positive == (effective[id] > 0)
                    && claim.gate_rank == native_gate_rank
                    && claim.gate_full_rank == gate_rank
                    && claim.mass_ties == mass.iter().filter(|&&m| m == mass[id]).count()
                    && claim.eligible_mass_ties
                        == (0..n)
                            .filter(|&j| effective[j] > 0 && mass[j] == mass[id])
                            .count()
                    && claim.prior_count == self.counts[id]
                    && claim.prior_rank == prior_rank
                    && claim.prior_ties
                        == self
                            .counts
                            .iter()
                            .filter(|&&c| c == self.counts[id])
                            .count(),
                &format!("token {id} gate mass/terms/ranks/ties"),
            )?;
            require(
                claim.writer_cell == self.cells[id]
                    && claim.writer_distance == distance[id]
                    && claim.writer_admitted == admitted_token[id]
                    && claim.writer_rank == native_writer_rank
                    && claim.writer_full_rank == writer_rank
                    && claim.writer_distance_ties
                        == distance.iter().filter(|&&d| d == distance[id]).count()
                    && claim.writer_admitted_ties
                        == (0..n)
                            .filter(|&j| admitted_token[j] && distance[j] == distance[id])
                            .count(),
                &format!("token {id} writer admission/distance/ranks/ties"),
            )?;
            *endpoint = Endpoint {
                mass: effective[id],
                raw_mass: mass[id],
                in_pool: pool[id],
                prior_rank,
                gate_rank,
                native_gate_rank,
                native_writer_rank,
                terms: raw[id],
            };
        }
        let weighted_context = |id: usize| effective[id].saturating_sub(u64::from(raw[id][0]));
        let context_winner = *self
            .prior
            .iter()
            .rev()
            .max_by_key(|&&id| weighted_context(id))
            .ok_or("context winner")?;
        let top_query_other = raw
            .iter()
            .enumerate()
            .filter(|(id, _)| *id != SPACE)
            .map(|(_, terms)| terms[0])
            .max()
            .ok_or("query competitor")?;
        Ok(Audited {
            probes,
            winner,
            gate_margin: effective[winner] - effective[gate[1]],
            query_space_margin: i64::from(raw[SPACE][0]) - i64::from(top_query_other),
            context_winner,
            context_max: weighted_context(context_winner),
            space_context_mass: weighted_context(SPACE),
            context_gold_mass: weighted_context(targets[0]),
            newline_distance: distance[NEWLINE],
            g_distance: distance[LOWERCASE_G],
            newline_admitted: admitted_token[NEWLINE],
            g_admitted: admitted_token[LOWERCASE_G],
            positive,
            max_excluded_raw_mass: mass
                .iter()
                .zip(&pool)
                .filter(|(_, member)| !**member)
                .map(|(&value, _)| value)
                .max()
                .unwrap_or(0),
        })
    }
}

#[derive(Default, Clone, Copy)]
struct Endpoint {
    mass: u64,
    raw_mass: u64,
    in_pool: bool,
    prior_rank: usize,
    gate_rank: usize,
    native_gate_rank: Option<usize>,
    native_writer_rank: Option<usize>,
    terms: [u32; 6],
}

struct Audited {
    probes: [Endpoint; 3],
    winner: usize,
    gate_margin: u64,
    query_space_margin: i64,
    context_winner: usize,
    context_max: u64,
    space_context_mass: u64,
    context_gold_mass: u64,
    newline_distance: u32,
    g_distance: u32,
    newline_admitted: bool,
    g_admitted: bool,
    positive: usize,
    max_excluded_raw_mass: u64,
}

#[derive(Default, Serialize)]
struct Summary {
    counts: BTreeMap<&'static str, u64>,
    ranks: BTreeMap<&'static str, Vec<usize>>,
    weighted_terms: BTreeMap<&'static str, [u64; 6]>,
    range: BTreeMap<&'static str, [i64; 2]>,
    context_winners: BTreeMap<usize, usize>,
}

impl Summary {
    fn count(&mut self, key: &'static str, value: bool) {
        *self.counts.entry(key).or_default() += u64::from(value);
    }

    fn rank(&mut self, key: &'static str, rank: Option<usize>) {
        if let Some(rank) = rank {
            self.ranks.entry(key).or_default().push(rank);
        }
    }

    fn range(&mut self, key: &'static str, value: i64) {
        let [lo, hi] = self.range.entry(key).or_insert([value, value]);
        *lo = (*lo).min(value);
        *hi = (*hi).max(value);
    }

    fn row(&mut self, row: &Row, a: &Audited, other: &Audited) {
        let [gold, rival, space] = a.probes;
        self.count("rows", true);
        for (label, endpoint) in ["gold", "rival", "space"].into_iter().zip(a.probes) {
            for (total, (count, weight)) in self
                .weighted_terms
                .entry(label)
                .or_default()
                .iter_mut()
                .zip(endpoint.terms.into_iter().zip(WEIGHTS))
            {
                if endpoint.in_pool {
                    *total += u64::from(count) * weight;
                }
            }
        }
        let crossed = other.probes[1];
        for (name, condition) in [
            ("pool_gold", gold.in_pool),
            ("positive_gold", gold.mass > 0),
            ("positive_rival", rival.mass > 0),
            ("positive_space", space.mass > 0),
            ("raw_zero_gold", gold.raw_mass == 0),
            ("raw_zero_rival", rival.raw_mass == 0),
            ("zero_positive_pool", a.positive == 0),
            ("excluded_raw_positive", a.max_excluded_raw_mass > 0),
            ("gold_pool_zero", gold.in_pool && gold.mass == 0),
            ("gold_above_space_mass", gold.mass > space.mass),
            ("gold_ties_space_mass", gold.mass == space.mass),
            ("gold_above_space_rank", gold.gate_rank < space.gate_rank),
            ("gold_improves_on_prior", gold.gate_rank < gold.prior_rank),
            ("gold_ties_prior", gold.gate_rank == gold.prior_rank),
            ("gold_declines_from_prior", gold.gate_rank > gold.prior_rank),
            (
                "gold_no_support_either_question",
                gold.mass == 0 && crossed.mass == 0,
            ),
            ("gold_own_mass_higher", gold.mass > crossed.mass),
            ("gold_own_mass_lower", gold.mass < crossed.mass),
            ("gold_own_mass_equal", gold.mass == crossed.mass),
            ("gold_own_rank_higher", gold.gate_rank < crossed.gate_rank),
            ("gold_own_rank_lower", gold.gate_rank > crossed.gate_rank),
            ("gold_own_rank_equal", gold.gate_rank == crossed.gate_rank),
            (
                "gold_rank_changes_at_equal_mass",
                gold.mass == crossed.mass && gold.gate_rank != crossed.gate_rank,
            ),
            ("gate_space", a.winner == SPACE),
            ("gate_gold", a.winner == row.input.target),
            ("emitted_space", row.emitted == SPACE),
            ("emitted_gold", row.emitted == row.input.target),
            ("writer_keeps_gate", row.emitted == a.winner),
            ("gold_in_writer", gold.native_writer_rank.is_some()),
            (
                "gold_writer_top8",
                gold.native_writer_rank.is_some_and(|rank| rank <= 8),
            ),
            (
                "gold_gate_top8",
                gold.native_gate_rank.is_some_and(|rank| rank <= 8),
            ),
            ("query_alone_unique_space", a.query_space_margin > 0),
            (
                "query_margin_larger_than_all_context_mass",
                a.query_space_margin > a.context_max as i64,
            ),
            ("context_only_space", a.context_winner == SPACE),
            ("context_only_gold", a.context_winner == row.input.target),
            (
                "context_only_gold_mass_above_space",
                a.context_gold_mass > a.space_context_mass,
            ),
            (
                "context_only_gold_mass_ties_space",
                a.context_gold_mass == a.space_context_mass,
            ),
            ("gate_newline", a.winner == NEWLINE),
            (
                "gate_newline_emits_lowercase_g",
                a.winner == NEWLINE && row.emitted == LOWERCASE_G,
            ),
            (
                "gate_newline_emits_newline",
                a.winner == NEWLINE && row.emitted == NEWLINE,
            ),
            (
                "gate_newline_has_both_admitted_at_zero_distance",
                a.winner == NEWLINE
                    && a.newline_admitted
                    && a.g_admitted
                    && a.newline_distance == 0
                    && a.g_distance == 0,
            ),
        ] {
            self.count(name, condition);
        }
        *self.context_winners.entry(a.context_winner).or_default() += 1;
        for (key, value) in [
            ("gate_winning_margin", a.gate_margin as i64),
            ("space_query_only_margin", a.query_space_margin),
            ("context_only_max_mass", a.context_max as i64),
            ("space_context_mass", a.space_context_mass as i64),
            (
                "space_minus_gold_mass",
                space.mass as i64 - gold.mass as i64,
            ),
            ("gold_mass", gold.mass as i64),
        ] {
            self.range(key, value);
        }
        for (key, rank) in [
            ("gold_prior", Some(gold.prior_rank)),
            ("gold_gate_full", Some(gold.gate_rank)),
            ("gold_gate_native", gold.native_gate_rank),
            ("gold_writer_native", gold.native_writer_rank),
            ("space_prior", Some(space.prior_rank)),
            ("space_gate_full", Some(space.gate_rank)),
            ("space_gate_native", space.native_gate_rank),
            ("space_writer_native", space.native_writer_rank),
        ] {
            self.rank(key, rank);
        }
    }

    fn pair(&mut self, [x, y]: [&Row; 2], [a, b]: [&Audited; 2]) -> [bool; 3] {
        let read = x.read_code != y.read_code;
        let recurrence = x.recurrence_code != y.recurrence_code;
        let left = a.probes[0].mass.cmp(&a.probes[1].mass);
        let right = b.probes[0].mass.cmp(&b.probes[1].mass);
        let strict = left.is_gt() && right.is_gt();
        let rank_flip = [a, b]
            .iter()
            .all(|p| p.probes[0].gate_rank < p.probes[1].gate_rank);
        let both_emit_gold = [x, y].iter().all(|r| r.emitted == r.input.target);
        for (name, condition) in [
            ("pairs", true),
            ("read_code_changes", read),
            ("recurrent_code_changes", recurrence),
            ("both_codes_change", read && recurrence),
            ("read_code_only", read && !recurrence),
            ("recurrent_code_only", !read && recurrence),
            ("neither_code_changes", !read && !recurrence),
            ("read_vector_changes", x.read_vector != y.read_vector),
            (
                "recurrent_vector_changes",
                x.recurrent_vector != y.recurrent_vector,
            ),
            ("recurrent_sum_changes", x.recurrent_sum != y.recurrent_sum),
            (
                "recurrent_frames_change",
                x.recurrent_frames != y.recurrent_frames,
            ),
            ("routes_change", x.routes != y.routes),
            ("writer_vector_changes", x.writer_vector != y.writer_vector),
            (
                "admitted_cells_change",
                x.admitted_cells != y.admitted_cells,
            ),
            (
                "source_position_changes",
                x.source_position != y.source_position,
            ),
            ("source_token_changes", x.source_token != y.source_token),
            (
                "gate_input_changes",
                read || recurrence
                    || x.source_token != y.source_token
                    || x.query_token != y.query_token,
            ),
            ("gate_winner_changes", a.winner != b.winner),
            ("emitted_id_changes", x.emitted != y.emitted),
            ("strict_mass_flip", strict),
            ("strict_reverse_mass_flip", left.is_lt() && right.is_lt()),
            ("both_tied", left.is_eq() && right.is_eq()),
            ("at_least_one_tied", left.is_eq() || right.is_eq()),
            ("rank_flip", rank_flip),
            (
                "both_gate_correct",
                a.winner == x.input.target && b.winner == y.input.target,
            ),
            ("both_emissions_correct", both_emit_gold),
        ] {
            self.count(name, condition);
        }
        self.row(x, a, b);
        self.row(y, b, a);
        [strict, rank_flip, both_emit_gold]
    }

    fn finish(self) -> Value {
        let ranks: BTreeMap<_, _> = self
            .ranks
            .into_iter()
            .map(|(key, mut values)| {
                values.sort_unstable();
                let n = values.len();
                (
                    key,
                    json!({"n": n, "min":values[0], "q1":values[(n-1)/4],
                "median":values[(n-1)/2], "q3":values[3*(n-1)/4], "max":values[n-1]}),
                )
            })
            .collect();
        json!({"counts":self.counts, "rank_quantiles_lower_order":ranks, "weighted_terms":self.weighted_terms,
            "ranges":self.range, "context_only_winner_ids":self.context_winners})
    }
}

fn read_rows<T: DeserializeOwned>(path: &str) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(&line?).map_err(|err| format!("{path}:{} {err}", i + 1).into())
        })
        .collect()
}

fn run(args: &[String]) -> Result<Value> {
    let [artifact, digest, panel, trace] = args else {
        return Err(
            "usage: gate-audit <D11> <verified D11 sha256> <frozen JSONL> <device JSONL>".into(),
        );
    };
    let frozen: Vec<Frozen> = read_rows(panel)?;
    let trace: Vec<Row> = read_rows(trace)?;
    require(
        trace.len() == 810 && frozen.len() == trace.len(),
        "panel size",
    )?;
    let mut observations = BTreeMap::new();
    for (index, row) in trace.iter().enumerate() {
        require(
            observations
                .insert((&row.input.pair_id, row.input.side), index)
                .is_none(),
            "duplicate observation",
        )?;
    }
    let keys: Vec<_> = trace.iter().map(Row::keys).collect::<Result<_>>()?;
    let model = Artifact::open(artifact, &keys)?;
    let n = model.counts.len();
    let mut frozen_keys = BTreeSet::new();
    let mut pairs = BTreeMap::new();
    for f in &frozen {
        let input = &f.input;
        require(
            input.side < 2 && frozen_keys.insert((&input.pair_id, input.side)),
            "duplicate or invalid panel side",
        )?;
        let index = *observations
            .get(&(&input.pair_id, input.side))
            .ok_or("missing observation")?;
        let row = &trace[index];
        require(
            &row.input == input
                && row.artifact_sha256 == *digest
                && row.duplicate_replay
                && input.prefix.last() == Some(&row.query_token)
                && row.space == SPACE
                && input.prefix.iter().all(|&id| id < n)
                && input.target < n
                && input.target != SPACE
                && row.source_position.map(|i| input.prefix.get(i).copied())
                    == row.source_token.map(Some),
            "frozen ID or observed gate input mismatch",
        )?;
        pairs.entry(&input.pair_id).or_insert([None; 2])[input.side] = Some((f, index));
    }
    let mut summary = BTreeMap::<String, Summary>::new();
    let mut groups = BTreeMap::<_, Vec<[bool; 3]>>::new();
    for (id, [left, right]) in pairs {
        let (f, i) = left.ok_or("missing panel side 0")?;
        let (g, j) = right.ok_or("missing panel side 1")?;
        let [x, y] = [&trace[i], &trace[j]];
        require(
            f.input.cohort == g.input.cohort
                && f.group_id == g.group_id
                && x.rival == y.input.target
                && y.rival == x.input.target
                && x.rival != y.rival,
            "invalid frozen target cross",
        )?;
        let a = model
            .audit(x, keys[i], [f.input.target, g.input.target, SPACE])
            .map_err(|err| format!("{id}/0: {err}"))?;
        let b = model
            .audit(y, keys[j], [g.input.target, f.input.target, SPACE])
            .map_err(|err| format!("{id}/1: {err}"))?;
        let cohort = &f.input.cohort;
        let result = summary
            .entry(cohort.clone())
            .or_default()
            .pair([x, y], [&a, &b]);
        groups
            .entry((cohort, &f.group_id))
            .or_default()
            .push(result);
        if cohort != "controlled" {
            let length = if f.input.prefix.len() == g.input.prefix.len() {
                "equal_length"
            } else {
                "unequal_length"
            };
            summary
                .entry(format!("{cohort}_{length}"))
                .or_default()
                .pair([x, y], [&a, &b]);
        }
    }
    let mut group_summary = BTreeMap::<_, [usize; 4]>::new();
    for ((cohort, _), signals) in groups {
        let [count, strict, rank, emit] = group_summary.entry(cohort).or_default();
        *count += 1;
        *strict += usize::from(signals.iter().all(|signal| signal[0]));
        *rank += usize::from(signals.iter().all(|signal| signal[1]));
        *emit += usize::from(signals.iter().all(|signal| signal[2]));
    }
    let summary: BTreeMap<_, _> = summary
        .into_iter()
        .map(|(key, stats)| (key, stats.finish()))
        .collect();
    Ok(json!({
        "verdict":"PASS", "artifact_sha256":digest, "rows":trace.len(),
        "probes_checked":trace.len()*3, "original_vocabulary":n,
        "candidate_scores_reconstructed":trace.len()*n,
        "sparse_entries_read":model.entries,
        "codebook_entries":[model.codebook_entry(LOWERCASE_G), model.codebook_entry(NEWLINE)],
        "method":"Scan raw table triples and stable-sort original IDs by eligible mass over the unigram/ID order. Recompute writer distances with saved post-gate coordinates and admitted cells.",
        "counterfactual":"Context-only excludes query marginal c_q and holds the observed candidate pool and context inputs fixed. It is a score decomposition.",
        "ordered_terms":["c_q","64*c_qr","2*c_r","32*c_qc","8*c_rr","pointer_bonus"],
        "ordered_group_counts":["groups","all_pairs_strict_mass_flip","all_pairs_rank_flip","all_first_emissions_correct"],
        "group_counts":group_summary, "strata":summary
    }))
}

fn main() -> std::process::ExitCode {
    match run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(report) => {
            println!("{report:#}");
            std::process::ExitCode::SUCCESS
        }
        Err(err) => {
            println!("{:#}", json!({"verdict":"FAIL","error":err.to_string()}));
            std::process::ExitCode::FAILURE
        }
    }
}
