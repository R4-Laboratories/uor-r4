use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, error::Error, fs, io};
use uor_r4_native_vector::{
    butterfly_rotor::{ButterflyLinear, HIER_COARSE, HIER_FINE},
    butterfly_rotor_v3::ButterflyV3Model,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const QUOTE_ID: usize = 1228;
const QUOTE_SLOT: usize = 192;
const QUOTE_CELL: usize = 3;
const PINS: [(&str, &str, u64); 3] = [
    (
        "frozen",
        "20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589",
        3333,
    ),
    (
        "reset",
        "3656e0575bd9e590e63305000e6f03ca010e1a663d6ac45d9061961f9a356bc2",
        3397,
    ),
    (
        "preserved",
        "8e0721cc8b989f12af9c80b88dfac53c3a33701e6a3e8e7d2d26d2dec7fd1340",
        3397,
    ),
];

#[derive(Clone, Copy, Default, Serialize)]
struct Pair {
    changed: usize,
    increased: usize,
    decreased: usize,
    max_abs_delta: u32,
    l1_delta: u64,
    signed_sum_delta: i64,
}

impl Pair {
    fn observe(&mut self, delta: i32) {
        let magnitude = delta.unsigned_abs();
        self.changed += usize::from(delta != 0);
        self.increased += usize::from(delta > 0);
        self.decreased += usize::from(delta < 0);
        self.max_abs_delta = self.max_abs_delta.max(magnitude);
        self.l1_delta += u64::from(magnitude);
        self.signed_sum_delta += i64::from(delta);
    }

    fn combine(&mut self, other: Self) {
        self.changed += other.changed;
        self.increased += other.increased;
        self.decreased += other.decreased;
        self.max_abs_delta = self.max_abs_delta.max(other.max_abs_delta);
        self.l1_delta += other.l1_delta;
        self.signed_sum_delta += other.signed_sum_delta;
    }
}

#[derive(Clone, Copy, Default, Serialize)]
struct Movement {
    words: usize,
    frozen_to_reset: Pair,
    frozen_to_preserved: Pair,
    reset_to_preserved: Pair,
    changed_in_both: usize,
    exact_same_nonzero_delta: usize,
    same_direction_nonzero: usize,
    opposite_direction_nonzero: usize,
}

impl Movement {
    fn compare([frozen, reset, preserved]: [&[i16]; 3]) -> Result<Self> {
        if frozen.len() != reset.len() || reset.len() != preserved.len() {
            return Err("parameter slice length mismatch".into());
        }
        let mut stats = Self {
            words: frozen.len(),
            ..Self::default()
        };
        for ((&f, &r), &p) in frozen.iter().zip(reset).zip(preserved) {
            let reset_delta = i32::from(r) - i32::from(f);
            let preserved_delta = i32::from(p) - i32::from(f);
            stats.frozen_to_reset.observe(reset_delta);
            stats.frozen_to_preserved.observe(preserved_delta);
            stats
                .reset_to_preserved
                .observe(preserved_delta - reset_delta);
            if reset_delta != 0 && preserved_delta != 0 {
                stats.changed_in_both += 1;
                stats.exact_same_nonzero_delta += usize::from(reset_delta == preserved_delta);
                if reset_delta.signum() == preserved_delta.signum() {
                    stats.same_direction_nonzero += 1;
                } else {
                    stats.opposite_direction_nonzero += 1;
                }
            }
        }
        Ok(stats)
    }

    fn combine(&mut self, other: Self) {
        self.words += other.words;
        self.frozen_to_reset.combine(other.frozen_to_reset);
        self.frozen_to_preserved.combine(other.frozen_to_preserved);
        self.reset_to_preserved.combine(other.reset_to_preserved);
        self.changed_in_both += other.changed_in_both;
        self.exact_same_nonzero_delta += other.exact_same_nonzero_delta;
        self.same_direction_nonzero += other.same_direction_nonzero;
        self.opposite_direction_nonzero += other.opposite_direction_nonzero;
    }
}

fn load(path: &str, (name, pin, step): (&str, &str, u64)) -> Result<ButterflyV3Model> {
    let bytes = fs::read(path)?;
    if hex::encode(Sha256::digest(&bytes)) != pin {
        return Err(format!("{name} SHA-256 differs from the registered artifact").into());
    }
    let model = ButterflyV3Model::from_bytes(&bytes)?;
    if model.step != step {
        return Err(format!("{name} registered step mismatch").into());
    }
    Ok(model)
}

fn add(
    families: &mut BTreeMap<&'static str, Movement>,
    name: &'static str,
    slices: [&[i16]; 3],
) -> Result<()> {
    families
        .entry(name)
        .or_default()
        .combine(Movement::compare(slices)?);
    Ok(())
}

fn linear([f, r, p]: [&ButterflyLinear; 3]) -> Result<Movement> {
    let mut result = Movement::compare([&f.b_in.shears, &r.b_in.shears, &p.b_in.shears])?;
    result.combine(Movement::compare([&f.diag_q12, &r.diag_q12, &p.diag_q12])?);
    result.combine(Movement::compare([
        &f.b_out.shears,
        &r.b_out.shears,
        &p.b_out.shears,
    ])?);
    Ok(result)
}

fn families(models: [&ButterflyV3Model; 3]) -> Result<BTreeMap<&'static str, Movement>> {
    let mut result = BTreeMap::new();
    add(
        &mut result,
        "embeddings_tied",
        models.map(|m| m.embeddings.as_slice()),
    )?;
    add(
        &mut result,
        "coarse_centroids",
        models.map(|m| m.coarse_centroids.as_slice()),
    )?;
    add(
        &mut result,
        "final_norm_q12",
        models.map(|m| m.final_norm_q12.as_slice()),
    )?;
    result.insert("head_proj", linear(models.map(|m| &m.head_proj))?);
    for layer in 0..models[0].cfg.layers {
        let layers = models.map(|m| &m.layers[layer]);
        add(
            &mut result,
            "layer.norm1_q12",
            layers.map(|l| l.norm1_q12.as_slice()),
        )?;
        add(
            &mut result,
            "layer.norm2_q12",
            layers.map(|l| l.norm2_q12.as_slice()),
        )?;
        add(
            &mut result,
            "layer.shift_mix_q15",
            layers.map(|l| l.shift_mix_q15.as_slice()),
        )?;
        add(
            &mut result,
            "layer.gate_bias_q15",
            layers.map(|l| l.gate_bias_q15.as_slice()),
        )?;
        add(
            &mut result,
            "layer.pkm.key_a",
            layers.map(|l| l.pkm.key_a.as_slice()),
        )?;
        add(
            &mut result,
            "layer.pkm.key_b",
            layers.map(|l| l.pkm.key_b.as_slice()),
        )?;
        add(
            &mut result,
            "layer.pkm.values",
            layers.map(|l| l.pkm.values.as_slice()),
        )?;
        for (name, projections) in [
            ("layer.rotor_proj", layers.map(|l| &l.rotor_proj)),
            ("layer.key_proj", layers.map(|l| &l.key_proj)),
            ("layer.value_proj", layers.map(|l| &l.value_proj)),
            ("layer.query_proj", layers.map(|l| &l.query_proj)),
        ] {
            result
                .entry(name)
                .or_default()
                .combine(linear(projections)?);
        }
        for branch in 0..models[0].cfg.ffn_branches {
            result
                .entry("layer.ffn_in")
                .or_default()
                .combine(linear(layers.map(|l| &l.ffn_in[branch]))?);
            result
                .entry("layer.ffn_out")
                .or_default()
                .combine(linear(layers.map(|l| &l.ffn_out[branch]))?);
        }
    }
    Ok(result)
}

#[derive(Serialize)]
struct Embedding {
    slot: usize,
    original_id: u16,
    movement: Movement,
}

#[derive(Serialize)]
struct Cell {
    cell: usize,
    centroid: Movement,
    embedding_rows_total: Movement,
    members: Vec<Embedding>,
}

fn cells(models: [&ButterflyV3Model; 3]) -> Result<Vec<Cell>> {
    let dim = models[0].cfg.dim;
    (0..HIER_COARSE)
        .map(|cell| {
            let coarse = cell * dim;
            let centroid =
                Movement::compare(models.map(|m| &m.coarse_centroids[coarse..coarse + dim]))?;
            let first_slot = cell * HIER_FINE;
            let members: Vec<Embedding> = (first_slot..first_slot + HIER_FINE)
                .map(|slot| {
                    let word = slot * dim;
                    Ok(Embedding {
                        slot,
                        original_id: models[0].vocab_inv[slot],
                        movement: Movement::compare(
                            models.map(|m| &m.embeddings[word..word + dim]),
                        )?,
                    })
                })
                .collect::<Result<_>>()?;
            let mut embedding_rows_total = Movement::default();
            for member in &members {
                embedding_rows_total.combine(member.movement);
            }
            Ok(Cell {
                cell,
                centroid,
                embedding_rows_total,
                members,
            })
        })
        .collect()
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [frozen, reset, preserved] = args.as_slice() else {
        return Err("usage: weight_inventory FROZEN RESET PRESERVED".into());
    };
    let frozen = load(frozen, PINS[0])?;
    let reset = load(reset, PINS[1])?;
    let preserved = load(preserved, PINS[2])?;
    let models = [&frozen, &reset, &preserved];
    if models
        .iter()
        .any(|model| model.cfg != frozen.cfg || model.vocab_perm != frozen.vocab_perm)
        || usize::from(frozen.vocab_perm[QUOTE_ID]) != QUOTE_SLOT
        || QUOTE_SLOT / HIER_FINE != QUOTE_CELL
    {
        return Err("registered configuration or original ID permutation mismatch".into());
    }

    let families = families(models)?;
    let cells = cells(models)?;
    let mut total = Movement::default();
    for movement in families.values() {
        total.combine(*movement);
    }
    if models
        .iter()
        .any(|model| model.param_count() != total.words)
    {
        return Err("parameter family inventory does not cover the serialized payload".into());
    }

    let quote = &cells[QUOTE_CELL].members[QUOTE_SLOT % HIER_FINE];
    serde_json::to_writer(
        io::stdout().lock(),
        &json!({
            "scope":"read-only static UORBFY03 signed i16 parameter comparison",
            "checkpoints":PINS.map(|(role, sha256, step)| json!({ "role":role, "sha256":sha256, "step":step })),
            "dimension":frozen.cfg.dim,
            "total":total, "parameter_families":families,
            "head_proj_components":{
                "b_in_q15":Movement::compare(models.map(|m| m.head_proj.b_in.shears.as_slice()))?,
                "diag_q12":Movement::compare(models.map(|m| m.head_proj.diag_q12.as_slice()))?,
                "b_out_q15":Movement::compare(models.map(|m| m.head_proj.b_out.shears.as_slice()))?,
            },
            "quote_embedding":quote,
            "quote_coarse_cell":QUOTE_CELL,
            "cells":cells,
        }),
    )?;
    println!();
    Ok(())
}
