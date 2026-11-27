#![forbid(unsafe_code)]

use crate::{fail, Result};
use serde_json::{json, Value};
use std::path::PathBuf;
use uor_r4_discrete::corpus::{BosIdentity, CorpusSource, FilePin, Split, MAX_BATCH_ROWS};
use uor_r4_discrete::training::{
    CodebookInitialization, CoordinateBlock, CounterfactualPath, DriverConfig, SchedulePolicy,
    DRIVER_SCHEMA,
};
use uor_r4_discrete::{ObjectivePolicy, ReadInitialization, ReadSpec, Spec};

#[cfg(feature = "cuda")]
use uor_r4_discrete::{device::Device, Identity, Parameter};

#[cfg(feature = "cuda")]
struct Lcg(u64);

#[cfg(feature = "cuda")]
impl Lcg {
    fn next(&mut self, bound: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % u64::from(bound)) as u32
    }
}

fn synthetic_driver_config(batch_rows: u32) -> DriverConfig {
    let pin = |name: &str, ch: char| FilePin {
        path: PathBuf::from(name),
        sha256: std::iter::repeat_n(ch, 64).collect(),
    };
    DriverConfig {
        schema: DRIVER_SCHEMA.to_owned(),
        spec: Spec {
            vocabulary: 16,
            blocks: 4,
            context: 12,
            kept_cells: 4,
        },
        read_spec: ReadSpec::Route,
        objective_policy: ObjectivePolicy::NearestAddressMarginV1,
        read_initialization: ReadInitialization::ZeroGatesV1,
        normalization_mode: Default::default(),
        position: None,
        address_metric: Default::default(),
        relation_learning_policy: Default::default(),
        counterfactual_path: CounterfactualPath::RowsV1,
        codebook_initialization: CodebookInitialization::SeededDeviceV1,
        coordinate_block: CoordinateBlock(64),
        seed: 4390,
        batch_rows,
        train: CorpusSource {
            split: Split::Train,
            tokens: pin("train.tokens.u16", 'a'),
            mask: pin("train.mask.u8", 'b'),
            expected_payload_tokens: 4096,
            expected_eligible_targets: 4096,
        },
        dev: CorpusSource {
            split: Split::Dev,
            tokens: pin("dev.tokens.u16", 'c'),
            mask: pin("dev.mask.u8", 'd'),
            expected_payload_tokens: 1024,
            expected_eligible_targets: 1024,
        },
        tokenizer: pin("tokenizer.json", 'e'),
        bos: BosIdentity {
            token: "<bos>".to_owned(),
            id: 0,
        },
        schedule: SchedulePolicy::CanonicalVectorThenReadV1,
    }
}

/// Verify the lifted `DriverConfig::validate` `1..=4,096` batch-row cap (`PR #4306`)
/// and, under `--features cuda`, execute a live `4,096`-row, `K = 64` `Device::carry_windows`
/// -> `score_rows_k` -> `propose_block` check on the RTX 4090 without modifying model state.
pub fn run_driver_4096_check() -> Result<Value> {
    synthetic_driver_config(MAX_BATCH_ROWS).validate()?;
    let zero_refused = synthetic_driver_config(0).validate().is_err();
    let over_refused = synthetic_driver_config(MAX_BATCH_ROWS + 1)
        .validate()
        .is_err();
    if !zero_refused || !over_refused {
        return Err(fail(
            "DriverConfig::validate did not refuse 0 or 4097 batch_rows",
        ));
    }

    #[cfg(feature = "cuda")]
    {
        let spec = Spec {
            vocabulary: 16,
            blocks: 4,
            context: 12,
            kept_cells: 4,
        };
        let device = Device::new(spec, 4390, 0)?;
        let rows = MAX_BATCH_ROWS as usize;
        let mut rng = Lcg(0x5055_4096_0001);
        let mut occ = 0u64;
        let mut windows = Vec::with_capacity(rows);
        let mut targets = Vec::with_capacity(rows);
        for _ in 0..rows {
            let len = 2 + rng.next(spec.context - 1);
            let mut w = Vec::with_capacity(len as usize);
            for _ in 0..len {
                occ += 1;
                w.push(Identity {
                    token: rng.next(spec.vocabulary),
                    occurrence: occ,
                    record: 1,
                    version: 1,
                });
            }
            windows.push(w);
            targets.push(rng.next(spec.vocabulary));
        }
        let carried = device.carry_windows(&windows)?;
        if carried.rows() != MAX_BATCH_ROWS {
            return Err(fail(format!(
                "carry_windows returned {} rows != {MAX_BATCH_ROWS}",
                carried.rows()
            )));
        }
        let block: Vec<Parameter> = (0..64u32).map(Parameter::VectorBlock).collect();
        let eligible = vec![true; block.len()];
        let rowsk = device.rowsk_scoring()?;
        let scores = device.score_rows_k(&carried, &targets, &block, &rowsk)?;
        if scores.len() != 64 {
            return Err(fail("score_rows_k at 4096 rows did not return 64 scores"));
        }
        let status_zero = scores.iter().filter(|s| s.status == 0).count();
        let bvgk = device.bvgk_kernels()?;
        let before_epoch = device.epoch()?;
        let proposal =
            device.propose_block(&carried, &targets, &block, &eligible, &rowsk, &bvgk)?;
        let after_epoch = device.epoch()?;
        if before_epoch != 0 || after_epoch != 0 {
            return Err(fail(
                "propose_block at 4096 rows mutated device epoch (UT violation)",
            ));
        }
        let rec = proposal.record();
        if rec.k != 64 {
            return Err(fail(format!("propose_block record.k {} != 64", rec.k)));
        }
        let rowsk_hashes = uor_r4_discrete::device::compiled_rowsk_image_sha256();
        let bvgk_hashes = uor_r4_discrete::device::compiled_bvgk_image_sha256();
        Ok(json!({
            "driver_config_4096_admitted": true,
            "driver_config_0_refused": zero_refused,
            "driver_config_4097_refused": over_refused,
            "cuda_executed": true,
            "batch_rows": MAX_BATCH_ROWS,
            "k": 64,
            "carried_rows": carried.rows(),
            "rowsk_scores_len": scores.len(),
            "rowsk_status_zero_count": status_zero,
            "propose_block_k": rec.k,
            "propose_block_s_len": rec.s_len,
            "propose_block_joint": rec.joint,
            "propose_block_l_before": rec.l_before,
            "propose_block_l_best": rec.l_best,
            "propose_block_l_joint": rec.l_joint,
            "propose_block_epoch_untouched": before_epoch == 0 && after_epoch == 0,
            "rowsk_cubin_sha256": rowsk_hashes[2],
            "bvgk_cubin_sha256": bvgk_hashes[2],
            "pass": true,
        }))
    }

    #[cfg(not(feature = "cuda"))]
    {
        Ok(json!({
            "driver_config_4096_admitted": true,
            "driver_config_0_refused": zero_refused,
            "driver_config_4097_refused": over_refused,
            "cuda_executed": false,
            "batch_rows": MAX_BATCH_ROWS,
            "k": 64,
            "pass": true,
        }))
    }
}
