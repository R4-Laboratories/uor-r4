use crate::{
    engine::P5Gpu,
    fail, measure, replay,
    rows::{NativeRow, Prepared, RowsGpu, Sample},
    sample::{self, Corpus, Position},
    Result,
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, ffi::OsString, fs, io::Write, path::PathBuf, time::Instant};
use uor_r4_discrete::p5_brier_core::{
    extract_s1_from_uorlut01, p5_random_block_coords, p5_schedule_block, P5S1State,
};

const ROWS: usize = 4096;
const BATCHES: usize = 16;
const DEV: usize = 512;
const K: usize = 64;
const P4_SHA: &str = "e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776";
const TRUNK_SHA: &str = "04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d";

fn samples(pos: &[Position]) -> Vec<Sample> {
    pos.iter()
        .map(|p| Sample {
            record: p.record,
            target_at: p.target_at,
        })
        .collect()
}

fn public_window(corpus: &Corpus, p: Position) -> Vec<u16> {
    let prefix = corpus.prefix(p);
    prefix[prefix.len().saturating_sub(2048)..].to_vec()
}

fn depth(pos: &[Position]) -> Value {
    let mut d: Vec<_> = pos.iter().map(|p| p.target_at).collect();
    d.sort_unstable();
    json!({"min":d[0],"median":d[d.len()/2],"max":d[d.len()-1],"slid":d.iter().filter(|&&n|n>2048).count()})
}

fn check_dev(
    gpu: &P5Gpu,
    reader: &RowsGpu,
    state: &P5S1State,
    corpus: &Corpus,
    positions: &[Sample],
    prepared: &Prepared,
) -> Result<(Vec<NativeRow>, Value)> {
    let start = Instant::now();
    let eval = gpu.evaluate(state, &prepared.batch)?;
    let table_us = start.elapsed().as_micros();
    let codes = state.derive_pick_codes_4()?;
    let t = Instant::now();
    let native = reader.evaluate(&corpus.ids, positions, Some(&codes))?;
    let native_us = t.elapsed().as_micros();
    if native.len() != positions.len()
        || eval.rows.len() != native.len()
        || prepared.oracle_by_row_q32.len() != native.len()
    {
        return Err(fail("development table/ordinary cardinalities differ"));
    }
    for (i, (lookup, row)) in eval.rows.iter().zip(&native).enumerate() {
        if lookup.loss_q32 != row.brier_q32
            || lookup.position != row.pick_pos.unwrap_or(0) as u32
            || lookup.distance != row.pick_distance.unwrap_or(u32::MAX)
            || lookup.successor != row.pick_successor.unwrap_or(0)
            || prepared.oracle_by_row_q32[i] > row.brier_q32
        {
            return Err(fail(format!("ordinary P4 and P5 lookup or ceiling disagree on dev row {i}: {lookup:?}, {row:?}, oracle {}",prepared.oracle_by_row_q32[i])));
        }
    }
    if eval.loss_q32 != native.iter().map(|r| r.brier_q32).sum::<u64>() {
        return Err(fail("device Brier total differs"));
    }
    Ok((
        native,
        json!({"table_wall_us":table_us,"table_device_us":eval.device_micros,"ordinary_wall_us":native_us}),
    ))
}

fn progress(log: &mut fs::File, v: &Value) -> Result<()> {
    writeln!(log, "{v}")?;
    log.flush()?;
    Ok(())
}

fn named_state<'a>(name: &str, names: &[&str], states: &'a [P5S1State]) -> Result<&'a P5S1State> {
    names
        .iter()
        .position(|&x| x == name)
        .and_then(|i| states.get(i))
        .ok_or_else(|| fail("unknown pilot arm"))
}

pub fn run(args: &[OsString]) -> Result<()> {
    let paths: Vec<_> = args.iter().map(PathBuf::from).collect();
    let (p4, trunk, train_file, dev_file, chat, out) = (
        &paths[0], &paths[1], &paths[2], &paths[3], &paths[4], &paths[5],
    );
    fs::create_dir(out)?;
    fs::create_dir(out.join("states"))?;
    let total = Instant::now();
    let mut log = fs::File::create_new(out.join("progress.jsonl"))?;
    let input_start = Instant::now();
    measure::verify_file(trunk, TRUNK_SHA)?;
    let base = extract_s1_from_uorlut01(&fs::read(trunk)?)?;
    if base.roots.len() != 16384 {
        return Err(fail("unexpected S1 geometry"));
    }
    let train = Corpus::load(train_file, sample::TRAIN_SHA)?;
    let dev = Corpus::load(dev_file, sample::DEV_SHA)?;
    if (
        train.record_count(),
        train.target_count(),
        dev.record_count(),
        dev.target_count(),
    ) != (4060, 15947584, 2670, 10807417)
    {
        return Err(fail("wrong R or DEV-tune record census"));
    }
    let schedule = train.train_sample(BATCHES, ROWS, 4308)?;
    let dev_pos = dev.dev_sample(DEV, 4309)?;
    let dev_samples = samples(&dev_pos);
    let input_us = input_start.elapsed().as_micros();
    let t = Instant::now();
    let reader = RowsGpu::new(p4)?;
    if reader.artifact_sha256 != P4_SHA {
        return Err(fail("loaded P4 artifact differs"));
    }
    let rows_load_us = t.elapsed().as_micros();
    let gpu = P5Gpu::new(0)?;
    gpu.check_finite_parity()?;
    progress(
        &mut log,
        &json!({"phase":"finite_passed","input_us":input_us,"rows_load_us":rows_load_us}),
    )?;
    let parity_t = Instant::now();
    // Contiguous, predetermined groups also exercise different successors when
    // a small fixed group happens to give identical losses for two outcomes.
    let mut frozen_spots = Vec::new();
    let mut saw_class = false;
    let mut saw_position = false;
    for group in dev_samples[..32].chunks_exact(4) {
        let checks = reader.check_parity(p4, &dev.ids, group, None)?;
        saw_class |= checks.class_sensitive_rows > 0;
        saw_position |= checks.position_sensitive_rows > 0;
        frozen_spots.push(checks);
        if saw_class && saw_position {
            break;
        }
    }
    if !(saw_class && saw_position) {
        return Err(fail("counterfactual parity could not distinguish both class and position in the preregistered 32 DEV spots"));
    }
    let frozen_spot_us = parity_t.elapsed().as_micros();
    let t = Instant::now();
    let dev_rows = reader.prepare(&dev.ids, &dev_samples)?;
    let dev_prep_us = t.elapsed().as_micros();
    let reseed = P5S1State::reseeded(base.vocab, base.blocks, 4310, base.parent_trunk_sha256)?;
    let (initial, initial_time) = check_dev(&gpu, &reader, &base, &dev, &dev_samples, &dev_rows)?;
    let (random, random_time) = check_dev(&gpu, &reader, &reseed, &dev, &dev_samples, &dev_rows)?;
    let initial_loss = initial.iter().map(|r| r.brier_q32).sum::<u64>();
    if dev_rows
        .native
        .iter()
        .zip(&initial)
        .any(|(a, b)| a.brier_q32 != b.brier_q32 || a.top1 != b.top1 || !measure::same_pick(a, b))
    {
        return Err(fail(
            "ordinary seed state disagrees with native preparation",
        ));
    }
    let headroom = json!({"rows":dev_rows.oracle_rows,"initial_q32":initial_loss,"oracle_q32":dev_rows.oracle_brier_q32,
      "gap_q32":initial_loss-dev_rows.oracle_brier_q32, "rows_with_headroom":initial.iter().zip(&dev_rows.oracle_by_row_q32).filter(|(r,o)|r.brier_q32>**o).count(),
      "source_only_q32":dev_rows.source_only_oracle_brier_q32,"class_only_q32":dev_rows.class_only_oracle_brier_q32,
      "depth":depth(&dev_pos),"dev_sha":sample::sample_digest(&dev,&dev_pos),"dev_prep_us":dev_prep_us,"dev_oracle_us":dev_rows.oracle_time.as_micros(),
      "baseline":measure::compare(&initial,&initial)?,"static_reseed":measure::compare(&initial,&random)?});
    measure::json_file(&out.join("headroom.json"), &headroom)?;
    progress(
        &mut log,
        &json!({"phase":"dev_ceiling","headroom":headroom}),
    )?;
    let names = ["geometric", "reseed_canonical", "reseed_random"];
    let mut states = [base.clone(), reseed.clone(), reseed.clone()];
    let mut rng = 4311u64;
    let mut batches = Vec::new();
    let train_start = Instant::now();
    let mut checkpoint_hashes = vec![String::new(); names.len()];
    for (b, pos) in schedule.iter().enumerate() {
        let t = Instant::now();
        let batch = reader.prepare_for_training(&train.ids, &samples(pos))?;
        let prep_us = t.elapsed().as_micros();
        if batch.batch.rows.len() != ROWS || batch.oracle_rows != 0 || !batch.oracle_time.is_zero()
        {
            return Err(fail(
                "wrong training batch cardinality or unexpected training oracle",
            ));
        }
        let canonical = p5_schedule_block(base.roots.len(), K, b as u64)?;
        let random_coords = p5_random_block_coords(base.roots.len(), K, &mut rng)?;
        let mut arms = BTreeMap::new();
        for (i, &name) in names.iter().enumerate() {
            let coords = if i == 2 { &random_coords } else { &canonical };
            let mut line = measure::matched_step(
                &gpu,
                &mut states[i],
                &batch.batch,
                coords,
                (b + i) % 2 == 0,
            )?;
            let path = out.join("states").join(format!("{name}-{b:02}.s1"));
            let (roundtrip, save_us, sha) = measure::checkpoint(&states[i], &path)?;
            states[i] = roundtrip;
            line["checkpoint_us"] = json!(save_us);
            line["checkpoint_sha256"] = json!(sha);
            checkpoint_hashes[i] = sha;
            arms.insert(name, line);
        }
        let row = json!({"batch":b,"sample_sha256":sample::sample_digest(&train,pos),"depth":depth(pos),"rows":pos.len(),"row_prep_us":prep_us,
          "row_gather_upload_us":batch.prepare_time.as_micros(),"row_gpu_us":batch.gpu_time.as_micros(),"row_materialize_us":batch.materialize_time.as_micros(),"arms":arms});
        measure::json_file(&out.join(format!("batch-{b:02}.json")), &row)?;
        progress(
            &mut log,
            &json!({"phase":"train","batch":b,"prep_us":prep_us,"geometric_epoch":states[0].epoch,"elapsed_s":total.elapsed().as_secs()}),
        )?;
        batches.push(row);
    }
    let train_wall_us = train_start.elapsed().as_micros();
    let final_t = Instant::now();
    let mut arms = BTreeMap::new();
    let mut dev_words = BTreeMap::new();
    let mut source_witness = None;
    let mut expected_pair = Vec::new();
    let mut changed_absolute_rows = Vec::new();
    for (i, &name) in names.iter().enumerate() {
        let (pred, timing) = check_dev(&gpu, &reader, &states[i], &dev, &dev_samples, &dev_rows)?;
        if i == 0 {
            let changed = |n: usize| {
                initial[n].top1 != pred[n].top1
                    || initial[n].target_mass_q48 != pred[n].target_mass_q48
                    || initial[n].top1_mass_q48 != pred[n].top1_mass_q48
                    || !measure::same_pick(&initial[n], &pred[n])
            };
            source_witness = (4..DEV).find(|&j| {
                changed(j)
                    && dev_pos[j].target_at >= 3
                    && (0..4)
                        .all(|k| public_window(&dev, dev_pos[k]) != public_window(&dev, dev_pos[j]))
            });
            for j in (0..4).chain(source_witness) {
                expected_pair.push((measure::row_json(&initial[j]), measure::row_json(&pred[j])));
            }
            changed_absolute_rows = (0..DEV).filter(|&j| changed(j)).collect::<Vec<_>>();
        }
        let from = if i == 0 { &initial } else { &random };
        arms.insert(name,json!({"vs_own_start":measure::compare(from,&pred)?,"vs_frozen_p4":measure::compare(&initial,&pred)?,"timing":timing,
          "epoch":states[i].epoch,"checkpoint_sha256":checkpoint_hashes[i],
          "changed_roots":states[i].roots.iter().zip(if i==0{&base.roots}else{&reseed.roots}).filter(|(a,b)|a!=b).count()}));
        dev_words.insert(name, pred);
    }
    let learned = named_state("geometric", &names, &states)?;
    let witness_samples: Vec<_> = (0..4)
        .chain(source_witness)
        .map(|j| dev_samples[j])
        .collect();
    let learned_codes = learned.derive_pick_codes_4()?;
    let endpoint_parity =
        reader.check_parity(p4, &dev.ids, &witness_samples, Some(&learned_codes))?;
    let before_hash = reader.mass_hashes(&dev.ids, &witness_samples, None)?;
    let after_hash = reader.mass_hashes(&dev.ids, &witness_samples, Some(&learned_codes))?;
    for (i, (a, b)) in expected_pair.iter_mut().enumerate() {
        a["mass_sha256"] = json!(before_hash[i]);
        b["mass_sha256"] = json!(after_hash[i]);
    }
    let final_us = final_t.elapsed().as_micros();
    let mut witness_rows = Vec::with_capacity(DEV);
    for (j, pos) in dev_pos.iter().enumerate() {
        let mut v = json!({"dev_index":j,"bos_offset":pos.record,"target_offset_in_record":pos.target_at,"target":dev.ids[pos.record+pos.target_at],
        "active_prefix_sha256":hex::encode(sample::prefix_digest(&dev,*pos)),
        "oracle_q32":dev_rows.oracle_by_row_q32[j],"frozen":measure::row_json(&initial[j]),"static_reseed":measure::row_json(&random[j])});
        for (&name, pred) in &dev_words {
            v[name] = measure::row_json(&pred[j]);
        }
        witness_rows.push(v);
    }
    measure::json_file(&out.join("dev_rows.json"), &json!(witness_rows))?;
    let head_to_head = json!({"geometric_vs_reseed_canonical":measure::compare(&dev_words["reseed_canonical"],&dev_words["geometric"])?,
      "geometric_vs_reseed_random":measure::compare(&dev_words["reseed_random"],&dev_words["geometric"])?});
    let gpu_summary = json!({"schema":"uor-r4.p5-wick-pilot/1","p4_sha":reader.artifact_sha256,"rows_cubin_sha":reader.cubin_sha256,
       "train_sha":sample::TRAIN_SHA,"dev_sha":sample::DEV_SHA,"trunk_sha":TRUNK_SHA,"n_batches":BATCHES,"rows":ROWS,"dev_records":DEV,
       "headroom":headroom,"arms":arms,"head_to_head":head_to_head,"fixed_dev_indices":[0,1,2,3],"extra_dev_witness":source_witness,
       "gpu_observed_changed_indices":changed_absolute_rows,"initial_epoch":base.epoch,
       "timings":{"input_us":input_us,"rows_load_us":rows_load_us,"train_wall_us":train_wall_us,"dev_prepare_us":dev_prep_us,"frozen_spot_us":frozen_spot_us,
         "initial_dev":initial_time,"reseed_dev":random_time,"final_dev_us":final_us},
       "frozen_ordinary_spots":frozen_spots,"learned_ordinary_spots":endpoint_parity,"batches":batches });
    measure::json_file(&out.join("gpu.json"), &gpu_summary)?;
    progress(
        &mut log,
        &json!({"phase":"trained_and_reloaded","elapsed_s":total.elapsed().as_secs()}),
    )?;
    drop(dev_rows);
    drop(reader);
    drop(gpu);
    let fixed = dev_pos[..4]
        .iter()
        .map(|&p| public_window(&dev, p))
        .collect::<Vec<_>>();
    let selected = source_witness.map(|j| public_window(&dev, dev_pos[j]));
    let cp = out.join("states/geometric-15.s1");
    let served = replay::run(
        p4,
        trunk,
        &cp,
        chat,
        &fixed,
        selected.as_deref(),
        &out.join("serving"),
    )?;
    if served["checkpoint_sha256"] != checkpoint_hashes[0] {
        return Err(fail("CLI replay checkpoint SHA differs from timed run"));
    }
    let actual = served["rows"]
        .as_array()
        .ok_or_else(|| fail("missing ordinary replay rows"))?;
    if actual.len() != expected_pair.len() {
        return Err(fail("CLI replay row count differs from GPU"));
    }
    for (row, (a, b)) in actual.iter().zip(&expected_pair) {
        for (key, exp) in [("baseline", a), ("trained", b)] {
            let trace = &row[key];
            for (left, right) in [
                ("token", "top1"),
                ("mass_q48", "top1_mass_q48"),
                ("pick_pos", "pick_pos"),
                ("pick_distance", "distance"),
                ("pick_successor", "successor"),
                ("pick_class", "class"),
                ("pick_ties", "ties"),
                ("bucket", "bucket"),
                ("mass_sha256", "mass_sha256"),
            ] {
                if trace[left] != exp[right] {
                    return Err(fail(format!("ordinary {key} {left} differs from GPU")));
                }
            }
        }
    }
    measure::json_file(
        &out.join("complete.json"),
        &json!({"ok":true,"total_us":total.elapsed().as_micros(),"ordinary_witness_rows":actual.len(),
      "selected_cli_row":served["selected_cli_row"],"reloaded_state_sha":checkpoint_hashes[0]}),
    )?;
    progress(
        &mut log,
        &json!({"phase":"complete","elapsed_s":total.elapsed().as_secs()}),
    )?;
    Ok(())
}
