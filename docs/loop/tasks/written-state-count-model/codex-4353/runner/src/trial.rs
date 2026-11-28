use crate::{
    device::{self, Capture, Gpu, Mode},
    fail, held,
    p4_cuda::{split_uorp4a01_sections, P4GpuEngine},
    Result,
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::os::unix::fs::DirBuilderExt;
use std::{
    collections::BTreeSet,
    fs,
    io::{self, BufRead, BufReader},
    path::Path,
    time::Instant,
};
use uor_r4_integer::discrete_d11::{
    count_deployment::{load_client_artifact, ClientArtifactFile, DiscreteCountModel},
    native_occurrence::lexical_commit::LexicalHead,
    p4_lexical_bfs,
};

const ARTIFACT: &str = "e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776";
const STATE: &str = "368033364f68ec16523735adf71b036b56902c5a9e9a816e6146a85a2874e778";
const TRAIN: &str = "5dd411d7dee1335784efc5868adf58a6979da18d4b743809e36b12bd44dc6829";
const LABELS: &str = "e26f239df8aba2e977bf5372dac8cdc9511e516dcf38914443cdcd9cd5431872";
pub struct Engines {
    pub lexical: Gpu,
    pub p4: P4GpuEngine,
}
impl Engines {
    pub fn capture(&self, prefixes: &[Vec<u16>]) -> Result<Capture> {
        let histories: Vec<_> = prefixes
            .iter()
            .map(|p| {
                let reset = p.iter().rposition(|&id| id == 0).unwrap_or(0);
                p[reset.max(p.len().saturating_sub(2048))..].to_vec()
            })
            .collect();
        let mut priors = Vec::with_capacity(histories.len() * 4095);
        for batch in histories.chunks(64) {
            let (steps, raw) =
                self.p4
                    .run_parity_batch(batch, &vec![1; batch.len()], &vec![true; batch.len()])?;
            if steps.len() != batch.len()
                || steps
                    .iter()
                    .zip(batch)
                    .any(|(s, h)| s.status != 0 || s.context_tokens as usize != h.len())
            {
                return Err(fail("P4 scoring failed or evaluated a different context"));
            }
            priors.extend(raw);
        }
        self.lexical.capture(&histories, &priors)
    }
}
pub struct Arm {
    pub name: String,
    pub mode: Mode,
    pub words: [i16; 256],
}
#[derive(Deserialize)]
struct Input {
    group: usize,
    turn: usize,
    lexical_prefix: Vec<u16>,
}
#[derive(Deserialize)]
struct Label {
    group: usize,
    turn: usize,
    target: u16,
}
pub fn verify(path: &Path, digest: &str) -> Result<()> {
    let mut sha = Sha256::new();
    io::copy(&mut fs::File::open(path)?, &mut sha)?;
    if hex::encode(sha.finalize()) != digest {
        return Err(fail(format!("input checksum differs: {}", path.display())));
    }
    Ok(())
}
fn rows<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(fs::File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}
fn orders(seed: u32, count: usize) -> Vec<Vec<u32>> {
    (0..8u32)
        .map(|epoch| {
            let mut ids: Vec<_> = (0..count as u32).collect();
            ids.sort_by_cached_key(|id| {
                let mut hash = Sha256::new();
                for v in [seed, epoch, *id] {
                    hash.update(v.to_le_bytes());
                }
                (<[u8; 32]>::from(hash.finalize()), *id)
            });
            ids
        })
        .collect()
}
pub fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "finite" {
        held::finite()?;
        device::finite()?;
        println!("finite PASS");
        return Ok(());
    }
    let [command, p4_file, inputs, output] = args.as_slice() else {
        return Err(fail(
            "expected finite or run P4 VERIFIED_PACKET_ROOT FRESH_OUTPUT",
        ));
    };
    if command != "run" {
        return Err(fail("unknown command"));
    }
    let packet = Path::new(inputs);
    let out = Path::new(output);
    fs::DirBuilder::new().mode(0o700).create(out)?;
    let started = Instant::now();
    let train = packet.join("train/inputs.jsonl");
    let labels = packet.join("train/labels.jsonl");
    verify(&train, TRAIN)?;
    verify(&labels, LABELS)?;
    let inputs: Vec<Input> = rows(&train)?;
    let labels: Vec<Label> = rows(&labels)?;
    let mut keys = BTreeSet::new();
    if inputs.len() != 3891
        || labels.len() != inputs.len()
        || inputs.iter().zip(&labels).any(|(r, l)| {
            (r.group, r.turn) != (l.group, l.turn)
                || !keys.insert((r.group, r.turn))
                || r.lexical_prefix.is_empty()
                || r.lexical_prefix.len() > 2048
                || !(2..4096).contains(&l.target)
                || l.target == 223
        })
    {
        return Err(fail("invalid or misjoined source training packet"));
    }
    let targets: Vec<_> = labels.iter().map(|l| l.target).collect();
    let prefixes: Vec<_> = inputs.into_iter().map(|r| r.lexical_prefix).collect();
    let ClientArtifactFile::Embedded {
        model,
        state_bytes,
        expected_state_sha256,
        identity,
    } = load_client_artifact(Path::new(p4_file), false)?
    else {
        return Err(fail("expected complete embedded P4"));
    };
    if identity.artifact_sha256 != ARTIFACT || expected_state_sha256 != STATE {
        return Err(fail("unexpected artifact or bound state"));
    }
    let deploy =
        DiscreteCountModel::bind_embedded(model, &state_bytes, &expected_state_sha256, identity)?;
    let inner = &deploy.p4.as_ref().ok_or_else(|| fail("no P4"))?.inner;
    let bind = inner
        .state_sha256()
        .ok_or_else(|| fail("no state identity"))?;
    if hex::encode(bind) != STATE {
        return Err(fail("bound runtime state differs"));
    }
    let raw = split_uorp4a01_sections(&state_bytes)?;
    let engines = Engines {
        lexical: Gpu::new(&deploy.model.p4_lexical_trunk_words()?, p4_lexical_bfs())?,
        p4: P4GpuEngine::new(
            raw.counts_v3,
            raw.s3_compact,
            raw.s2_skip_compact,
            raw.s4_table,
            &inner.s6.weights_q32,
            &inner.codes,
        )?,
    };
    let load_ms = started.elapsed().as_millis();
    let capture = engines.capture(&prefixes)?;
    let capture_ms = started.elapsed().as_millis() - load_ms;
    let mut arms = vec![Arm {
        name: "zero".into(),
        mode: Mode::Geometry,
        words: [0; 256],
    }];
    let mut fit_log = vec![];
    for seed in [435300, 435301] {
        let schedule = orders(seed, targets.len());
        for (tag, mode) in [("geometry", Mode::Geometry), ("ordered", Mode::Ordered)] {
            let clock = Instant::now();
            let fit = engines.lexical.fit(&capture, &targets, &schedule, mode)?;
            let name = format!("{tag}-{seed}");
            fit_log.push(json!({"arm":name,"epochs":fit.epochs,"fit_ms":clock.elapsed().as_millis(),"nonzero":fit.weights.iter().filter(|&&w|w!=0).count()}));
            arms.push(Arm {
                name,
                mode,
                words: fit.weights,
            });
        }
    }
    for arm in &arms {
        let bytes = LexicalHead::encode_bound(bind, arm.mode == Mode::Ordered, &arm.words);
        let path = out.join(format!("{}.lex", arm.name));
        fs::write(&path, bytes)?;
        let restored = fs::read(&path)?;
        if restored != bytes
            || LexicalHead::bind(&restored, &bind)?.is_ordered() != (arm.mode == Mode::Ordered)
            || LexicalHead::bind(&restored, &[0; 32]).is_ok()
        {
            return Err(fail("roundtrip/state control failed"));
        }
    }
    drop(capture);
    fs::write(
        out.join("fit.json"),
        serde_json::to_vec_pretty(
            &json!({"artifact":ARTIFACT,"state":STATE,"device_cubin":engines.lexical.digest(),"training_rows":targets.len(),"load_ms":load_ms,"capture_ms":capture_ms,"arms":fit_log}),
        )?,
    )?;
    for (pool, blind_hash, truth_hash) in [
        (
            "supplement",
            "5142514aa245d7cf99690bb60e4f5120cce36094a33678f50e52573912c23d62",
            "483bf82f99e7ba454986726abc98b9bdee66b0df230b6a2c089a862dd364feee",
        ),
        (
            "dev",
            "0e16de2d14aa0335b4646037f9c8d6ffeed453fd4e00893820617c042f0e970f",
            "40f4166523171de1ca599c8250c0f86e8fdf3db62e9528f6cd0b0e129602eb58",
        ),
    ] {
        let blind = packet.join(pool).join("blind.jsonl");
        let truth = packet.join(pool).join("truth.jsonl");
        verify(&blind, blind_hash)?;
        let dir = out.join(pool);

        held::run(&engines, &arms, &blind, &truth, &dir)?;
        verify(&truth, truth_hash)?;
    }
    fs::write(
        out.join("elapsed-ms"),
        started.elapsed().as_millis().to_string(),
    )?;
    Ok(())
}
