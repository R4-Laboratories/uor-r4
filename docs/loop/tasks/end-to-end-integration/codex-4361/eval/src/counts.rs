fn grids(blocks: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub fn counts(args: &[String]) -> Result<()> {
    let [kernel, ksha, train, train_sha, held, held_sha, output] = args else {
        return Err("counts CUBIN SHA TRAIN SHA HELD SHA FRESH_JSONL".into());
    };
    let fit = load_articles(&checked_input(train, train_sha)?)?;
    let panel = load_articles(&checked_input(held, held_sha)?)?;
    let ctx = CudaContext::new(0)?;
    if ctx.compute_capability()? != (8, 9) {
        return Err("counts requires sm_89".into());
    }
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_binary(checked_input(kernel, ksha)?))?;
    let mut prev = Vec::new();
    let mut next = Vec::new();
    for article in &fit {
        let mut last = *article.row.prefix.last().ok_or("empty prefix")?;
        for &token in &article.row.expected {
            prev.push(last);
            next.push(token);
            last = token;
        }
    }
    let d_prev = stream.clone_htod(&prev)?;
    let d_next = stream.clone_htod(&next)?;
    let mut table = stream.alloc_zeros::<u32>(4097 * 4096)?;
    let mut winners = stream.alloc_zeros::<u16>(4097)?;
    let n = u32::try_from(prev.len())?;
    let add = module.load_function("add_counts")?;
    unsafe {
        stream
            .launch_builder(&add)
            .arg(&d_prev)
            .arg(&d_next)
            .arg(&n)
            .arg(&mut table)
            .launch(grids(prev.len().div_ceil(256)))?;
    }
    let select = module.load_function("select_ids")?;
    unsafe {
        stream
            .launch_builder(&select)
            .arg(&table)
            .arg(&mut winners)
            .launch(grids(4097))?;
    }
    let elected = stream.clone_dtoh(&winners)?;
    let global = elected[4096];
    // An independent recount of supervised counts verifies the GPU artifact.
    let mut host = BTreeMap::<(usize, u16), u32>::new();
    for (&p, &t) in prev.iter().zip(&next) {
        *host.entry((p as usize, t)).or_default() += 1;
        *host.entry((4096, t)).or_default() += 1;
    }
    let hist = stream.clone_dtoh(&table)?;
    if hist.iter().filter(|&&count| count != 0).count() != host.len()
        || host
            .iter()
            .any(|(&(row, token), &count)| hist[row * 4096 + usize::from(token)] != count)
    {
        return Err("GPU histogram differs from the independent TRAIN recount".into());
    }
    let mut gold = vec![(0u32, 0u16); 4097];
    for ((row, token), count) in host {
        if count > gold[row].0 {
            gold[row] = (count, token);
        }
    }
    if global == 0 || !gold.iter().map(|x| x.1).eq(elected.iter().copied()) {
        return Err("count election mismatch".into());
    }
    let tails: Vec<_> = panel
        .iter()
        .map(|a| a.row.prefix[a.row.prefix.len() - 1])
        .collect();
    let d_tails = stream.clone_htod(&tails)?;
    let mut d_out = stream.alloc_zeros::<u16>(panel.len() * 64)?;
    let m = u32::try_from(panel.len())?;
    let predict = module.load_function("predict")?;
    unsafe {
        stream
            .launch_builder(&predict)
            .arg(&winners)
            .arg(&d_tails)
            .arg(&m)
            .arg(&mut d_out)
            .launch(grids((panel.len() * 2).div_ceil(256)))?;
    }
    let results = stream.clone_dtoh(&d_out)?;
    let mut file = BufWriter::new(
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output)?,
    );
    for (i, article) in panel.iter().enumerate() {
        let mut row = json!({"id":article.id,"train_sha256":train_sha,"held_sha256":held_sha});
        for (k, name) in ["unigram", "successor"].iter().enumerate() {
            let off = (k * panel.len() + i) * 32;
            let ids = &results[off..off + 32];
            let len = ids.iter().position(|&id| id == 1).map_or(32, |pos| pos + 1);
            let mut last = tails[i];
            for &id in &ids[..len] {
                let predicted = if k == 1 && elected[last as usize] != 0 {
                    elected[last as usize]
                } else {
                    global
                };
                if predicted != id {
                    return Err("GPU count rollout mismatch".into());
                }
                last = id;
            }
            if ids[..len].contains(&0) {
                return Err("count decoder emitted BOS".into());
            }
            row[*name] = json!(&ids[..len]);
        }
        writeln!(file, "{row}")?;
    }
    file.flush()?;
    println!(
        "{}",
        json!({"gpu":ctx.name()?,"training_targets":n,"rows":panel.len(),"unigram":global,
        "election_bytes":elected.len()*2,"training_table_bytes":4097*4096*4,
        "histogram_words_checked":hist.len(),"elected_rows_checked":4097})
    );
    Ok(())
}
