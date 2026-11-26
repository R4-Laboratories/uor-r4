// The signed interpreter above is the independently maintained #4289 implementation.
pub fn read_labels(path: &Path, rows: usize) -> Result<Vec<[[usize; 2]; 2]>> {
    labels(path, rows)
}

pub struct Observer<'a> {
    examples: Vec<Example<'a>>,
}

impl<'a> Observer<'a> {
    pub fn new(rows: &'a [Row], labels: Vec<[[usize; 2]; 2]>, state: &[Sign; 2]) -> Result<Self> {
        let mut examples = Vec::new();
        for (row, labels) in rows.iter().zip(labels) {
            for (sign, link) in labels.into_iter().enumerate() {
                examples.push(Example::new(row, sign, link, state)?);
            }
        }
        Ok(Self { examples })
    }

    pub fn observations(&self, state: &[Sign; 2]) -> Result<Vec<Value>> {
        let bank = std::array::from_fn(|sign| state[sign].transport.clone());
        self.examples
            .iter()
            .enumerate()
            .map(|(index, e)| {
                Ok(json!({
                    "row": index / 2, "sign": e.sign, "governor_word": e.word, "gold_site": e.gold,
                    "eligible_for_fit": e.eligible(), "eligible_rivals": e.sites.len(),
                    "before": brief(&e.read(&bank)?)
                }))
            })
            .collect()
    }

    pub fn steps(&self, state: &[Sign; 2]) -> Result<[[[i32; EDGE]; 8]; 2]> {
        let bank = std::array::from_fn(|sign| state[sign].transport.clone());
        let mut votes = [[[0; EDGE]; 8]; 2];
        for example in self.examples.iter().filter(|e| e.eligible()) {
            let read = example.read(&bank)?;
            if read.margin.ok_or("eligible edge has no margin")? > 0 {
                continue;
            }
            let first_rival = *read.rivals.first().ok_or("violated edge has no rival")?;
            let center = read
                .costs
                .iter()
                .find(|c| c.site == first_rival)
                .ok_or("missing first rival cost")?
                .center;
            for (site, slot, direction) in [
                (example.gold, read.gold_center, 1),
                (first_rival, center, -1),
            ] {
                let point = example.edge(site);
                for axis in 0..EDGE {
                    let base = bank[example.sign][slot][axis];
                    votes[example.sign][slot][axis] +=
                        direction * (i32::from(point[axis]) - i32::from(base)).signum();
                }
            }
        }
        Ok(votes)
    }
}
