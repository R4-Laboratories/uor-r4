use serde_json::{json, Value};

use crate::{address, check, Bank, Result, Row};

pub fn aid(raw: &[u8], rows: &[Row], labels: &[[[usize; 2]; 2]]) -> Result<()> {
    check(
        raw.len() == 12 + 24 * rows.len() && raw.starts_with(b"UORAID01"),
        "wrong supplied-address packet",
    )?;
    let fields = raw[8..]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        .collect::<Vec<_>>();
    check(
        fields[0] == rows.len() && labels.len() == rows.len(),
        "wrong supplied address count",
    )?;
    for (i, item) in fields[1..].chunks_exact(6).enumerate() {
        let [a, e] = labels[i];
        check(
            item == [a[0], e[0], a[0], a[1], e[0], e[1]]
                || item == [a[0], e[0], e[0], e[1], a[0], a[1]],
            format!("supplied packet differs from physical original addresses at {i}"),
        )?;
        let row = &rows[i];
        check(
            a[0] != e[0] && a[1] != e[1],
            format!("reused supplied address at {i}"),
        )?;
        for word in [a[0], e[0]] {
            for site in [a[1], e[1]] {
                check(
                    !row.words
                        .get(word)
                        .ok_or("unknown supplied governor")?
                        .address
                        .overlaps(row.sites.get(site).ok_or("unknown supplied site")?.address),
                    format!("overlapping original control at {i}"),
                )?;
            }
        }
    }
    Ok(())
}

pub fn frozen(
    ordinal: usize,
    row: &Row,
    banks: &[Bank; 3],
    label: Option<[[usize; 2]; 2]>,
    arm: &str,
) -> Result<Value> {
    check(
        matches!(
            (arm, label),
            ("native", None) | ("correct" | "exchanged", Some(_))
        ),
        "unsupported frozen control",
    )?;
    let costs = row
        .sites
        .iter()
        .map(|site| -> Result<[u64; 3]> {
            let mut values = [0; 3];
            for (i, bank) in banks.iter().enumerate() {
                values[i] = bank
                    .iter()
                    .map(|center| {
                        center
                            .iter()
                            .zip(&site.point)
                            .map(|(a, b)| u64::from(a.abs_diff(*b)))
                            .sum()
                    })
                    .min()
                    .ok_or("empty frozen tuple bank")?;
            }
            Ok(values)
        })
        .collect::<Result<Vec<_>>>()?;
    let labels = label.map(|mut labels| {
        if arm == "exchanged" {
            let [a, e] = &mut labels;
            std::mem::swap(&mut a[1], &mut e[1]);
        }
        labels
    });
    let mut candidates = row
        .tuples
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, t)| labels.is_none_or(|[a, e]| t[..2] == [a[1], e[1]]))
        .map(|(i, t)| (costs[t[0]][0] + costs[t[1]][1] + costs[t[2]][2], i, t))
        .collect::<Vec<_>>();
    candidates.sort_unstable();
    let winner_bank = candidates.first().map(|c| c.0);
    let rival_bank = candidates.get(1).map(|c| c.0);
    let ties = candidates
        .iter()
        .take_while(|c| Some(c.0) == winner_bank)
        .count();
    let winner = candidates.first().filter(|_| ties == 1);
    let links = std::array::from_fn::<_, 2, _>(|sign| {
        let link = labels.map(|l| l[sign]);
        let present = usize::from(link.is_some());
        json!({"word":link.map(|l|l[0]),"site":link.map(|l|l[1]),
            "governor":link.map(|l|address(row.words[l[0]].address)),"noun":link.map(|l|address(row.sites[l[1]].address)),
            "word_ties":present,"site_ties":present,"link_ties":present,"eligible":present,
            "best":[null,null],"rival":[null,null]})
    });
    Ok(
        json!({"row":ordinal,"arm":arm,"links":links,"independent_links":links,"collision":false,
        "tuple":{"ordinal":winner.map(|c|c.1),"sites":winner.map(|c|c.2),
            "addresses":winner.map(|c|c.2.map(|i|address(row.sites[i].address))),"ties":ties,"eligible":candidates.len(),
            "bank":winner_bank,"rival_bank":rival_bank,"best":[null,null,winner_bank],"rival":[null,null,rival_bank]}}),
    )
}
