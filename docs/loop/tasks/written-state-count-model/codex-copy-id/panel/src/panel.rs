use super::*;
use history::Prior;
use serde_json::{json, Value};

const SUBJECTS: &str = "Abel Abram Adele Agnes Aidan Alan Alina Alma Alvin Amos Anita Ansel Anton Arlo Arnold Astrid Avery Basil Berta Boris Bran Bruce Bruno Byron Celia Cora Cyrus Dante Darcy Daria Declan Diego Dorian Duncan Edna Edwin Elaine Elise Elsa Elton Enid Erin Ethan Ethel Evan Ezra Flora Floyd Freya Frida Gemma Gina Gloria Greta Hugo Ida Ilan Ines Ivan Ivor Jade Jake Jasper Jessa Jonah Jude Julius Kira Lara Leon Leroy Levi Lina Lionel Logan Lola Lorna Luca Lyle Maeve Maia Marco Margot Marta Matilda Mina Milo Mona Nadia Nando Natalie Nell Niko Nola Octavia Odin Omar Orion Orla Otis Otto Petra Quinn Ramona Remy Rhoda Roman Rory Rowan Rudy Sacha Sadie Sasha Saul Seth Sheila Silas Sonia Theo Toby Tony Una Ursula Val Valeria Vance Veda Vince Viola Willa Wilma Xena Yara Yona Yuri Zane Zara Zelda";
const LABELS: &str = "abbey acorn acre alchemy almond altar amp amber angle antler anvil apron arcade arch arrow ash atlas avenue awning axle badge badger bagel bamboo banana banner barley barn barrel basil basin basket baton battery bay beacon bead beans bear beet bell belt bench berry birch biscuit blade blanket blaze bloom blossom bluff bolt bone bonnet boot border bottle bowl branch brass bread breeze brick bronze broom brush bubble bucket bulb bundle button cabin cable cactus cage cake camel candle candy canoe canvas canyon cape capsule caravan carpet carrot carton cart cedar ceiling cellar chain chalk chapel charm chart checker cherry chest chick chili chime chip cider circus clam clay cliff cloak clock cloth cloud clover coal coast cocoa collar comet compass copper coral cord cork corn cotton cove crab crane creek crest crown crystal cube curtain daisy dart dawn deer delta den dew diamond dock door dove drum dune dusk dust eagle earth echo elm enamel engine fern field fig fin fir flag flame flax fleece flint floor flour flute fog fork fox frame frost fruit funnel fur garden garlic gauze gem ginger glass globe glove glue goat goblet granite grape grass gravel grove guava gull gum hail halo hammer harp hay herb hide hill hive holly honey hood hook horn hour ink iris iron island ivory jade jam jar jasmine jelly jet jewel kiln kite kiwi knot lace lagoon lamp lance lantern latch laurel leaf leek lemon lens lichen lilac lily lime linen lion lizard llama loaf lodge lotus lye maple marble marsh mast meadow melon mesh mint mirror mist mold moss moth mouse muffin mule nectar nest nickel oak oasis oat olive onyx opal orange orchid ore otter owl paddle palm panda pearl pebble pecan pedal pepper perch petal piano pickle pier pigeon pine plank plum plume pocket pond poppy porch powder prism pulse pumpkin quartz quill quilt rabbit rain raisin rake ramp raven reed reef ribbon ridge rifle robin robot roof root rope rose rosemary rug sage sail salt sand satin sauce scallop scale scarf seed shale shawl shell shield shore shrub silk skate slate sled slope slug smoke snail soap sock soda soil soot spade sparrow spear spice sponge spool spruce squash squirrel staff steam steel stem stitch stone stump sugar swan syrup tack tape tea teal thorn thyme tick tile timber tin toast tomato torch tower trout tulip tuna turf twig urn vase velvet vine violet visor walnut wand wasp wax wheat wheel wick willow wind wire wool wreath wrench yam zinc";

#[derive(Clone, Copy)]
struct Family {
    name: &'static str,
    lead: &'static str,
    subject_first: &'static str,
    value_first: &'static str,
    omitted: &'static str,
    question: &'static str,
}

impl Family {
    fn fact(self, subject: &str, label: Option<&str>, value_first: bool) -> String {
        match label {
            None => self.omitted.replace("{subject}", subject),
            Some(label) => (if value_first {
                self.value_first
            } else {
                self.subject_first
            })
            .replace("{subject}", subject)
            .replace("{value}", label),
        }
    }

    fn question(self, subject: &str, punctuation: bool) -> String {
        let instruction = if punctuation {
            "Write the single word followed by a period."
        } else {
            "Write only the word without punctuation."
        };
        format!(
            "{} {instruction}",
            self.question.replace("{subject}", subject)
        )
    }
}

const FAMILIES: [Family; 4] = [
    Family {
        name: "observatory",
        lead: "The observatory keeps a code beside every telescope booking.",
        subject_first: "For the booking filed by {subject}, the code is {value}.",
        value_first: "A code of {value} appears beside the booking from {subject}.",
        omitted: "The booking submitted by {subject} was checked in the morning.",
        question: "Which booking code should I look up for {subject}?",
    },
    Family {
        name: "marionette",
        lead: "The puppet troupe puts a private word on each marionette case.",
        subject_first: "The case transported by {subject} bears the word {value}.",
        value_first: "The word {value} is painted on the case transported by {subject}.",
        omitted: "The case transported by {subject} has a broken strap.",
        question: "What word would I see on the case transported by {subject}?",
    },
    Family {
        name: "trail_checkpoint",
        lead: "The trail repair crew identifies each field station with a call word.",
        subject_first: "The station inspected by {subject} has the call word {value}.",
        value_first: "A call word of {value} identifies the station inspected by {subject}.",
        omitted: "The station inspected by {subject} was checked before lunch.",
        question: "Give me the call word of the trail station inspected by {subject}.",
    },
    Family {
        name: "sample_cabinet",
        lead: "The stone collection marks each sample drawer with a word.",
        subject_first: "The drawer cataloged by {subject} is marked {value}.",
        value_first: "The marking {value} indicates the drawer cataloged by {subject}.",
        omitted: "The drawer cataloged by {subject} is on the lower shelf.",
        question: "What marking identifies the drawer cataloged by {subject}?",
    },
];

#[derive(Clone, Copy)]
struct Label<'a> {
    id: u32,
    text: &'a str,
}

#[derive(Clone, Copy)]
struct Settings {
    family: Family,
    first_fact_side: usize,
    value_first: bool,
    punctuation: bool,
    collision: bool,
}

impl Settings {
    fn stratum(self) -> &'static str {
        if self.collision {
            "same_code_1808_3984"
        } else {
            "distinct_held_words"
        }
    }
}

#[derive(Clone)]
struct Group<'a> {
    id: String,
    settings: Settings,
    names: [&'a str; 3],
    labels: [Label<'a>; 2],
}

#[derive(Clone, Copy)]
enum Kind {
    Positive,
    MissingValue,
    MissingSubject,
    Duplicate { note_first: bool },
    MalformedSeparator,
    MalformedMarker,
}

impl Kind {
    fn cohort(self) -> &'static str {
        match self {
            Self::Positive => "controlled",
            Self::MissingValue => "value_omitted",
            Self::MissingSubject => "subject_unbound",
            Self::Duplicate { .. } => "same_id_wrong_address",
            Self::MalformedSeparator | Self::MalformedMarker => "malformed",
        }
    }

    fn variant(self) -> Option<&'static str> {
        match self {
            Self::Duplicate { note_first: true } => Some("duplicate_note_before_authorizing_fact"),
            Self::Duplicate { note_first: false } => Some("duplicate_note_after_authorizing_fact"),
            Self::MalformedSeparator => Some("single_space_separator"),
            Self::MalformedMarker => Some("broken_full_assistant_marker"),
            _ => None,
        }
    }
}

type Record = (Row, Key);

struct GroupRows {
    held: Vec<Record>,
    absent: Vec<Record>,
}

pub(crate) struct Tapes {
    pub held: Vec<Record>,
    pub absent: Vec<Record>,
    pub finite: Vec<Record>,
    pub audit: Value,
}

struct Candidate {
    text: String,
    fact_bounds: [[usize; 2]; 2],
    note_bounds: Option<[usize; 2]>,
}

fn choose(domain: &str, key: &str, count: usize) -> usize {
    let digest = Sha256::digest(format!("{SEED}\0{domain}\0{key}"));
    let mut first = [0; 8];
    first.copy_from_slice(&digest[..8]);
    (u64::from_le_bytes(first) % count as u64) as usize
}

fn key_id(key: &str) -> String {
    hex::encode(&Sha256::digest(format!("{SEED}\0{key}"))[..12])
}

fn append_sentence(body: &mut String, sentence: &str) -> [usize; 2] {
    if !body.is_empty() {
        body.push(' ');
    }
    let start = body.len();
    body.push_str(sentence);
    [start, body.len()]
}

impl Group<'_> {
    fn context(&self, labels: [Label<'_>; 2], side: usize, kind: Kind) -> Candidate {
        let mut text = self.settings.family.lead.to_owned();
        let mut note_bounds = None;
        let mut fact_bounds = [[0; 2]; 2];
        let note = format!(
            "On the unrelated sketch, {} appears as an example word.",
            labels[side].text
        );
        if matches!(kind, Kind::Duplicate { note_first: true }) {
            note_bounds = Some(append_sentence(&mut text, &note));
        }
        for fact_side in [
            self.settings.first_fact_side,
            1 - self.settings.first_fact_side,
        ] {
            let label = (!(matches!(kind, Kind::MissingValue) && side == fact_side))
                .then_some(labels[fact_side].text);
            let fact =
                self.settings
                    .family
                    .fact(self.names[fact_side], label, self.settings.value_first);
            fact_bounds[fact_side] = append_sentence(&mut text, &fact);
        }
        if matches!(kind, Kind::Duplicate { note_first: false }) {
            note_bounds = Some(append_sentence(&mut text, &note));
        }
        Candidate {
            text,
            fact_bounds,
            note_bounds,
        }
    }

    fn row(
        &self,
        builder: &Builder<'_>,
        prior: &Prior,
        assignment: usize,
        side: usize,
        kind: Kind,
    ) -> Result<Option<Record>> {
        let labels = if assignment == 0 {
            self.labels
        } else {
            [self.labels[1], self.labels[0]]
        };
        let [target, rival] = [labels[side], labels[1 - side]];
        let candidate = self.context(labels, side, kind);
        let requested_subject = self.names[if matches!(kind, Kind::MissingSubject) {
            2
        } else {
            side
        }];
        let question = self
            .settings
            .family
            .question(requested_subject, self.settings.punctuation);
        let supported = matches!(kind, Kind::Positive | Kind::Duplicate { .. });
        let reply = supported.then(|| {
            if self.settings.punctuation {
                format!("{}.", target.text)
            } else {
                target.text.to_owned()
            }
        });
        let separator = if matches!(kind, Kind::MalformedSeparator) {
            " "
        } else {
            "\n\n"
        };
        let Some(mut case) =
            builder.case(&candidate.text, &question, reply.as_deref(), separator)?
        else {
            return Ok(None);
        };
        let was_canonical = !matches!(kind, Kind::MalformedMarker);
        if !was_canonical {
            let marker_start = case.prefix.len() - MARKER.len();
            case.prefix[marker_start + 1] = SPACE;
        }
        if !prior.fresh_prompt(&question, &case.user, &case.prefix) {
            return Ok(None);
        }
        let rival_pos = case.unique_source(rival.id);
        if rival_pos.is_none() {
            return Ok(None);
        }
        let (gold_pos, wrong_same_id_pos, witness) = match kind {
            Kind::Duplicate { .. } => {
                let fact = candidate.fact_bounds[side];
                let note = candidate
                    .note_bounds
                    .ok_or("duplicate is missing its note")?;
                let (Some(fact_token_pos), Some(note_token_pos)) = (
                    case.source_in(builder.tokenizer, target.id, fact),
                    case.source_in(builder.tokenizer, target.id, note),
                ) else {
                    return Ok(None);
                };
                let mut expected_positions = [fact_token_pos, note_token_pos];
                expected_positions.sort_unstable();
                if fact_token_pos == note_token_pos
                    || case.positions(target.id) != expected_positions
                {
                    return Ok(None);
                }
                let witness = Witness {
                    authorizing_fact_bytes: fact,
                    duplicate_note_bytes: note,
                    fact_text: candidate.text[fact[0]..fact[1]].into(),
                    note_text: candidate.text[note[0]..note[1]].into(),
                    fact_token_pos,
                    note_token_pos,
                    original_id: target.id,
                };
                (Some(fact_token_pos), Some(note_token_pos), Some(witness))
            }
            Kind::Positive => {
                let Some(pos) = case.unique_source(target.id) else {
                    return Ok(None);
                };
                if case.source_in(builder.tokenizer, target.id, candidate.fact_bounds[side])
                    != Some(pos)
                {
                    return Ok(None);
                }
                (Some(pos), None, None)
            }
            Kind::MissingValue => {
                if !case.positions(target.id).is_empty() {
                    return Ok(None);
                }
                (None, None, None)
            }
            Kind::MissingSubject | Kind::MalformedSeparator | Kind::MalformedMarker => {
                if case.unique_source(target.id).is_none() {
                    return Ok(None);
                }
                (None, None, None)
            }
        };
        if supported
            && (case.gold.first() != Some(&target.id)
                || case.gold.len() != if self.settings.punctuation { 3 } else { 2 })
        {
            return Ok(None);
        }
        if supported && self.settings.punctuation && builder.piece(case.gold[1]) != "." {
            return Ok(None);
        }
        let group_id = self.id.clone();
        let pair_id = key_id(&format!(
            "{group_id}/{assignment}/{}",
            kind.variant().unwrap_or(kind.cohort())
        ));
        let key = Key {
            pair_id: pair_id.clone(),
            group_id: group_id.clone(),
            assignment,
            side,
            matched_group: matches!(kind, Kind::MissingValue | Kind::MissingSubject)
                .then(|| group_id.clone()),
            context: candidate.text,
            user: case.user,
            question,
            reply,
            pieces: case.gold.iter().map(|&id| builder.piece(id)).collect(),
            topic: self.settings.family.name,
            subjects: self.names[..2]
                .iter()
                .map(|name| name.to_string())
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| "two subject names required")?,
            values_by_subject: labels.map(|label| label.text.to_owned()),
            requested_subject: requested_subject.into(),
            first_fact_side: self.settings.first_fact_side,
            value_before_subject: self.settings.value_first,
            punctuation: self.settings.punctuation,
            occurrence_witness: witness,
            gold: case.gold,
        };
        let row = Row {
            pair_id,
            group_id,
            assignment,
            side,
            cohort: kind.cohort().into(),
            stratum: self.settings.stratum().into(),
            prefix: case.prefix,
            target: target.id,
            rival: rival.id,
            raw_target: target.id,
            raw_rival: rival.id,
            lexical_offset: 0,
            leading_whitespace: Vec::new(),
            space_token: SPACE,
            equal_length: matches!(kind, Kind::Positive),
            context_start: case.context_start,
            context_end: case.context_end,
            gold_pos,
            rival_pos,
            wrong_same_id_pos,
            source_status: if supported { "unique" } else { "absent" },
            expected_second: key.gold.get(1).copied(),
            expected_terminal_after_first: supported.then_some(false),
            expected_terminal_after_second: supported.then_some(!self.settings.punctuation),
            variant: kind.variant(),
        };
        Ok(Some((row, key)))
    }
}

fn quartet(
    group: &Group<'_>,
    builder: &Builder<'_>,
    prior: &Prior,
    serial: usize,
) -> Result<Option<GroupRows>> {
    let mut held = Vec::with_capacity(4);
    let mut absent = Vec::with_capacity(3);
    for assignment in 0..2 {
        for side in 0..2 {
            let Some(row) = group.row(builder, prior, assignment, side, Kind::Positive)? else {
                return Ok(None);
            };
            held.push(row);
        }
    }
    let order = serial % 2;
    for side in 0..2 {
        let Some(row) = group.row(builder, prior, order, side, Kind::MissingValue)? else {
            return Ok(None);
        };
        absent.push(row);
    }
    let Some(row) = group.row(builder, prior, 1 - order, order, Kind::MissingSubject)? else {
        return Ok(None);
    };
    absent.push(row);
    if !is_crossed(&held) {
        return Ok(None);
    }
    Ok(Some(GroupRows { held, absent }))
}

fn is_crossed(rows: &[Record]) -> bool {
    let [(a, ka), (b, kb), (c, kc), (d, kd)] = rows else {
        return false;
    };
    let question_swap = |left: &Row, right: &Row, kl: &Key, kr: &Key| {
        left.pair_id == right.pair_id
            && left.assignment == right.assignment
            && left.side != right.side
            && left.prefix.len() == right.prefix.len()
            && left.context_end == right.context_end
            && left.prefix[..left.context_end] == right.prefix[..right.context_end]
            && kl.context == kr.context
            && kl.question != kr.question
            && left.gold_pos == right.rival_pos
            && left.rival_pos == right.gold_pos
            && left.target == right.rival
            && left.rival == right.target
    };
    let assignment_swap = |left: &Row, right: &Row, kl: &Key, kr: &Key| {
        let difference = left
            .prefix
            .iter()
            .zip(&right.prefix)
            .enumerate()
            .filter_map(|(pos, (l, r))| (l != r).then_some(pos))
            .collect::<Vec<_>>();
        left.pair_id != right.pair_id
            && left.assignment != right.assignment
            && left.side == right.side
            && left.prefix.len() == right.prefix.len()
            && left.context_end == right.context_end
            && left.gold_pos == right.gold_pos
            && left.rival_pos == right.rival_pos
            && difference.len() == 2
            && difference
                .iter()
                .all(|pos| Some(*pos) == left.gold_pos || Some(*pos) == left.rival_pos)
            && kl.question == kr.question
            && left.target == right.rival
            && left.rival == right.target
    };
    question_swap(a, b, ka, kb)
        && question_swap(c, d, kc, kd)
        && assignment_swap(a, c, ka, kc)
        && assignment_swap(b, d, kb, kd)
}

struct Selection<'a> {
    subjects: Vec<&'a str>,
    fresh_labels: Vec<Label<'a>>,
    collision_labels: [Label<'a>; 2],
}

impl<'a> Selection<'a> {
    fn new(builder: &Builder<'_>, prior: &Prior) -> Result<Self> {
        let mut fresh_labels = Vec::new();
        for word in LABELS.split_ascii_whitespace() {
            let with_space = format!(" {word}");
            if let [id] = builder.tokenizer.encode(&with_space).as_slice() {
                if prior.fresh_label(*id, word)
                    && builder.tokenizer.decode_bytes(&[*id]) == with_space.as_bytes()
                {
                    fresh_labels.push(Label {
                        id: *id,
                        text: word,
                    });
                }
            }
        }
        require(
            fresh_labels.len() >= 20,
            format!(
                "too few independently screened original-ID labels: {}",
                fresh_labels.len()
            ),
        )?;
        let collision_labels = [
            Label {
                id: COLLISION[0],
                text: "lake",
            },
            Label {
                id: COLLISION[1],
                text: "anchor",
            },
        ];
        for label in collision_labels {
            require(
                builder.tokenizer.encode(&format!(" {}", label.text)) == [label.id]
                    && !prior.train_ids.contains(&label.id)
                    && prior.held_ids.contains(&label.id),
                "collision labels differ from the registered held originals",
            )?;
        }
        Ok(Self {
            subjects: SUBJECTS.split_ascii_whitespace().collect(),
            fresh_labels,
            collision_labels,
        })
    }

    fn choose(&self, settings: Settings, search_key: &str) -> Option<Group<'a>> {
        let subject = |domain| self.subjects[choose(domain, search_key, self.subjects.len())];
        let names = [subject("left"), subject("right"), subject("missing")];
        if names[0] == names[1] || names[0] == names[2] || names[1] == names[2] {
            return None;
        }
        let label = |domain| self.fresh_labels[choose(domain, search_key, self.fresh_labels.len())];
        let labels = if settings.collision {
            if choose("collision-order", search_key, 2) == 0 {
                self.collision_labels
            } else {
                [self.collision_labels[1], self.collision_labels[0]]
            }
        } else {
            [label("first-label"), label("second-label")]
        };
        (labels[0].id != labels[1].id).then(|| Group {
            id: key_id(search_key),
            settings,
            names,
            labels,
        })
    }
}

fn subject_pair(group: &Group<'_>) -> [String; 2] {
    let mut pair = [group.names[0].to_owned(), group.names[1].to_owned()];
    pair.sort_unstable();
    pair
}

fn fresh_group<T>(
    selection: &Selection<'_>,
    prior: &Prior,
    used: &mut BTreeSet<[String; 2]>,
    settings: Settings,
    domain: &str,
    index: usize,
    mut record: impl FnMut(&Group<'_>) -> Result<Option<T>>,
) -> Result<T> {
    for attempt in 0..10_000 {
        let key = format!(
            "{domain}/{}/{}/{}/{}/{}/{index}/{attempt}",
            settings.family.name,
            settings.first_fact_side,
            settings.value_first,
            settings.punctuation,
            settings.collision
        );
        let Some(group) = selection.choose(settings, &key) else {
            continue;
        };
        let pair = subject_pair(&group);
        if prior.subject_pairs.contains(&pair) || used.contains(&pair) {
            continue;
        }
        if let Some(value) = record(&group)? {
            used.insert(pair);
            return Ok(value);
        }
    }
    Err(format!(
        "no canonical group available for {domain}/{} at {index}",
        settings.family.name
    )
    .into())
}

pub(crate) fn build(builder: &Builder<'_>, prior: &Prior) -> Result<Tapes> {
    let mut held = Vec::with_capacity(800);
    let mut absent = Vec::with_capacity(600);
    let mut finite = Vec::with_capacity(16);
    let mut used = BTreeSet::new();
    let selection = Selection::new(builder, prior)?;
    for (family_index, family) in FAMILIES.into_iter().enumerate() {
        for first_fact_side in 0..2 {
            for value_first in [false, true] {
                for punctuation in [false, true] {
                    for collision in [false, true] {
                        let settings = Settings {
                            family,
                            first_fact_side,
                            value_first,
                            punctuation,
                            collision,
                        };
                        // 40 distinct and 10 collision groups per family. The two extra
                        // collision cells have opposite factor levels in all three axes.
                        let aligned = (first_fact_side ^ (family_index & 1))
                            == usize::from(value_first)
                            && value_first == punctuation;
                        let groups = if collision {
                            1 + usize::from(aligned)
                        } else {
                            5
                        };
                        for index in 0..groups {
                            let records = fresh_group(
                                &selection,
                                prior,
                                &mut used,
                                settings,
                                "held",
                                index,
                                |group| {
                                    quartet(
                                        group,
                                        builder,
                                        prior,
                                        index
                                            + first_fact_side
                                            + usize::from(value_first)
                                            + usize::from(punctuation),
                                    )
                                },
                            )?;
                            held.extend(records.held);
                            absent.extend(records.absent);
                        }
                    }
                }
            }
        }
        for (index, kind) in [
            Kind::Duplicate { note_first: true },
            Kind::Duplicate { note_first: false },
            Kind::MalformedSeparator,
            Kind::MalformedMarker,
        ]
        .into_iter()
        .enumerate()
        {
            let settings = Settings {
                family,
                first_fact_side: (family_index + index) % 2,
                value_first: index % 2 == 0,
                punctuation: (family_index + index) % 2 == 0,
                collision: index < 2 && family_index < 2,
            };
            let row = fresh_group(
                &selection,
                prior,
                &mut used,
                settings,
                "finite",
                index,
                |group| group.row(builder, prior, 0, 0, kind),
            )?;
            finite.push(row);
        }
    }
    let audit = audit(&held, &absent, &finite, prior, selection.fresh_labels.len())?;
    Ok(Tapes {
        held,
        absent,
        finite,
        audit,
    })
}

fn audit(
    held: &[Record],
    absent: &[Record],
    finite: &[Record],
    prior: &Prior,
    screened_labels: usize,
) -> Result<Value> {
    require(
        held.len() == 800 && absent.len() == 600 && finite.len() == 16,
        "panel population changed",
    )?;
    let mut groups = BTreeMap::new();
    let mut prefixes = BTreeSet::new();
    let mut users = BTreeSet::new();
    let mut ids = BTreeMap::<_, BTreeSet<_>>::new();
    let mut counts = BTreeMap::new();
    let mut balance = BTreeMap::new();
    let mut exposure = BTreeMap::new();
    let mut terminal = BTreeMap::new();
    for (row, key) in held.iter().chain(absent).chain(finite) {
        let valid_text_key = row.variant != Some("broken_full_assistant_marker");
        require(
            prefixes.insert(&row.prefix)
                && (!valid_text_key || users.insert(normalize(&key.user)))
                && prior.fresh_prompt(&key.question, &key.user, &row.prefix),
            "duplicate or historical prompt",
        )?;
        *counts.entry(row.cohort.as_str()).or_insert(0) += 1;
        if row.cohort != "controlled" {
            continue;
        }
        groups
            .entry(&row.group_id)
            .or_insert_with(Vec::new)
            .push((row, key));
        ids.entry(row.stratum.as_str())
            .or_default()
            .insert(row.target);
        *balance
            .entry(format!(
                "{}/{}/{}/{}/{}",
                key.topic,
                row.stratum,
                key.first_fact_side,
                key.value_before_subject,
                key.punctuation
            ))
            .or_insert(0) += 1;
        *terminal
            .entry(row.expected_second.ok_or("missing next token")?)
            .or_insert(0) += 1;
        let tally = exposure.entry(row.stratum.as_str()).or_insert([0; 6]);
        tally[0] += usize::from(prior.train_ids.contains(&row.target));
        tally[1] += usize::from(prior.held_ids.contains(&row.target));
        tally[2] += usize::from(prior.marker_train_ids.contains(&row.target));
        tally[3] += usize::from(prior.donor_opening_ids.contains(&row.target));
        tally[4] += usize::from(prior.donor_raw_vocabulary.contains(&row.target));
        tally[5] += usize::from(prior.donor_colon_ids.contains(&row.target));
    }
    require(
        groups.len() == 200
            && groups.values().all(|group| group.len() == 4)
            && counts.get("same_id_wrong_address") == Some(&8)
            && counts.get("malformed") == Some(&8)
            && counts.get("value_omitted") == Some(&400)
            && counts.get("subject_unbound") == Some(&200)
            && terminal.get(&EOS) == Some(&400),
        "panel strata or terminal counts changed",
    )?;
    require(
        exposure
            .get("distinct_held_words")
            .is_some_and(|values| values[..4] == [0; 4])
            && exposure
                .get("same_code_1808_3984")
                .is_some_and(|values| values[..4] == [0, 160, 0, 0])
            && ids
                .get("same_code_1808_3984")
                .is_some_and(|values| values.iter().copied().eq(COLLISION)),
        "supervised opening exposure changed",
    )?;
    for (row, _) in absent {
        let matching = groups
            .get(&row.group_id)
            .ok_or("absence row without group")?;
        let (_, key) = matching
            .iter()
            .find(|(candidate, _)| {
                candidate.assignment == row.assignment && candidate.side == row.side
            })
            .ok_or("absence row without matched assignment/question")?;
        let actual = absent
            .iter()
            .find(|(candidate, _)| candidate.pair_id == row.pair_id && candidate.side == row.side)
            .ok_or("absence row key lost")?;
        require(
            row.gold_pos.is_none()
                && row.expected_second.is_none()
                && ((row.cohort == "value_omitted" && key.question == actual.1.question)
                    || (row.cohort == "subject_unbound"
                        && key.context == actual.1.context
                        && !actual
                            .1
                            .context
                            .split(|ch: char| !ch.is_alphanumeric())
                            .any(|word| word == actual.1.requested_subject))),
            "absence changed the wrong factor",
        )?;
    }
    let [shortest, longest] = [
        held.iter().map(|(row, _)| row.prefix.len()).min(),
        held.iter().map(|(row, _)| row.prefix.len()).max(),
    ];
    Ok(json!({
        "status": "PASS", "controlled_groups": groups.len(), "cohort_rows": counts, "exact_id_sets": ids,
        "controlled_rows_by_family_stratum_first_fact_value_first_punctuation": balance,
        "expected_second_id_counts": terminal, "eligible_independently_screened_labels": screened_labels,
        "controlled_prefix_length_range": [shortest, longest], "previous_exact_prefixes_screened": prior.prefixes.len(),
        "previous_question_types_screened": prior.questions.len(), "previous_subject_pairs_screened": prior.subject_pairs.len(),
        "donor_exact_prefixes_screened": prior.donor_prefixes.len(),
        "target_exposure_fields": ["4315_controlled_train", "4315_held_or_human", "4314_fallback_train",
            "donor_genuine_immediate_or_lexical_opening", "donor_raw_vocabulary", "donor_all_colon_targets"],
        "target_exposure_counts_by_stratum": exposure,
        "finite_controls_in_primary_denominator": false, "model_executed": false,
    }))
}

pub(crate) fn serialized<T: Serialize>(values: impl IntoIterator<Item = T>) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for value in values {
        serde_json::to_writer(&mut bytes, &value)?;
        bytes.push(b'\n');
    }
    Ok(bytes)
}
