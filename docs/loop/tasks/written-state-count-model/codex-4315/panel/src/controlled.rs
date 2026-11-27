use crate::*;
use inputs::{Donor, Prior};

pub(crate) const GROUPS_PER_FAMILY: usize = 50;
pub(crate) const GROUPS: usize = 200;
const SUBJECTS: &str = "Abel Abram Adele Agnes Aidan Alan Alina Alma Alvin Amos Anita Ansel Anton Arlo Arnold Astrid Avery Basil Berta Boris Bran Bruce Bruno Byron Celia Cora Cyrus Dante Darcy Daria Declan Diego Dorian Duncan Edna Edwin Elaine Elise Elsa Elton Enid Erin Ethan Ethel Evan Ezra Flora Floyd Freya Frida Gemma Gina Gloria Greta Hugo Ida Ilan Ines Ivan Ivor Jade Jake Jasper Jessa Jonah Jude Julius Kira Lara Leon Leroy Levi Lina Lionel Logan Lola Lorna Luca Lyle Maeve Maia Marco Margot Marta Matilda Mina Milo Mona Nadia Nadia Nando Natalie Nell Niko Nola Octavia Odin Omar Orion Orla Otis Otto Petra Quinn Ramona Remy Rhoda Roman Rory Rowan Rudy Sacha Sadie Sasha Saul Seth Sheila Silas Sonia Theo Toby Tony Una Ursula Val Valeria Vance Veda Vince Viola Willa Wilma Xena Yara Yona Yuri Zane Zara Zelda";
const VALUES: &str = "acorn amber anchor angle antler apron arcade arch arrow atlas avenue badge bagel bamboo banana banner barley barrel basin basket beacon bead beans bear beet bell bench berry birch blade blanket bloom blossom bolt bone boot bowl branch brass breeze brick bronze broom brush bucket bulb bundle button cabin cable cactus cage cake camel canoe canvas canyon cape caribou carpet carrot cart cedar cellar chain chalk charm cherry chest chick chili chime chip cider circus clam clay cliff cloak clock cloud clover coal coast cocoa comet copper coral cork corn cove crab crane creek crown crystal cube curtain daisy dart dawn deer delta den dew diamond dock dove drum dune dusk dust eagle earth echo elm enamel engine fern fig fin fir flame flax fleece flint flour flute fog fork fox frame frost fruit funnel fur garden garlic gauze gem ginger glass globe glue goat goblet granite grape grass gravel grove guava gull gum hail halo hammer harp hay herb hide hill hive holly honey hook horn hour ink iron ivory jade jam jar jasmine jelly jewel kiln kite kiwi knot lace lake lamp lance lantern latch laurel leaf leek lemon lens lichen lilac lily lime linen lion lizard llama loaf lodge lotus lye marble marsh mast meadow melon mint mist moss moth mouse muffin mule nectar nest nickel oak oasis oat olive onyx opal orange orchid ore otter owl paddle palm panda pearl pebble pecan pedal pepper perch petal piano pickle pier pigeon pine plank plum plume pocket pond poppy porch prism pulse pumpkin quartz quill quilt rabbit rain raisin rake ramp raven reed reef ribbon ridge rifle robin robot root rope rose rosemary rug sage sail salt sand satin sauce scallop scale scarf seed shale shawl shell shield shore shrub silk skate slate sled slope slug smoke snail soap sock soda soil soot spade sparrow spear spice sponge spool spruce squash squirrel staff steam steel stem stitch stump sugar swan syrup tack tape tea teal thorn thyme tick tile timber tin toast tomato torch tower trout tulip tuna turf twig urn vase velvet vine violet visor walnut wand wasp wax wheat wheel wick willow wool wreath wrench yam zinc";

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Family {
    Museum,
    Radio,
    Bakery,
    Workshop,
    Greenhouse,
    Archive,
    Regatta,
    Courtyard,
}

pub(crate) const TRAIN_FAMILIES: [Family; 4] = [
    Family::Museum,
    Family::Radio,
    Family::Bakery,
    Family::Workshop,
];
pub(crate) const HELD_FAMILIES: [Family; 4] = [
    Family::Greenhouse,
    Family::Archive,
    Family::Regatta,
    Family::Courtyard,
];

impl Family {
    pub fn name(self) -> &'static str {
        match self {
            Self::Museum => "museum",
            Self::Radio => "radio",
            Self::Bakery => "bakery",
            Self::Workshop => "workshop",
            Self::Greenhouse => "greenhouse",
            Self::Archive => "archive",
            Self::Regatta => "regatta",
            Self::Courtyard => "courtyard",
        }
    }

    fn lead(self) -> &'static str {
        match self {
            Self::Museum => "The curator registered two volunteer badges.",
            Self::Radio => "The radio producer has two substitute hosts.",
            Self::Bakery => "The bakery counters carry hand-lettered nicknames.",
            Self::Workshop => "The repair workshop has personalized bench signs.",
            Self::Greenhouse => "In the annual greenhouse inventory, each bed is indexed by a tag.",
            Self::Archive => "A key to the archive shelf list links each researcher to a shelf label.",
            Self::Regatta => "The committee used temporary pennant names to distinguish the boats at the regatta.",
            Self::Courtyard => "A community courtyard map uses one-word inscriptions to distinguish donated tiles.",
        }
    }

    fn fact(self, subject: &str, answer: Option<&str>, value_first: bool) -> String {
        let Some(value) = answer else {
            return match self {
                Self::Greenhouse => format!("{subject} tended a bed in the morning."),
                Self::Archive => format!("{subject} visited the archive before noon."),
                Self::Regatta => format!("{subject} attended the regatta briefing."),
                Self::Courtyard => format!("{subject} attended the courtyard opening."),
                _ => format!("{subject} attended the event."),
            };
        };
        match (self, value_first) {
            (Self::Museum, false) => format!("{subject} wears the badge labeled {value}."),
            (Self::Museum, true) => {
                format!("The badge labeled {value} belongs to volunteer {subject}.")
            }
            (Self::Radio, false) => format!("{subject} broadcasts using the pseudonym {value}."),
            (Self::Radio, true) => format!("The pseudonym {value} identifies the host {subject}."),
            (Self::Bakery, false) => format!("At {subject}'s counter the nickname is {value}."),
            (Self::Bakery, true) => {
                format!("The nickname {value} identifies the counter operated by {subject}.")
            }
            (Self::Workshop, false) => {
                format!("{subject} works at the bench whose sign reads {value}.")
            }
            (Self::Workshop, true) => {
                format!("The sign reading {value} identifies the bench used by {subject}.")
            }
            (Self::Greenhouse, false) => {
                format!("The bed tended by {subject} appears in the index as {value}.")
            }
            (Self::Greenhouse, true) => {
                format!("Listed under {value} is the bed tended by {subject}.")
            }
            (Self::Archive, false) => {
                format!("A researcher named {subject} files items on the shelf designated {value}.")
            }
            (Self::Archive, true) => {
                format!("Items on the shelf designated {value} were filed by {subject}.")
            }
            (Self::Regatta, false) => {
                format!("The boat steered by {subject} flies a pennant named {value}.")
            }
            (Self::Regatta, true) => {
                format!("Flying the pennant named {value} is the boat steered by {subject}.")
            }
            (Self::Courtyard, false) => {
                format!("The tile donated by {subject} carries the inscription {value}.")
            }
            (Self::Courtyard, true) => {
                format!("Inscribed with {value} is the tile donated by {subject}.")
            }
        }
    }

    pub fn context(
        self,
        names: [&str; 2],
        labels: [Option<&str>; 2],
        first_side: usize,
        value_first: bool,
    ) -> String {
        let fact = |side| self.fact(names[side], labels[side], value_first);
        format!(
            "{} {} {}",
            self.lead(),
            fact(first_side),
            fact(1 - first_side)
        )
    }

    pub fn question(self, subject: &str) -> String {
        match self {
            Self::Museum => format!("Tell me the badge label worn by the volunteer {subject}."),
            Self::Radio => format!("On air, what pseudonym does the substitute host {subject} use?"),
            Self::Bakery => format!("Supply the bakery nickname displayed at the counter where {subject} works."),
            Self::Workshop => format!("Transcribe the sign fixed to the repair bench used by {subject}."),
            Self::Greenhouse => format!("Which tag would locate the bed attributed to {subject} in the greenhouse inventory?"),
            Self::Archive => format!("Look up the shelf designation under which {subject} filed items in the archive."),
            Self::Regatta => format!("For the regatta boat steered by {subject}, how was its pennant named?"),
            Self::Courtyard => format!("The community courtyard map places what inscription on the tile from {subject}?"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Source {
    pub family: Family,
    pub subjects: [String; 2],
    pub values_by_subject: [String; 2],
    pub first_fact_side: usize,
    pub value_before_subject: bool,
    pub requested_subject: String,
    pub absence_kind: Option<String>,
    pub matched_pair_id: String,
    pub matched_side: usize,
    pub diagnostic_target: Option<String>,
    pub diagnostic_rival: Option<String>,
}

#[derive(Clone, Copy)]
struct Label<'a> {
    text: &'a str,
    id: u32,
}

fn pool<'a>(builder: &Builder<'_>) -> Vec<Label<'a>> {
    let mut ids = HashSet::new();
    VALUES
        .split_whitespace()
        .filter_map(|text| {
            let encoded = builder.tokenizer.encode(&format!(" {text}"));
            match *encoded.as_slice() {
                [id] if id > SPACE && ids.insert(id) => Some(Label { text, id }),
                _ => None,
            }
        })
        .collect()
}

fn select<const N: usize>(domain: &str, key: &str, count: usize) -> [usize; N] {
    let bytes = hash(domain, key);
    let mut indexes = [0; N];
    for i in 0..N {
        let j = i * 2;
        let mut index = usize::from(u16::from_le_bytes([bytes[j], bytes[j + 1]])) % (count - i);
        let mut excluded = indexes[..i].to_vec();
        excluded.sort_unstable();
        for used in excluded {
            index += usize::from(index >= used);
        }
        indexes[i] = index;
    }
    indexes
}

fn subject_key(subjects: [&str; 2]) -> [String; 2] {
    let mut key = subjects.map(str::to_owned);
    key.sort_unstable();
    key
}

pub(crate) fn train(builder: &Builder<'_>, prior: &Prior) -> Result<(Tape, Tape)> {
    let mut labels = pool(builder);
    labels.sort_by_cached_key(|label| hash("label-split", label.text));
    let cutoff = labels.len().div_ceil(2);
    let [train, train_absent] = generate(
        builder,
        prior,
        "train",
        TRAIN_FAMILIES,
        &labels[..cutoff],
        prior.subject_pairs.clone(),
    )?;
    Ok((train, train_absent))
}

pub(crate) fn held(
    builder: &Builder<'_>,
    prior: &Prior,
    donor: &Donor,
    train: &Tape,
) -> Result<(Tape, Tape)> {
    let train_ids: HashSet<_> = train
        .rows
        .iter()
        .flat_map(|r| [r.target, r.rival, r.raw_target, r.raw_rival])
        .collect();
    let train_words: HashSet<_> = train
        .keys
        .iter()
        .filter_map(|key| key.reply.as_deref())
        .collect();
    let labels: Vec<_> = pool(builder)
        .into_iter()
        .filter(|label| {
            !train_ids.contains(&label.id)
                && !prior.fallback_contrast.contains(&label.id)
                && !train_words.contains(label.text)
                && !donor.immediate.contains(&label.id)
                && !donor.lexical.contains(&label.id)
                && !donor.first_words.contains(label.text)
        })
        .collect();
    let mut used = prior.subject_pairs.clone();
    for key in &train.keys {
        let source: Source = serde_json::from_value(key.source.clone())?;
        used.insert(subject_key(source.subjects.each_ref().map(String::as_str)));
    }
    let [held, absent] = generate(builder, prior, "held", HELD_FAMILIES, &labels, used)?;
    Ok((held, absent))
}

struct Group<'a> {
    id: String,
    family: Family,
    subjects: [&'a str; 2],
    third_subject: &'a str,
    labels: [Label<'a>; 2],
    first_fact_side: usize,
    value_before_subject: bool,
    control_side: usize,
}

impl Group<'_> {
    fn assigned(&self, assignment: usize) -> [Label<'_>; 2] {
        if assignment == 0 {
            self.labels
        } else {
            [self.labels[1], self.labels[0]]
        }
    }

    fn pair_id(&self, assignment: usize) -> String {
        id("controlled-pair", &format!("{}:{assignment}", self.id))
    }

    fn source(&self, assignment: usize, side: usize) -> Source {
        Source {
            family: self.family,
            subjects: self.subjects.map(str::to_owned),
            values_by_subject: self.assigned(assignment).map(|label| label.text.to_owned()),
            first_fact_side: self.first_fact_side,
            value_before_subject: self.value_before_subject,
            requested_subject: self.subjects[side].to_owned(),
            absence_kind: None,
            matched_pair_id: self.pair_id(assignment),
            matched_side: side,
            diagnostic_target: None,
            diagnostic_rival: None,
        }
    }

    fn render_context(&self, labels: [Option<&str>; 2]) -> String {
        self.family.context(
            self.subjects,
            labels,
            self.first_fact_side,
            self.value_before_subject,
        )
    }

    fn controlled_cases(
        &self,
        builder: &Builder<'_>,
        prior: &Prior,
    ) -> Result<Option<Vec<[Case; 2]>>> {
        let mut cases = Vec::with_capacity(2);
        for assignment in 0..2 {
            let labels = self.assigned(assignment);
            let context = self.render_context(labels.map(|label| Some(label.text)));
            let [left, right] = self.subjects.map(|subject| self.family.question(subject));
            let Some(left) = builder.case(&context, &left, Some(labels[0].text))? else {
                return Ok(None);
            };
            let Some(right) = builder.case(&context, &right, Some(labels[1].text))? else {
                return Ok(None);
            };
            let pair = [left, right];
            if pair[0].prefix.len() != pair[1].prefix.len()
                || pair.iter().enumerate().any(|(side, case)| {
                    !prior.allows_case(case)
                        || case.gold != [labels[side].id]
                        || labels
                            .iter()
                            .any(|label| case.unique_prefix_position(label.id).is_none())
                })
            {
                return Ok(None);
            }
            cases.push(pair);
        }
        for (before, after) in cases[0].iter().zip(&cases[1]) {
            let a = &before.prefix;
            let b = &after.prefix;
            let changes: Vec<_> = a
                .iter()
                .zip(b)
                .enumerate()
                .filter_map(|(idx, (a, b))| (a != b).then_some(idx))
                .collect();
            let positions: HashSet<_> = self
                .labels
                .iter()
                .filter_map(|label| before.unique_prefix_position(label.id))
                .collect();
            if a.len() != b.len()
                || changes.len() != 2
                || changes.iter().any(|pos| !positions.contains(pos))
            {
                return Ok(None);
            }
        }
        Ok(Some(cases))
    }

    fn controls(
        &self,
        builder: &Builder<'_>,
        prior: &Prior,
        split: &str,
        control: &mut Tape,
    ) -> Result<bool> {
        let requests: &[(usize, bool)] = if split == "train" {
            &[(self.control_side, false)]
        } else {
            &[(0, true), (1, true), (self.control_side, false)]
        };
        let side = self.control_side;
        let mut pending = Vec::new();
        for &(assignment, omit_value) in requests {
            let labels = self.assigned(assignment);
            let mut values = labels.map(|label| Some(label.text));
            if omit_value {
                values[side] = None;
            }
            let source_context = self.render_context(values);
            let subject = if omit_value {
                self.subjects[side]
            } else {
                self.third_subject
            };
            let question = self.family.question(subject);
            let Some(case) = builder.case(&source_context, &question, None)? else {
                return Ok(false);
            };
            let Some(rival_pos) = case.unique_prefix_position(labels[1 - side].id) else {
                return Ok(false);
            };
            if !prior.allows_case(&case)
                || (omit_value && case.prefix.contains(&labels[side].id))
                || (!omit_value && case.unique_prefix_position(labels[side].id).is_none())
            {
                return Ok(false);
            }
            let kind = if omit_value {
                "value_omitted"
            } else {
                "subject_unbound"
            };
            let pair_id = id("absent-pair", &format!("{}:{assignment}:{kind}", self.id));
            let mut source = self.source(assignment, side);
            source.requested_subject = subject.to_owned();
            source.absence_kind = Some(kind.to_owned());
            source.diagnostic_target = Some(labels[side].text.to_owned());
            source.diagnostic_rival = Some(labels[1 - side].text.to_owned());
            pending.push((
                case,
                source,
                label_ids(labels[side]),
                label_ids(labels[1 - side]),
                rival_pos,
                kind,
                pair_id,
                assignment,
            ));
        }
        for (case, source, label, rival, rival_pos, kind, pair_id, assignment) in pending {
            control.push(
                Entry {
                    group: &self.id,
                    pair: &pair_id,
                    cohort: kind,
                    topic: self.family.name(),
                    assignment,
                    side,
                    equal_length: false,
                },
                case,
                label,
                rival,
                Annotation {
                    gold_pos: None,
                    rival_pos: Some(rival_pos),
                    status: Status::Absent,
                },
                serde_json::to_value(source)?,
            );
        }
        Ok(true)
    }
}

fn label_ids(label: Label<'_>) -> (u32, u32) {
    (label.id, label.id)
}

fn generate(
    builder: &Builder<'_>,
    prior: &Prior,
    split: &str,
    families: [Family; 4],
    labels: &[Label<'_>],
    mut used: HashSet<[String; 2]>,
) -> Result<[Tape; 2]> {
    require(
        labels.len() >= 30,
        format!("only {} eligible {split} answer IDs", labels.len()),
    )?;
    let mut names: Vec<_> = SUBJECTS.split_whitespace().collect();
    names.sort_unstable();
    names.dedup();
    let mut rows = Tape::default();
    let mut control = Tape::default();
    let mut seen = HashSet::new();
    for (family_index, family) in families.into_iter().enumerate() {
        let mut produced = 0;
        for candidate in 0..10_000 {
            let key = format!("{split}:{}:{candidate}", family.name());
            let [left, right, third] = select::<3>("subjects", &key, names.len()).map(|i| names[i]);
            let subjects = [left, right];
            let subject_key = subject_key(subjects);
            if used.contains(&subject_key) {
                continue;
            }
            let group = Group {
                id: id("controlled-group", &key),
                family,
                subjects,
                third_subject: third,
                labels: select::<2>("answers", &key, labels.len()).map(|i| labels[i]),
                first_fact_side: (produced + family_index / 2) % 2,
                value_before_subject: (produced / 2 + family_index) % 2 == 0,
                control_side: (produced / 2 + family_index) % 2,
            };
            let Some(cases) = group.controlled_cases(builder, prior)? else {
                continue;
            };
            if cases
                .iter()
                .flatten()
                .any(|case| seen.contains(&case.prefix))
            {
                continue;
            }
            let mut group_controls = Tape::default();
            if !group.controls(builder, prior, split, &mut group_controls)?
                || group_controls
                    .rows
                    .iter()
                    .any(|row| seen.contains(&row.prefix))
            {
                continue;
            }
            for (assignment, pair) in cases.into_iter().enumerate() {
                let labels = group.assigned(assignment);
                let pair_id = group.pair_id(assignment);
                for (side, case) in pair.into_iter().enumerate() {
                    let gold_pos = case.unique_prefix_position(labels[side].id);
                    let rival_pos = case.unique_prefix_position(labels[1 - side].id);
                    seen.insert(case.prefix.clone());
                    rows.push(
                        Entry {
                            group: &group.id,
                            pair: &pair_id,
                            cohort: "controlled",
                            topic: family.name(),
                            assignment,
                            side,
                            equal_length: true,
                        },
                        case,
                        label_ids(labels[side]),
                        label_ids(labels[1 - side]),
                        Annotation {
                            gold_pos,
                            rival_pos,
                            status: Status::Unique,
                        },
                        serde_json::to_value(group.source(assignment, side))?,
                    );
                }
            }
            seen.extend(group_controls.rows.iter().map(|row| row.prefix.clone()));
            control.rows.extend(group_controls.rows);
            control.keys.extend(group_controls.keys);
            used.insert(subject_key);
            produced += 1;
            if produced == GROUPS_PER_FAMILY {
                break;
            }
        }
        require(
            produced == GROUPS_PER_FAMILY,
            format!("incomplete {split} controlled family {}", family.name()),
        )?;
    }
    Ok([rows, control])
}
