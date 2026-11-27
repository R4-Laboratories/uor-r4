pub(crate) struct EndingFamily {
    pub name: &'static str,
    pub variants: &'static [[&'static str; 2]],
}

pub(crate) const TRAIN: &[EndingFamily] = &[
    EndingFamily {
        name: "train_direct_commands",
        variants: &[
            [
                "Return the word alone. Do not add a period.",
                "Return the word. Do not forget to put a period after it.",
            ],
            [
                "Copy the word. Stop there with no punctuation.",
                "Copy the word. Put a period after it and stop.",
            ],
        ],
    },
    EndingFamily {
        name: "train_with_without",
        variants: &[
            [
                "Reply with the word without a final period.",
                "Reply with the word with a final period.",
            ],
            [
                "Give the word with no mark after it.",
                "Give the word with a period after it and no other mark.",
            ],
        ],
    },
    EndingFamily {
        name: "train_sequential",
        variants: &[
            [
                "First give the word. Then end the reply.",
                "First give the word. Then put a period at the end.",
            ],
            [
                "Write the word first. Leave the rest empty.",
                "Write the word first. Follow it with a period.",
            ],
        ],
    },
    EndingFamily {
        name: "train_yes_no",
        variants: &[
            [
                "Use the word. Period at the end: no.",
                "Use the word. Period at the end: yes.",
            ],
            [
                "Copy the word. Final punctuation: none.",
                "Copy the word. Final punctuation: period.",
            ],
        ],
    },
    EndingFamily {
        name: "train_banned_required",
        variants: &[
            [
                "Provide the word. Extra marks are forbidden.",
                "Provide the word. An ending period is required.",
            ],
            [
                "Write one word. You must not put a period after the word.",
                "Write one word. You must put a period after the word.",
            ],
        ],
    },
    EndingFamily {
        name: "train_format",
        variants: &[
            [
                "Use the plain word as the entire reply.",
                "Use the word plus a period as the entire reply.",
            ],
            [
                "The format is a word only, with nothing following it.",
                "The format is a word followed by a period, with nothing following that.",
            ],
        ],
    },
];

pub(crate) const HELD: &[EndingFamily] = &[
    EndingFamily {
        name: "held_contrastive_correction",
        variants: &[
            ["Use just a word, not a word and a period.", "Use a word and a period, not just a word."],
            ["Give just the word, not the word plus a period.", "Give the word plus a period, not just the word."],
        ],
    },
    EndingFamily {
        name: "held_final_character",
        variants: &[
            ["For this field the last character must be a letter, so end with the word.", "For this field the last character must be a period, so put one after the word."],
            ["The final character of your reply should belong to the word.", "The final character of your reply should be a period after the word."],
        ],
    },
    EndingFamily {
        name: "held_referenced_rule",
        variants: &[
            ["A period is forbidden and the word is required. Include only what is required.", "The word is required and so is a period. Include both in that order."],
            ["The word is allowed and a period is banned. Preserve the allowed part.", "The word is required and an ending period is allowed. Use that allowance."],
        ],
    },
    EndingFamily {
        name: "held_scope_exception",
        variants: &[
            ["Copy the word exactly. The permission to add a period applies only to other replies.", "Copy the word exactly, with one exception for this reply. Add a period."],
            ["Keep only the copied word even if other replies have a period.", "Keep only the copied word, except that this reply must also carry a period."],
        ],
    },
];

pub(crate) const FINITE: &[EndingFamily] = &[
    EndingFamily {
        name: "finite_format_label",
        variants: &[["One word, nothing else.", "One word, then a period."]],
    },
    EndingFamily {
        name: "finite_destination_constraint",
        variants: &[[
            "The box accepts the word by itself.",
            "The box accepts the word only if a period follows it.",
        ]],
    },
];

pub(crate) struct FactFamily {
    pub name: &'static str,
    pub background: &'static str,
    pub facts: [&'static str; 2],
    pub question: &'static str,
}

pub(crate) const TRAIN_FACTS: &[FactFamily] = &[
    FactFamily {
        name: "visitor",
        background: "The front desk records a pass word for each visitor.",
        facts: [
            "The visitor {person} gives the word {word}.",
            "The word {word} is recorded under the visitor {person}.",
        ],
        question: "Can you give me the pass word of the visitor {person}?",
    },
    FactFamily {
        name: "workshop",
        background: "Every workshop reservation has a label in the daily register.",
        facts: [
            "The reservation from {person} has the label {word}.",
            "The label {word} identifies the reservation from {person}.",
        ],
        question: "In the register, which label appears for the reservation from {person}?",
    },
    FactFamily {
        name: "locker",
        background: "The pool staff use a different word for each occupied locker.",
        facts: [
            "The locker used by {person} has been assigned {word}.",
            "A card reading {word} hangs on the locker of {person}.",
        ],
        question: "For the locker of {person}, what word was assigned?",
    },
    FactFamily {
        name: "parcel",
        background: "The dispatch log records a route word for each recipient.",
        facts: [
            "The parcel going to {person} is routed under {word}.",
            "The route word {word} is on the parcel for {person}.",
        ],
        question: "What should I read as the route word on the parcel for {person}?",
    },
];

pub(crate) const HELD_FACTS: &[FactFamily] = &[
    FactFamily {
        name: "costume",
        background: "The theater writes a tag word on each costume bag.",
        facts: [
            "The bag collected by {person} has a tag reading {word}.",
            "The tag {word} was put on the bag collected by {person}.",
        ],
        question: "If {person} shows me a costume bag from this list, what does its tag say?",
    },
    FactFamily {
        name: "seed_trial",
        background: "The crop trial ledger gives each planting row a short code.",
        facts: [
            "The planting row measured by {person} goes by {word}.",
            "The code {word} belongs to the planting row measured by {person}.",
        ],
        question: "Under what code would I find the measurement taken by {person}?",
    },
    FactFamily {
        name: "radio_check",
        background: "Each radio check in the log is filed under a signal word.",
        facts: [
            "The radio check from {person} is filed as {word}.",
            "The signal {word} identifies the radio check from {person}.",
        ],
        question:
            "Looking only at the list above, what was the signal used for the check from {person}?",
    },
    FactFamily {
        name: "gallery",
        background: "The gallery gives each crate a word for its return trip.",
        facts: [
            "The crate sealed by {person} carries the word {word}.",
            "A slip saying {word} sits on the crate sealed by {person}.",
        ],
        question: "The seal names {person}; what word should I read from the matching crate?",
    },
];

pub(crate) const FINITE_FACTS: &[FactFamily] = &[FactFamily {
    name: "sample_trial",
    background: "The repair shop labels each tray so it can be collected after closing.",
    facts: [
        "The tray tested by {person} carries the label {word}.",
        "The label {word} sits by the tray tested by {person}.",
    ],
    question: "Which word identifies the tray that {person} tested?",
}];

pub(crate) const TRAIN_LABELS: &str = "apple river paper water cloud train snow garden window road stone bread bell orange button fish";
pub(crate) const HELD_LABELS: &str = "lake anchor flag candy tiger bridge mirror lemon moon glass";
pub(crate) const FINITE_LABELS: &str = "knife clock smoke";

const GIVEN: &[&str] = &[
    "Ada", "Alice", "Amir", "Anders", "Angus", "Anne", "Audrey", "Bea", "Ben", "Carmen", "Daphne",
    "Darius", "Della", "Eli", "Emil", "Esther", "Felix", "Fern", "Finn", "Gavin", "Grace", "Hope",
    "Irene", "Isaac", "Ivy", "James", "Joan", "Joy", "Kay", "Kurt", "Lena", "Luke", "Mara", "Mark",
    "Maya", "Nina", "Noah", "Pia", "Reed", "Rosa", "Sam", "Tess", "Zoe",
];
const SURNAMES: &[&str] = &[
    "Abbott", "Bennett", "Carver", "Dawson", "Evans", "Faber", "Fisher", "Garner", "Gates",
    "Hayes", "Holt", "Irwin", "Jameson", "Keller", "Lawson", "Mills", "Nash", "Oakley", "Porter",
    "Ramos",
];

pub(crate) fn person(index: usize) -> String {
    format!(
        "{} {}",
        GIVEN[index % GIVEN.len()],
        SURNAMES[index / GIVEN.len()]
    )
}
