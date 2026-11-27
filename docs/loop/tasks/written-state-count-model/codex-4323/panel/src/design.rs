//! Authored templates for a crossed instruction-scope experiment.
pub const PAIRS_PER_FAMILY: usize = 32;
pub const LABELS_PER_SPLIT: usize = 8;

pub struct Family {
    pub name: &'static str,
    /// The one-word and two-word clauses, in that order.
    pub counts: [&'static str; 2],
    /// The selector frames for an operative first or second clause.
    /// A direct request has neither selector nor a rival clause.
    pub selectors: Option<[&'static str; 2]>,
    pub duplicate_words: bool,
    pub quoted: bool,
}

impl Family {
    pub fn directives(&self, cross: usize) -> [String; 2] {
        let first = cross % 2;
        let change_selector = cross >= 2;
        match self.selectors {
            Some(selectors) => std::array::from_fn(|side| {
                let selector = if change_selector { first ^ side } else { first };
                let clause = if change_selector { first } else { first ^ side };
                selectors[selector]
                    .replace("{a}", self.counts[clause])
                    .replace("{b}", self.counts[clause ^ 1])
            }),
            None => self.counts.map(str::to_owned),
        }
    }
}

pub struct Split {
    pub name: &'static str,
    pub fact: [&'static str; 2],
    pub question: &'static str,
    pub families: [Family; 8],
}

pub const SPLITS: [Split; 3] = [
    Split {
        name: "train",
        fact: [
            "In crate {name}, slot {role} holds {word}.",
            "In crate {name}, slot {role} holds {word};",
        ],
        question: "What should crate {name} print?",
        families: [
            Family {
                name: "archive-adopt",
                counts: ["give one word", "give two words"],
                selectors: Some(["Adopt '{a}'; archive '{b}'", "Archive '{a}'; adopt '{b}'"]),
                duplicate_words: false,
                quoted: true,
            },
            Family {
                name: "training-slash",
                counts: ["output a single word", "output a pair of words"],
                selectors: Some([
                    "Use the rule to the left of the divider: '{a}' / '{b}'",
                    "Use the rule to the right of the divider: '{a}' / '{b}'",
                ]),
                duplicate_words: false,
                quoted: true,
            },
            Family {
                name: "approved-draft",
                counts: ["stop before word two", "stop after word two"],
                selectors: Some([
                    "The approved draft says '{a}' and the old draft says '{b}'",
                    "The old draft says '{a}' and the approved draft says '{b}'",
                ]),
                duplicate_words: false,
                quoted: true,
            },
            Family {
                name: "second-slip",
                counts: ["leave the next word out", "let the next word through"],
                selectors: Some([
                    "Between the notes '{a}' and '{b}', enforce the former",
                    "Between the notes '{a}' and '{b}', enforce the latter",
                ]),
                duplicate_words: false,
                quoted: true,
            },
            Family {
                name: "training-repeat",
                counts: ["one word, one word only", "two words, two words only"],
                selectors: Some([
                    "Apply '{a}' and dismiss '{b}'",
                    "Dismiss '{a}' and apply '{b}'",
                ]),
                duplicate_words: true,
                quoted: true,
            },
            Family {
                name: "clerk-clock",
                counts: [
                    "close the reply after the opening word",
                    "close the reply after the following word",
                ],
                selectors: Some([
                    "The clerk marked '{a}' current and '{b}' expired",
                    "The clerk marked '{a}' expired and '{b}' current",
                ]),
                duplicate_words: false,
                quoted: true,
            },
            Family {
                name: "unquoted-vs",
                counts: ["copy the lead word only", "copy the leading two words"],
                selectors: Some([
                    "Please {a} instead of following the suggestion to {b}",
                    "The suggestion to {a} can wait; please {b}",
                ]),
                duplicate_words: false,
                quoted: false,
            },
            Family {
                name: "direct-train",
                counts: [
                    "Output just the opening word",
                    "Output the opening word and its successor",
                ],
                selectors: None,
                duplicate_words: false,
                quoted: false,
            },
        ],
    },
    Split {
        name: "dev",
        fact: [
            "Box {name} lists {word} at position {role}.",
            "Box {name} lists {word} at position {role};",
        ],
        question: "What goes in the reply for box {name}?",
        families: [
            Family {
                name: "highlighted",
                counts: ["end before saying item two", "end once item two is said"],
                selectors: Some([
                    "The order in bold is yours: **{a}** or '{b}'",
                    "The order in bold is yours: '{a}' or **{b}**",
                ]),
                duplicate_words: false,
                quoted: true,
            },
            Family {
                name: "test-then-live",
                counts: ["read only the initial word", "read both opening words"],
                selectors: Some([
                    "Live cue: '{a}'. Rehearsal cue: '{b}'",
                    "Rehearsal cue: '{a}'. Live cue: '{b}'",
                ]),
                duplicate_words: false,
                quoted: true,
            },
            Family {
                name: "square-bracket",
                counts: [
                    "give the lead word alone",
                    "give the lead and the next word",
                ],
                selectors: Some([
                    "The setting inside brackets is active: [{a}] ({b})",
                    "The setting inside brackets is active: ({a}) [{b}]",
                ]),
                duplicate_words: false,
                quoted: false,
            },
            Family {
                name: "reject-veto",
                counts: ["take a one word excerpt", "take a two word excerpt"],
                selectors: Some([
                    "Please follow '{a}'; '{b}' was vetoed",
                    "The note proposes '{a}', but '{b}' remains in effect",
                ]),
                duplicate_words: false,
                quoted: true,
            },
            Family {
                name: "dev-repeat",
                counts: [
                    "read one word, and one word is enough",
                    "read two words, and two words are enough",
                ],
                selectors: Some([
                    "Required: '{a}'. Example only: '{b}'",
                    "Example only: '{a}'. Required: '{b}'",
                ]),
                duplicate_words: true,
                quoted: true,
            },
            Family {
                name: "scribe-underlined",
                counts: [
                    "skip the second entry",
                    "include the second entry and then finish",
                ],
                selectors: Some([
                    "Copy according to the underlined rule: _{a}_ versus '{b}'",
                    "Copy according to the underlined rule: '{a}' versus _{b}_",
                ]),
                duplicate_words: false,
                quoted: true,
            },
            Family {
                name: "unquoted-replace",
                counts: [
                    "reproduce the initial word alone",
                    "reproduce two words from the start",
                ],
                selectors: Some([
                    "You must {a}; drop the earlier request to {b}",
                    "Drop the earlier request to {a}; you must {b}",
                ]),
                duplicate_words: false,
                quoted: false,
            },
            Family {
                name: "direct-dev",
                counts: [
                    "Fit your entire answer into one word",
                    "Fit your entire answer into two words",
                ],
                selectors: None,
                duplicate_words: false,
                quoted: false,
            },
        ],
    },
    Split {
        name: "held",
        fact: [
            "On ticket {name}, {word} is entry {role}.",
            "On ticket {name}, {word} is entry {role};",
        ],
        question: "How should ticket {name} be read back?",
        families: [
            Family {
                name: "left-column",
                counts: [
                    "return the opening word by itself",
                    "return exactly the two opening words",
                ],
                selectors: Some([
                    "Use the entry on the left of the bar: '{a}' | '{b}'",
                    "Use the entry on the right of the bar: '{a}' | '{b}'",
                ]),
                duplicate_words: false,
                quoted: true,
            },
            Family {
                name: "countermanded-memo",
                counts: [
                    "say nothing beyond the first word",
                    "end with the second word",
                ],
                selectors: Some([
                    "The final memo says '{a}', superseding '{b}'",
                    "The initial memo says '{a}', superseded by '{b}'",
                ]),
                duplicate_words: false,
                quoted: true,
            },
            Family {
                name: "ticket-print",
                counts: [
                    "cut off the reply at word one",
                    "cut off the reply at word two",
                ],
                selectors: Some([
                    "Please go with the printed line: '{a}'. The penciled line reads '{b}'",
                    "The penciled line reads '{a}'. Please go with the printed line: '{b}'",
                ]),
                duplicate_words: false,
                quoted: true,
            },
            Family {
                name: "margin-correction",
                counts: ["end at the period", "offer the next word and then finish"],
                selectors: Some([
                    "The rule in parentheses governs: ({a}) [note: {b}]",
                    "The rule in parentheses governs: [note: {a}] ({b})",
                ]),
                duplicate_words: false,
                quoted: false,
            },
            Family {
                name: "held-repeat",
                counts: [
                    "start with one word and finish with that one word",
                    "start with two words and finish with those two words",
                ],
                selectors: Some([
                    "Use '{a}' for this request and save '{b}' for tomorrow",
                    "Save '{a}' for tomorrow and use '{b}' for this request",
                ]),
                duplicate_words: true,
                quoted: true,
            },
            Family {
                name: "voice-cue",
                counts: [
                    "sign off after the first item",
                    "sign off after the second item",
                ],
                selectors: Some([
                    "The cue to obey is '{a}'; '{b}' is background chatter",
                    "The background chatter is '{a}'; '{b}' is the cue to obey",
                ]),
                duplicate_words: false,
                quoted: true,
            },
            Family {
                name: "unquoted-change",
                counts: ["display just one word", "display just two words"],
                selectors: Some([
                    "Please {a}. We had planned to {b}, but that changed",
                    "We had planned to {a}, but that changed. Please {b}",
                ]),
                duplicate_words: false,
                quoted: false,
            },
            Family {
                name: "direct-held",
                counts: [
                    "Conclude at the end of item one",
                    "Conclude at the end of item two",
                ],
                selectors: None,
                duplicate_words: false,
                quoted: false,
            },
        ],
    },
];

pub const ROLE: [&str; 4] = ["one", "two", "three", "four"];
