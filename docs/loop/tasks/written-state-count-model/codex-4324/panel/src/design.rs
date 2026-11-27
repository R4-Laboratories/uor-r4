//! Fresh wording and balanced factual assignments for the release diagnostic.

pub const FACT: &str = "Folio {name} marks line {role} with {word}";
pub const QUESTION: &str = "Which words belong in the reply for folio {name}?";
pub const FORMAT: &str =
    "Put a period right after the opening word and separate any remaining words with a space.";
pub const ROLES: [&str; 4] = ["one", "two", "three", "four"];
pub const ORDERS: [[usize; 4]; 8] = [
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0],
    [0, 2, 1, 3],
    [1, 3, 0, 2],
    [3, 0, 2, 1],
    [2, 1, 3, 0],
];
pub const LABEL_OFFSETS: [[usize; 4]; 12] = [
    [0, 1, 3, 5],
    [0, 1, 5, 3],
    [0, 3, 1, 5],
    [0, 3, 5, 1],
    [0, 5, 1, 3],
    [0, 5, 3, 1],
    [0, 2, 4, 6],
    [0, 2, 6, 4],
    [0, 4, 2, 6],
    [0, 4, 6, 2],
    [0, 6, 2, 4],
    [0, 6, 4, 2],
];

pub struct Scoped {
    pub name: &'static str,
    pub clauses: [&'static str; 2],
    pub selectors: [&'static str; 2],
    pub quoted: bool,
}

impl Scoped {
    pub fn directive(&self, operative: usize, short: usize) -> String {
        self.selectors[operative]
            .replace("{a}", self.clauses[short])
            .replace("{b}", self.clauses[1 - short])
    }
}

pub const SCOPED: [Scoped; 8] = [
    Scoped {
        name: "dial-setting",
        clauses: [
            "send only the opening entry",
            "send the opening entry and its immediate successor",
        ],
        selectors: [
            "I set the dial to low. Low means '{a}'; high means '{b}'.",
            "I set the dial to high. Low means '{a}'; high means '{b}'.",
        ],
        quoted: true,
    },
    Scoped {
        name: "rule-elimination",
        clauses: ["keep the first entry alone", "keep the first and second entries"],
        selectors: [
            "Rules in order: '{a}' / '{b}'. Strike rule two and follow the remaining rule.",
            "Rules in order: '{a}' / '{b}'. Strike rule one and follow the remaining rule.",
        ],
        quoted: true,
    },
    Scoped {
        name: "agreed-speaker",
        clauses: ["leave out every word after the opener", "add exactly one word after the opener"],
        selectors: [
            "Dana asked you to {a}; Eli asked you to {b}. I agreed with Dana.",
            "Dana asked you to {a}; Eli asked you to {b}. I agreed with Eli.",
        ],
        quoted: false,
    },
    Scoped {
        name: "check-mark",
        clauses: ["keep the reply to a solitary word", "keep the reply to a pair of words"],
        selectors: [
            "Follow the instruction marked YES. YES: '{a}'. NO: '{b}'.",
            "Follow the instruction marked YES. NO: '{a}'. YES: '{b}'.",
        ],
        quoted: true,
    },
    Scoped {
        name: "recording-time",
        clauses: ["break off before the next entry", "break off after the next entry"],
        selectors: [
            "The request filed at noon is in force. At noon: '{a}'. At dusk: '{b}'.",
            "The request filed at dusk is in force. At noon: '{a}'. At dusk: '{b}'.",
        ],
        quoted: true,
    },
    Scoped {
        name: "instruction-aside",
        clauses: ["read line one alone", "read lines one and two"],
        selectors: [
            "Instruction: {a}. Aside for my notes: {b}.",
            "Aside for my notes: {a}. Instruction: {b}.",
        ],
        quoted: false,
    },
    Scoped {
        name: "trailing-star",
        clauses: ["return the opener and omit the rest", "return the opener and exactly its successor"],
        selectors: [
            "Apply the statement carrying a star. {a} *; {b} -.",
            "Apply the statement carrying a star. {a} -; {b} *.",
        ],
        quoted: false,
    },
    Scoped {
        name: "spoken-label",
        clauses: ["finish once the first line has been copied", "finish once the second line has been copied"],
        selectors: [
            "The caller said alpha. Alpha calls for '{a}'; beta calls for '{b}'. Use the caller's label.",
            "The caller said beta. Alpha calls for '{a}'; beta calls for '{b}'. Use the caller's label.",
        ],
        quoted: true,
    },
];

pub struct Direct {
    pub name: &'static str,
    pub contrast: u8,
    pub directives: [&'static str; 2],
}

pub const DIRECT: [Direct; 4] = [
    Direct {
        name: "readback-count",
        contrast: 0,
        directives: [
            "Read back the first word and omit the rest.",
            "Read back the first two words and omit the rest.",
        ],
    },
    Direct {
        name: "entry-budget",
        contrast: 0,
        directives: [
            "There is room for one entry on my reply slip, so use one.",
            "There is room for two entries on my reply slip, so use two.",
        ],
    },
    Direct {
        name: "last-line",
        contrast: 1,
        directives: [
            "Make the value from line two your last word.",
            "Make the value from line three your last word.",
        ],
    },
    Direct {
        name: "prefix-excerpt",
        contrast: 1,
        directives: [
            "Give me a two-word excerpt that starts at line one.",
            "Give me a three-word excerpt that starts at line one.",
        ],
    },
];
