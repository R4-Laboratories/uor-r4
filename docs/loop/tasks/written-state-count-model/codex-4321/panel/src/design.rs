// Each family supplies two paired directives at each release. The second pair
// reverses the placement of the affirmative and negative instructions.
pub type Pair = [&'static str; 2];
pub struct Family {
    pub name: &'static str,
    pub wording: [[Pair; 2]; 2],
}

pub const TRAIN: &[Family] = &[
    Family {
        name: "orders",
        wording: [
            [
                [
                    "Stop before the second word, not after",
                    "Stop after the second word, not before",
                ],
                [
                    "Do not stop after the second word. Stop before it",
                    "Do not stop before the second word. Stop after it",
                ],
            ],
            [
                [
                    "Stop before the third word, not after",
                    "Stop after the third word, not before",
                ],
                [
                    "Do not stop after the third word. Stop before it",
                    "Do not stop before the third word. Stop after it",
                ],
            ],
        ],
    },
    Family {
        name: "inclusion",
        wording: [
            [
                [
                    "Exclude the second word and include the first word",
                    "Include the second word and exclude any later word",
                ],
                [
                    "End after the first word and omit the second",
                    "End after the second word and omit the third",
                ],
            ],
            [
                [
                    "Exclude the third word and include the second word",
                    "Include the third word and exclude any later word",
                ],
                [
                    "End after the second word and omit the third",
                    "End after the third word and omit the fourth",
                ],
            ],
        ],
    },
    Family {
        name: "quota",
        wording: [
            [
                [
                    "Give me one word in all, with a period after the first",
                    "Give me two words in all, with a period after the first",
                ],
                [
                    "Two words would be too many. Give one",
                    "One word would be too few. Give two",
                ],
            ],
            [
                [
                    "Give me two words in all, with a period after the first",
                    "Give me three words in all, with a period after the first",
                ],
                [
                    "Three words would be too many. Give two",
                    "Two words would be too few. Give three",
                ],
            ],
        ],
    },
    Family {
        name: "movement",
        wording: [
            [
                [
                    "Continue no further than the opening period",
                    "Continue one word beyond the opening period",
                ],
                [
                    "Leave out the word following the period",
                    "Add in the word following the period",
                ],
            ],
            [
                [
                    "Continue no further than the second word",
                    "Continue one word beyond the second word",
                ],
                [
                    "Leave out the word following the second word",
                    "Add in the word following the second word",
                ],
            ],
        ],
    },
];

pub const PRIMARY: &[Family] = &[
    Family {
        name: "transcript",
        wording: [
            [EMPTY, EMPTY],
            [
                [
                    "The transcript ends just ahead of the third word",
                    "The transcript ends just beyond the third word",
                ],
                [
                    "Cut the transcript before word three and never after it",
                    "Cut the transcript after word three and never before it",
                ],
            ],
        ],
    },
    Family {
        name: "editor",
        wording: [
            [EMPTY, EMPTY],
            [
                [
                    "Keep word two. Drop word three",
                    "Keep word three. Drop word four",
                ],
                [
                    "Remove word three from the reply, retaining word two",
                    "Retain word three in the reply, removing word four",
                ],
            ],
        ],
    },
    Family {
        name: "handoff",
        wording: [
            [EMPTY, EMPTY],
            [
                [
                    "Leave the third word unspoken and finish",
                    "Speak the third word aloud and finish",
                ],
                [
                    "The third word stays on the page, the second goes out",
                    "The second word goes out, and so does the third on the page",
                ],
            ],
        ],
    },
    Family {
        name: "ceiling",
        wording: [
            [EMPTY, EMPTY],
            [
                [
                    "The reply has exactly two words and no more",
                    "The reply has exactly three words and no more",
                ],
                [
                    "The allowance is two words total. Use it all",
                    "The allowance is three words total. Use it all",
                ],
            ],
        ],
    },
];

pub const SECONDARY: &[Family] = &[
    Family {
        name: "printing",
        wording: [
            [
                [
                    "Print just before word two is reached",
                    "Print until after word two is reached",
                ],
                [
                    "At word two, leave it off the printed answer",
                    "At word two, put it on the printed answer",
                ],
            ],
            [EMPTY, EMPTY],
        ],
    },
    Family {
        name: "line",
        wording: [
            [
                [
                    "Draw the line ahead of word two",
                    "Draw the line beyond word two",
                ],
                [
                    "Word two stays outside the answer, never inside",
                    "Word two stays inside the answer, never outside",
                ],
            ],
            [EMPTY, EMPTY],
        ],
    },
    Family {
        name: "deadline",
        wording: [
            [
                [
                    "Close the reply once you give the opening mark",
                    "Close the reply after giving the word beyond the opening mark",
                ],
                [
                    "Finish with the mark, without the next word",
                    "Finish with the next word, after the mark",
                ],
            ],
            [EMPTY, EMPTY],
        ],
    },
    Family {
        name: "remaining",
        wording: [
            [
                [
                    "Zero words belong beyond the opening mark",
                    "One word belongs beyond the opening mark",
                ],
                [
                    "Use one word for the whole answer. Count the opening word",
                    "Use two words for the whole answer. Count the opening word",
                ],
            ],
            [EMPTY, EMPTY],
        ],
    },
];

pub const FINITE: &[Family] = &[Family {
    name: "finite-controls",
    wording: [
        [
            [
                "The opening word completes the response",
                "The following word completes the response",
            ],
            [
                "One word suffices for this reply",
                "Two words suffice for this reply",
            ],
        ],
        [
            [
                "The second word completes this response",
                "The third word completes this response",
            ],
            [
                "Two words suffice for this reply",
                "Three words suffice for this reply",
            ],
        ],
    ],
}];

const EMPTY: Pair = ["", ""];

pub const FACTS: [&str; 4] = [
    "For {name}, the {role} word is {word}.",
    "The ledger for {name} lists {word} as the {role} word.",
    "In the record of {name}, {word} serves as the {role} word.",
    "{name} has {word} written as the {role} word.",
];

pub const QUESTIONS: [&str; 4] = [
    "What is the reply for {name}?",
    "How would you answer for {name}?",
    "What should be said for {name}?",
    "What answer belongs to {name}?",
];

pub const ROLE: [&str; 4] = ["first", "second", "third", "fourth"];
