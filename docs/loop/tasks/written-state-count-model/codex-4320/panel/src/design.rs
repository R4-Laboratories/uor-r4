pub(super) struct Family {
    pub name: &'static str,
    pub variants: [[&'static str; 2]; 2],
}

pub(super) const HELD: &[Family] = &[
    Family {
        name: "destination_fields",
        variants: [
            [
                "The printed answer box takes only the opening word and its period. Leave the follow-up field off the reply.",
                "The printed answer box takes the opening word and its period. Copy the follow-up word into the same reply after one space, with nothing after it.",
            ],
            [
                "Fill the outgoing field with the opening word and a period. The follow-up word belongs in an unused field.",
                "Fill the outgoing field with the opening word and a period. The follow-up word belongs in that same field after one space, without its own punctuation.",
            ],
        ],
    },
    Family {
        name: "transcription_boundary",
        variants: [
            [
                "Transcribe the opening word with a period after it. The transcription boundary falls directly after this mark.",
                "Transcribe the opening word with a period after it. The transcription boundary falls directly after the follow-up word, separated from the mark by one space.",
            ],
            [
                "The opening word and its period are the full transcription. Cut the reply at that point.",
                "The opening word and its period start the transcription. Cut the reply just after the follow-up word, which begins one space after the period.",
            ],
        ],
    },
    Family {
        name: "two_pass",
        variants: [
            [
                "On the first pass, copy the opening word and put a period after it. The second pass is disabled, so end there.",
                "On the first pass, copy the opening word and put a period after it. On the second pass, add a space and that person's follow-up word without another mark.",
            ],
            [
                "In pass one, give the opening word and period. Skip pass two, then finish the reply.",
                "In pass one, give the opening word and period. In pass two, give one space and the follow-up word, then finish the reply.",
            ],
        ],
    },
    Family {
        name: "recipient_forms",
        variants: [
            [
                "Send the brief form, which consists of the opening word followed by a period. The reply ends there.",
                "Send the extended form, which consists of the opening word followed by a period and one space, then the follow-up word with no trailing mark.",
            ],
            [
                "Choose the short dispatch. Its entire text is the opening word and one period.",
                "Choose the long dispatch. Its entire text is the opening word, one period, one space, and the follow-up word in that order.",
            ],
        ],
    },
];

pub(super) const FINITE: &[Family] = &[Family {
    name: "finite_length",
    variants: [
        [
            "Return the opening word with a period, then finish.",
            "Return the opening word with a period, then one space and the follow-up word with no punctuation.",
        ],
        [
            "The reply must contain just the opening word and a period.",
            "The reply must contain just the opening word, a period, a space, and the follow-up word.",
        ],
    ],
}];

pub(super) struct Topic {
    pub name: &'static str,
    pub question: &'static str,
    pub facts: [&'static str; 2],
}

pub(super) const TOPICS: &[Topic] = &[
    Topic {
        name: "trail_stamp",
        question: "What opening word is on the hike stamp issued to {person}?",
        facts: [
            "The hike stamp issued to {person} shows opening word {first} and follow-up word {next}.",
            "The follow-up word {next} and opening word {first} appear on the hike stamp issued to {person}.",
        ],
    },
    Topic {
        name: "ferry_permit",
        question: "Which opening word appears on {person}'s ferry permit?",
        facts: [
            "The ferry permit for {person} lists opening word {first} and follow-up word {next}.",
            "Follow-up word {next} and opening word {first} identify the ferry permit for {person}.",
        ],
    },
    Topic {
        name: "weather_diary",
        question: "How does the code entry open in the weather diary kept by {person}?",
        facts: [
            "The weather diary kept by {person} lists opening word {first} and follow-up word {next}.",
            "The follow-up word {next} and opening word {first} identify the weather diary kept by {person}.",
        ],
    },
    Topic {
        name: "archive_stub",
        question: "What is the opening word printed on the archive stub held by {person}?",
        facts: [
            "The archive stub held by {person} pairs opening word {first} with follow-up word {next}.",
            "The archive stub uses follow-up word {next} and opening word {first} for {person}.",
        ],
    },
];

pub(super) const FINITE_TOPIC: Topic = Topic {
    name: "weigh_station",
    question: "What opening word is printed on the weigh station ticket signed by {person}?",
    facts: [
        "The weigh station ticket signed by {person} assigns opening word {first} and follow-up word {next}.",
        "The follow-up word {next} and opening word {first} are on the weigh station ticket signed by {person}.",
    ],
};

pub(super) const WORDS: &str = "acorn beetle cabin candle chess clay comet crab deer doll drum feather fence flute fork fox frog glove goat helmet honey island jelly jewel kite leaf marble mouse mushroom needle nest ocean onion otter owl paint peach pencil pepper pirate rabbit radio rain rope rose salad sand sheep shell soap spoon star storm tent thread toast tomato tower turtle violin wagon whale wheat wolf";

const FIRST: &[&str] = &[
    "Adela", "Anya", "Armand", "Beth", "Blaise", "Bruno", "Clara", "Cleo", "Cora", "Desmond",
    "Devin", "Dina", "Dora", "Edith", "Elena", "Elsa", "Enoch", "Esme", "Etta", "Fiona", "Freda",
    "Garth", "Gemma", "Greta", "Hana", "Hector", "Ida", "Ines", "Jared", "Jasper", "Jill", "Jules",
    "Kara", "Karl", "Kent", "Kiara", "Lana", "Leif", "Leo", "Lila", "Lucia", "Mabel", "Marta",
    "Milo", "Mina", "Oleg", "Orla", "Otis", "Petra", "Rhea", "Rina", "Robin", "Ronan", "Ruth",
    "Sid", "Sven", "Tara", "Una", "Vera", "Willa", "Yara", "Yuri",
];

const LAST: &[&str] = &[
    "Arden", "Baxter", "Briggs", "Calloway", "Dalton", "Frost", "Hawke", "Iverson", "Jensen",
    "Keane", "Larkin", "Morse", "Parker", "Quill", "Reeve", "Salem", "Tanner", "Vale", "Wren",
];

pub(super) fn name(index: usize) -> String {
    format!(
        "{} {}",
        FIRST[index % FIRST.len()],
        LAST[index / FIRST.len()]
    )
}
