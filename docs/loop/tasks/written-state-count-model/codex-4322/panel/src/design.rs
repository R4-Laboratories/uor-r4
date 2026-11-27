// Each family pairs two natural directives, in both variants. Side 0
// releases at the sampled decision and side 1 copies one additional word.
pub type Pair = [&'static str; 2];

pub struct Family {
    pub name: &'static str,
    pub wording: [Pair; 2],
}

pub const DEV_FIRST: &[Family] = &[
    Family {
        name: "proofreading",
        wording: [
            ["Cross out the second word and every word after it", "Cross out the third word and every word after it"],
            ["Remove everything from the second word onward", "Remove everything from the third word onward"],
        ],
    },
    Family {
        name: "output-length",
        wording: [
            ["Your whole reply should have just one word", "Your whole reply should have just two words"],
            ["Limit the reply to the opening word alone", "Limit the reply to the opening two words"],
        ],
    },
    Family {
        name: "dictation",
        wording: [
            ["Break off the dictation before saying the second word", "Break off the dictation after saying the second word"],
            ["Finish as soon as the first word is spoken", "Finish as soon as the second word is spoken"],
        ],
    },
    Family {
        name: "stop-scope",
        wording: [
            ["Disregard 'stop after the second word'. Follow 'stop before the second word'", "Disregard 'stop before the second word'. Follow 'stop after the second word'"],
            ["Of 'stop after the second word' and 'stop before the second word', apply the latter", "Of 'stop before the second word' and 'stop after the second word', apply the latter"],
        ],
    },
];

pub const DEV_POST: &[Family] = &[
    Family {
        name: "excerpt-end",
        wording: [
            ["End your excerpt just before the third word", "End your excerpt just after the third word"],
            ["Your excerpt ends before you get to the third word", "Your excerpt ends after you give the third word"],
        ],
    },
    Family {
        name: "copyediting",
        wording: [
            ["Keep the second word and omit everything that follows it", "Keep the third word too and omit everything that follows it"],
            ["Print the first two words and leave the rest out", "Print the first three words and leave the rest out"],
        ],
    },
    Family {
        name: "length-budget",
        wording: [
            ["The answer must contain two words in total", "The answer must contain three words in total"],
            ["Use up a budget of two words for this reply", "Use up a budget of three words for this reply"],
        ],
    },
    Family {
        name: "selection-scope",
        wording: [
            ["When deciding where to end, use 'before the third word' and ignore 'after the third word'", "When deciding where to end, use 'after the third word' and ignore 'before the third word'"],
            ["Between 'end after the third word' and 'end before the third word', choose the second instruction", "Between 'end before the third word' and 'end after the third word', choose the second instruction"],
        ],
    },
];

pub const HELD_FIRST: &[Family] = &[
    Family {
        name: "display-window",
        wording: [
            ["Show the opening word by itself", "Show the opening word with the one immediately following it"],
            ["Leave only the opening word visible", "Leave only the first two words visible"],
        ],
    },
    Family {
        name: "next-entry",
        wording: [
            ["Do not proceed to the second word", "Proceed as far as the second word and finish there"],
            ["Let the period be the end of the reply", "Let the next word be the end of the reply"],
        ],
    },
    Family {
        name: "allowance",
        wording: [
            ["Use an allowance of a single word for the entire answer", "Use an allowance of a pair of words for the entire answer"],
            ["Make the total word count one", "Make the total word count two"],
        ],
    },
    Family {
        name: "selected-scope",
        wording: [
            ["Follow the quoted instruction that comes first: 'stop before word two' or 'stop after word two'", "Follow the quoted instruction that comes first: 'stop after word two' or 'stop before word two'"],
            ["Use the former of these choices: 'leave out the second word' or 'add the second word and end there'", "Use the former of these choices: 'add the second word and end there' or 'leave out the second word'"],
        ],
    },
];

pub const HELD_POST: &[Family] = &[
    Family {
        name: "cutoff",
        wording: [
            ["Place the end of the reply before word number three", "Place the end of the reply after word number three"],
            ["Finish without reaching the third word", "Finish once you have given the third word"],
        ],
    },
    Family {
        name: "proof-cut",
        wording: [
            ["Retain the first and second words. Omit the rest", "Retain the first, second and third words. Omit the rest"],
            ["Keep the second word but leave the third out", "Keep the second and the third but leave the fourth out"],
        ],
    },
    Family {
        name: "space-allotment",
        wording: [
            ["Fill the reply with a total of two words", "Fill the reply with a total of three words"],
            ["Count out two words and finish there", "Count out three words and finish there"],
        ],
    },
    Family {
        name: "exception-scope",
        wording: [
            ["Obey the instruction before the slash: 'finish before word three' / 'finish after word three'", "Obey the instruction before the slash: 'finish after word three' / 'finish before word three'"],
            ["For the ending, pick the instruction in the second set of brackets: [after the third word] [before the third word]", "For the ending, pick the instruction in the second set of brackets: [before the third word] [after the third word]"],
        ],
    },
];

pub const FACTS: [&str; 2] = [
    "Card {name} places {word} in the {role} slot.",
    "Sheet {name} uses {word} for place {role}.",
];

pub const QUESTIONS: [&str; 2] = [
    "What text should card {name} produce?",
    "What should sheet {name} say?",
];

pub const ROLE: [&str; 4] = ["first", "second", "third", "fourth"];
