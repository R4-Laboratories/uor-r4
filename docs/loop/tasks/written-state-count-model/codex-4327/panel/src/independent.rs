use super::prior_design::{Direct, Scoped};

pub const TYPE_CASE: Scoped = Scoped {
    name: "paired-type-case",
    clauses: [
        "set the leading entry by itself",
        "set the leading pair of entries in succession",
    ],
    selectors: [
        "Use the line marked in type. In type: {a}. In pencil: {b}.",
        "Use the line marked in type. In pencil: {a}. In type: {b}.",
    ],
};

pub const BELL: Scoped = Scoped {
    name: "paired-bell",
    clauses: [
        "call out only the opening entry",
        "call out the opening entry along with its immediate successor in slot order",
    ],
    selectors: [
        "One bell rang. For one bell: {a}. For two bells: {b}.",
        "Two bells rang. For one bell: {a}. For two bells: {b}.",
    ],
};

pub const TRANSLATION_BOOTH: Scoped = Scoped {
    name: "paired-translation-booth",
    clauses: [
        "voice the opening entry and fall silent",
        "voice the opening two entries, in sequence",
    ],
    selectors: [
        "The copper lamp is lit. Copper asks: {a}. Jade asks: {b}.",
        "The jade lamp is lit. Copper asks: {a}. Jade asks: {b}.",
    ],
};

pub const RECEIPT: Scoped = Scoped {
    name: "paired-receipt",
    clauses: [
        "transcribe the head entry alone",
        "transcribe the head entry and the entry immediately after it",
    ],
    selectors: [
        "Honor the receipt with the seal. Sealed: {a}. Unsealed: {b}.",
        "Honor the receipt with the seal. Unsealed: {a}. Sealed: {b}.",
    ],
};

pub const CLERK: Direct = Direct {
    name: "paired-direct-clerk",
    listing: "Ledger {name} gives these entries in order",
    question: "What should I send the receiving clerk?",
    contrast: 0,
    directives: [
        "The receiving clerk needs a prefix of the list ending at the first entry.",
        "The receiving clerk needs a prefix of the list ending at the second entry.",
    ],
};

pub const REHEARSAL: Direct = Direct {
    name: "paired-direct-rehearsal",
    listing: "Cue sheet {name} lists these entries from left to right",
    question: "What should I read from the cue sheet?",
    contrast: 1,
    directives: [
        "For the rehearsal, read through the second entry, inclusive.",
        "For the rehearsal, read through the third entry, inclusive.",
    ],
};
