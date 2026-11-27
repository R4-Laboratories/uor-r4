use super::prior_design::{Design, Direct, Scoped};

pub const HELD: Design = Design {
    split: "blinded",
    fact_header: "Voucher {name}.",
    fact: "{role} slot: {word}",
    question: "Which entries should I transcribe in slot order?",
    format: "Append a period to the opening entry. Separate later entries by spaces.",
    scoped: &[
        Scoped {
            name: "paired-verdict",
            clauses: ["transcribe one entry from the start", "transcribe two entries from the start"],
            selectors: [
                "The panel upheld ruling East. East reads '{a}'. West reads '{b}'.",
                "The panel upheld ruling West. East reads '{a}'. West reads '{b}'.",
            ],
        },
        Scoped {
            name: "paired-desk-lamp",
            clauses: ["write down the first slot only", "write down both of the first two slots"],
            selectors: [
                "There are two instruction desks. The desk with the lit lamp decides. Lit desk: {a}. Dark desk: {b}.",
                "There are two instruction desks. The desk with the lit lamp decides. Dark desk: {a}. Lit desk: {b}.",
            ],
        },
        Scoped {
            name: "paired-approval-code",
            clauses: ["transcribe through slot one", "transcribe through slot two"],
            selectors: [
                "Approval code Sun. Rule Sun says to {a}. Rule Moon says to {b}. Obey the rule whose name matches the approval code.",
                "Approval code Moon. Rule Sun says to {a}. Rule Moon says to {b}. Obey the rule whose name matches the approval code.",
            ],
        },
        Scoped {
            name: "paired-overrule",
            clauses: ["include the first slot and end there", "include the first and second slots and end there"],
            selectors: [
                "Proposal Oak: {a}. Proposal Elm: {b}. The judge struck out Elm, leaving the other proposal in force.",
                "Proposal Oak: {a}. Proposal Elm: {b}. The judge struck out Oak, leaving the other proposal in force.",
            ],
        },
        // The remaining four families come from the independent wording author.
        super::independent::TYPE_CASE,
        super::independent::BELL,
        super::independent::TRANSLATION_BOOTH,
        super::independent::RECEIPT,
    ],
    direct: &[
        Direct {
            name: "paired-direct-tray",
            listing: "Sample tray {name} records the slots in order",
            question: "Which tray entries should I copy?",
            contrast: 0,
            directives: [
                "Transcribe up to and including slot one, in list order.",
                "Transcribe up to and including slot two, in list order.",
            ],
        },
        super::independent::CLERK,
        Direct {
            name: "paired-direct-crate",
            listing: "Crate survey {name} records these labels in numbered order",
            question: "Which crate labels go in this transcription?",
            contrast: 1,
            directives: [
                "From the beginning of the list, include a total of two entries.",
                "From the beginning of the list, include a total of three entries.",
            ],
        },
        super::independent::REHEARSAL,
    ],
};
