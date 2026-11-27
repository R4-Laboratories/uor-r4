pub struct Scoped {
    pub name: &'static str,
    pub clauses: [&'static str; 2],
    pub selectors: [&'static str; 2],
}

impl Scoped {
    pub fn directive(&self, operative: usize, short: usize) -> String {
        self.selectors[operative]
            .replace("{a}", self.clauses[short])
            .replace("{b}", self.clauses[1 - short])
    }
}

pub struct Direct {
    pub name: &'static str,
    pub listing: &'static str,
    pub question: &'static str,
    pub contrast: u8,
    pub directives: [&'static str; 2],
}

pub struct Design {
    pub split: &'static str,
    pub format: &'static str,
    pub fact_header: &'static str,
    pub fact: &'static str,
    pub question: &'static str,
    pub scoped: &'static [Scoped; 8],
    pub direct: &'static [Direct; 4],
}

pub const TRAIN: Design = Design {
    split: "train",
    format: "Add a period immediately after the initial value. Space any later values normally.",
    fact_header: "Register {name} log.",
    fact: "The {role} position holds {word}",
    question: "What should the register {name} worker send in numbered order?",
    scoped: &[
        Scoped {
            name: "training-live-dead",
            clauses: ["issue a single entry", "issue a pair of entries"],
            selectors: [
                "Green is live and red is dead. Green says '{a}'. Red says '{b}'.",
                "Red is live and green is dead. Green says '{a}'. Red says '{b}'.",
            ],
        },
        Scoped {
            name: "training-pending",
            clauses: [
                "supply the initial value alone",
                "supply the initial value and one more",
            ],
            selectors: [
                "From today's card follow '{a}'. On the pending card I wrote '{b}'.",
                "On the pending card I wrote '{a}'. From today's card follow '{b}'.",
            ],
        },
        Scoped {
            name: "training-transfer-desk",
            clauses: [
                "pass along only line one",
                "pass along line one and line two",
            ],
            selectors: [
                "Intake says {a}. Dispatch says {b}. Let intake govern the transfer.",
                "Intake says {a}. Dispatch says {b}. Let dispatch govern the transfer.",
            ],
        },
        Scoped {
            name: "training-layout-approval",
            clauses: [
                "publish the head entry by itself",
                "publish the head entry with its successor",
            ],
            selectors: [
                "Version left is approved. Left says '{a}'. Right says '{b}'.",
                "Version right is approved. Left says '{a}'. Right says '{b}'.",
            ],
        },
        Scoped {
            name: "training-raised-tag",
            clauses: [
                "finish without a second value",
                "finish with a second value",
            ],
            selectors: [
                "Accept the rule with a raised tag. Raised {a}. Flat {b}.",
                "Accept the rule with a raised tag. Flat {a}. Raised {b}.",
            ],
        },
        Scoped {
            name: "training-valid-window",
            clauses: [
                "record just the earliest slot",
                "record the earliest two slots",
            ],
            selectors: [
                "Today's window is open. Today {a}. Tomorrow {b}.",
                "Tomorrow's window is open. Today {a}. Tomorrow {b}.",
            ],
        },
        Scoped {
            name: "training-signed-slip",
            clauses: [
                "keep no value beyond the opener",
                "keep exactly one value beyond the opener",
            ],
            selectors: [
                "An unsigned note says '{a}'. A signed note says '{b}'. Honor the unsigned note.",
                "An unsigned note says '{a}'. A signed note says '{b}'. Honor the signed note.",
            ],
        },
        Scoped {
            name: "training-circuit",
            clauses: [
                "return the start of the list alone",
                "return the start of the list plus its neighbor",
            ],
            selectors: [
                "Circuit A is active. A tells you to {a}. B tells you to {b}.",
                "Circuit B is active. A tells you to {a}. B tells you to {b}.",
            ],
        },
    ],
    direct: &[
        Direct {
            name: "training-direct-opening",
            listing: "Ticket {name} records the queue in this order",
            question: "Which values should be returned for this ticket?",
            contrast: 0,
            directives: [
                "End when the opener has been sent.",
                "End when the opener and its successor have been sent.",
            ],
        },
        Direct {
            name: "training-direct-copy-limit",
            listing: "Sheet {name} numbers the entries in this sequence",
            question: "Which values should the runner carry forward?",
            contrast: 0,
            directives: [
                "Copy a single value from the start of the sequence.",
                "Copy a pair of values from the start of the sequence.",
            ],
        },
        Direct {
            name: "training-direct-last-item",
            listing: "Bundle {name} puts the words in this order",
            question: "Which values should go on the reply?",
            contrast: 1,
            directives: [
                "Work in order and end immediately after item two.",
                "Work in order and end immediately after item three.",
            ],
        },
        Direct {
            name: "training-direct-early-items",
            listing: "Form {name} lists the entries from earliest to latest",
            question: "Which values should we transcribe?",
            contrast: 1,
            directives: [
                "Send a pair of entries beginning at the head of the list.",
                "Send a trio of entries beginning at the head of the list.",
            ],
        },
    ],
};

pub const HELD: Design = Design {
    split: "held",
    format: "The opening item takes a period. Any items afterward are separated by spaces.",
    fact_header: "Pass {name} data.",
    fact: "The {role} field reads {word}",
    question: "Which values are due from pass {name} in field order?",
    scoped: &[
        Scoped {
            name: "held-edition",
            clauses: ["print the topmost field and nothing else", "print the topmost field and the following field"],
            selectors: [
                "This is the morning edition. The morning editor said '{a}'. The evening editor said '{b}'.",
                "This is the evening edition. The morning editor said '{a}'. The evening editor said '{b}'.",
            ],
        },
        Scoped {
            name: "held-drawer",
            clauses: ["include just the lead value", "include the lead value and the value beside it"],
            selectors: [
                "The rule in the upper drawer is binding. Upper {a}. Lower {b}.",
                "The rule in the lower drawer is binding. Upper {a}. Lower {b}.",
            ],
        },
        Scoped {
            name: "held-advisors",
            clauses: ["deliver only the lead field", "deliver the lead field followed by the next field"],
            selectors: [
                "Avery suggested that you {a}. Casey suggested that you {b}. Avery has the final say.",
                "Avery suggested that you {a}. Casey suggested that you {b}. Casey has the final say.",
            ],
        },
        Scoped {
            name: "held-withdrawal",
            clauses: ["give the value at the head by itself", "give the first two values together"],
            selectors: [
                "The later note was withdrawn. Earlier '{a}'. Later '{b}'.",
                "The earlier note was withdrawn. Earlier '{a}'. Later '{b}'.",
            ],
        },
        Scoped {
            name: "held-ink",
            clauses: ["stop as soon as the leading field is written", "stop as soon as the next field is written"],
            selectors: [
                "Trust the rule written in black ink. Black '{a}'. Blue '{b}'.",
                "Trust the rule written in blue ink. Black '{a}'. Blue '{b}'.",
            ],
        },
        Scoped {
            name: "held-forward",
            clauses: ["use a reply of one item", "use a reply of two items"],
            selectors: [
                "Send me the result of the forwarded note. Forwarded '{a}'. Shelved '{b}'.",
                "Send me the result of the forwarded note. Shelved '{a}'. Forwarded '{b}'.",
            ],
        },
        Scoped {
            name: "held-public-copy",
            clauses: ["omit all fields after the initial one", "retain one field after the initial one"],
            selectors: [
                "The public version asks you to {a}. The private version asks you to {b}. Follow the public version.",
                "The public version asks you to {a}. The private version asks you to {b}. Follow the private version.",
            ],
        },
        Scoped {
            name: "held-translation",
            clauses: ["output only the value in the first field", "output the values in the first two fields"],
            selectors: [
                "I translated both messages. The clear one says '{a}'. The noisy one says '{b}'. Apply the clear message.",
                "I translated both messages. The clear one says '{a}'. The noisy one says '{b}'. Apply the noisy message.",
            ],
        },
    ],
    // Independently authored directives from the product replay agent.
    direct: &[
        Direct {
            name: "held-direct-cafeteria",
            listing: "The cafe chalkboard for ticket {name} lists today's soup codes in order",
            question: "What should I copy off the chalkboard?",
            contrast: 0,
            directives: [
                "Give me the first entry only.",
                "Give me the first entry and the one that follows it.",
            ],
        },
        Direct {
            name: "held-direct-garden-club",
            listing: "The garden club recorded these projects in order on page {name}",
            question: "What did the garden club record?",
            contrast: 0,
            directives: [
                "Use the entry at the top of the list. Leave the rest out.",
                "Use the entry at the top of the list. Include the next entry as well.",
            ],
        },
        Direct {
            name: "held-direct-bakery",
            listing: "The bakery order {name} lists four cake flavors in order",
            question: "What flavors does the order list?",
            contrast: 1,
            directives: [
                "Copy the first entry and the next one, then stop.",
                "Copy the first entry and the next two, then stop.",
            ],
        },
        Direct {
            name: "held-direct-trip",
            listing: "Trip form {name} lists the meeting points in this order",
            question: "Which meeting points should I write down?",
            contrast: 1,
            directives: [
                "Show me which entries fill the first two places on the form.",
                "Show me which entries fill the first three places on the form.",
            ],
        },
    ],
};
