# Conditional ordinary-client review

Lab: codex

The independent reviewer `links_fit` returned **FIT** for pinning client methods v5 and then verified the v6 per-row prompt, distance and BPE identity checks. The approved v6 methods manifest is `6ab042d912e0629b33175c54be47c7f54dcdc40e3228a49161714f3665d1cade`. Its finite fixture manifest is `722c5e103f47f02401e7f7dfcf52c2b15ffdf04ab46c6a3d159a0a692198f286`. The parent released capture after the registered source pass, gate JSON `dd4b15aee4246d1ccb364adc21c5d490a2fec8f6e1d5a60f4fa17f091037dcea`, against device endpoints `36da6cac65bd6c54c24b46256ad1897bffd0674de8a8663a783e34a62cbba11a`.

The fixture and review cover equality of generated assistant history, target IDs, visible reply bytes and matched stock traces, original-source cursor transitions, the no-winner path, retry, and correct token prefixes at the cap. Deliberately wrong output tokens, text, source addresses and prompt bindings failed their checks. Ordinary native learns only the scope field in this comparison. Both first role and pending completion are supplied on every arm. For a partial intervention with no winner, three empty byte ranges bound to the authenticated prompt sends the binding through its existing invalid-binding path. Report these separately from a reader's explicit exclusion. A cap-matched stock prefix cannot earn an EOS result.
