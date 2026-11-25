//! The hologram-forge trust core, as Lean verified it.
//!
//! Every body in `generated/hologram_forge_core.rs` is the exact `prod-codegen`
//! output from the LCNF export of the kernel checked module
//! `HologramForge.Forge` (Lean 4.32.1, leanchecker replayed,
//! exact per declaration axiom sets), produced by `scripts/lane.sh` from the
//! PrismPM commit in `PRISMPM_REV`. Nothing here is written by hand except
//! this file, which only names the refusal type the generated code returns,
//! and the PrismPM archive code under `holo/` and `error.rs`, vendored verbatim
//! from the pinned commit so a published application is composed by PrismPM's
//! own writer (the lane checks the vendored bytes against PrismPM).
//! The generated code uses owned strings and vectors, so this crate links std.

#![recursion_limit = "256"]
#![deny(unsafe_code)]
#![allow(non_snake_case, unused_parens, unused_variables, clippy::all, clippy::pedantic)]

extern crate alloc;

/// Refusals of the generated bodies: checked arithmetic overflow and
/// caller owned output buffer exhaustion. Never wrapping, never a panic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeError {
    AddOverflow,
    MulOverflow,
    OutputTooSmall,
}

include!("../../generated/hologram_forge_core.rs");

/// PrismPM's diagnostics, verbatim (`crates/prismpm/src/error.rs`).
#[allow(dead_code)]
pub mod error;
/// PrismPM's Hologram application composer and validator, verbatim (`crates/prismpm/src/holo/`).
#[allow(dead_code)]
pub mod holo;

/// The browser boundary, a host transport adapter over the generated core.
#[allow(unsafe_code)]
pub mod abi;

/// Lean-certified Verdant Studio lifecycle/planning core, vendored verbatim from `afflom/verdant-studio`.
#[allow(dead_code, unused_imports, non_snake_case, unused_parens, unused_variables, clippy::all, clippy::pedantic)]
pub mod verdant_studio;
/// In-memory virtual workspace effect driver for `verdant_studio::run`.
pub mod verdant_driver;
/// Pure-integer 4 KiB geometric virtual-RAM backing store adapted from `afflom/vRAM`.
pub mod vram_store;
