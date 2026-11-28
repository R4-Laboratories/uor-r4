//! Select the ordinary client compile graph while retaining the same executable name.

#[cfg(feature = "p4-only-client")]
#[path = "uor-chat-p4.rs"]
mod client;

#[cfg(all(feature = "serving-wasm", not(feature = "p4-only-client")))]
#[path = "uor-chat.rs"]
mod client;

#[cfg(not(any(feature = "serving-wasm", feature = "p4-only-client")))]
compile_error!("uor-chat requires either p4-only-client or serving-wasm");

fn main() {
    client::main();
}
