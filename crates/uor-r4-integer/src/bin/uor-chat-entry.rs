//! Select the ordinary client compile graph while retaining the same executable name.

#[cfg(feature = "p4-only-client")]
#[path = "uor-chat-p4.rs"]
mod client;

#[cfg(all(feature = "serving-wasm", not(feature = "p4-only-client")))]
#[path = "uor-chat.rs"]
mod client;

#[cfg(feature = "butterfly-v3-client")]
#[path = "uor-chat-butterfly-v3.rs"]
mod butterfly_v3;

#[cfg(not(any(
    feature = "serving-wasm",
    feature = "p4-only-client",
    feature = "butterfly-v3-client"
)))]
compile_error!("uor-chat requires a serving client feature");

fn main() {
    #[cfg(feature = "butterfly-v3-client")]
    if butterfly_v3::try_main() {
        return;
    }

    #[cfg(any(feature = "serving-wasm", feature = "p4-only-client"))]
    client::main();

    #[cfg(not(any(feature = "serving-wasm", feature = "p4-only-client")))]
    butterfly_v3::usage();
}
