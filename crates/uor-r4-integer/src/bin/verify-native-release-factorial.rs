#[path = "shared/verify_release_factorial.rs"]
mod verify_release_factorial;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    verify_release_factorial::run(&args, "verify-native-release-factorial")
}
