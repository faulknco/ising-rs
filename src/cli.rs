/// Shared CLI argument parsing helpers for all binaries.
///
/// Provides `get_arg` for safe value extraction and `parse_arg` for
/// type-safe parsing with clear error messages.
use std::process;

use crate::lattice::Geometry;

/// Extract the value following a flag, exiting with a message if missing.
pub fn get_arg(args: &[String], i: usize, flag: &str) -> String {
    if i + 1 >= args.len() {
        eprintln!("Error: {flag} requires a value");
        process::exit(1);
    }
    args[i + 1].clone()
}

/// Parse a flag's value into the target type, exiting with a clear message on failure.
pub fn parse_arg<T: std::str::FromStr>(args: &[String], i: usize, flag: &str) -> T
where
    T::Err: std::fmt::Display,
{
    let val = get_arg(args, i, flag);
    val.parse().unwrap_or_else(|e| {
        eprintln!("Error: invalid value '{val}' for {flag}: {e}");
        process::exit(1);
    })
}

/// Parse a `--geometry` flag value into a `Geometry` variant.
pub fn parse_geometry(args: &[String], i: usize) -> Geometry {
    match get_arg(args, i, "--geometry").as_str() {
        "cubic" => Geometry::Cubic3D,
        "triangular" => Geometry::Triangular2D,
        "square" => Geometry::Square2D,
        other => {
            eprintln!("Error: unknown geometry '{other}' (expected: square, triangular, cubic)");
            process::exit(1);
        }
    }
}
