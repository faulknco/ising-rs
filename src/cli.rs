/// Shared CLI argument parsing helpers for all binaries.
///
/// Provides safe argument extraction, type-safe parsing, `--help` support,
/// unknown-flag warnings, and input validation.
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

/// If `--help` or `-h` is present, print usage and exit.
pub fn check_help(args: &[String], usage: &str) {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("{usage}");
        process::exit(0);
    }
}

/// Warn on unrecognised flags (args starting with `--` not in `known`).
pub fn warn_unknown_flags(args: &[String], known: &[&str]) {
    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if arg.starts_with("--") {
            if known.contains(&arg.as_str()) {
                // Skip this flag's value (if it takes one) — flags without values
                // are also in `known` so we just advance by 1 for those.
                i += 1;
            } else {
                eprintln!("Warning: unknown flag '{arg}' (ignored)");
            }
        }
        i += 1;
    }
}

/// Validate that `t_steps >= 2` (needed to avoid division by zero in temperature linspace).
pub fn validate_t_steps(t_steps: usize) {
    if t_steps < 2 {
        eprintln!("Error: --steps must be at least 2 (got {t_steps})");
        process::exit(1);
    }
}

/// Validate that `n >= 2` (a 1-site lattice is physically meaningless).
pub fn validate_lattice_size(n: usize) {
    if n < 2 {
        eprintln!("Error: --n must be at least 2 (got {n})");
        process::exit(1);
    }
}

/// Validate that `samples >= 1`.
pub fn validate_samples(samples: usize, flag: &str) {
    if samples == 0 {
        eprintln!("Error: {flag} must be at least 1");
        process::exit(1);
    }
}

/// Validate that `t_min < t_max` and both are positive.
pub fn validate_temp_range(t_min: f64, t_max: f64) {
    if t_min <= 0.0 {
        eprintln!("Error: --tmin must be positive (got {t_min})");
        process::exit(1);
    }
    if t_max <= 0.0 {
        eprintln!("Error: --tmax must be positive (got {t_max})");
        process::exit(1);
    }
    if t_min >= t_max {
        eprintln!("Error: --tmin ({t_min}) must be less than --tmax ({t_max})");
        process::exit(1);
    }
}
