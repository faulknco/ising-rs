/// CLI: run a temperature sweep and print CSV to stdout.
///
/// Usage:
///   cargo run --release --bin sweep
///   cargo run --release --bin sweep -- --n 30 --geometry cubic --j 1.0
///
/// Output columns: T, E, M, Cv, chi
///
/// Expected results (J=1, h=0, periodic BCs):
///   Square2D:     ground state E ≈ -2.0, Tc ≈ 2.27
///   Triangular2D: ground state E ≈ -3.0, Tc ≈ 3.64
///   Cubic3D:      ground state E ≈ -3.0, Tc ≈ 4.51

use std::env;
use ising::{
    lattice::Geometry,
    sweep::{run, SweepConfig},
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let config = parse_args(&args);

    eprintln!(
        "Running {} sweep: N={}, J={}, h={}, T=[{:.1}..{:.1}], warmup={}, samples={}",
        geometry_name(config.geometry),
        config.n,
        config.j,
        config.h,
        config.t_min,
        config.t_max,
        config.warmup_sweeps,
        config.sample_sweeps,
    );

    let results = run(&config);

    // CSV header
    println!("T,E,M,Cv,chi");
    for obs in &results {
        println!(
            "{:.4},{:.6},{:.6},{:.6},{:.6}",
            obs.temperature,
            obs.energy,
            obs.magnetisation,
            obs.heat_capacity,
            obs.susceptibility,
        );
    }

    // Summary to stderr
    let tc = estimate_curie(&results);
    let ground_e = results.first().map(|o| o.energy).unwrap_or(0.0);
    eprintln!("Ground state E/spin ≈ {:.4}  (expected: {})", ground_e, expected_ground(config.geometry));
    eprintln!("Estimated Tc        ≈ {:.2}  (expected: {})", tc, expected_tc(config.geometry));
}

/// Find Tc as the temperature of maximum |dM/dT|
fn estimate_curie(results: &[ising::observables::Observables]) -> f64 {
    if results.len() < 3 {
        return 0.0;
    }
    let mut max_dm = 0.0_f64;
    let mut tc = results[0].temperature;
    for i in 1..results.len() - 1 {
        let dm = (results[i - 1].magnetisation - results[i + 1].magnetisation).abs();
        if dm > max_dm {
            max_dm = dm;
            tc = results[i].temperature;
        }
    }
    tc
}

fn geometry_name(g: Geometry) -> &'static str {
    match g {
        Geometry::Square2D => "Square2D",
        Geometry::Triangular2D => "Triangular2D",
        Geometry::Cubic3D => "Cubic3D",
    }
}

fn expected_ground(g: Geometry) -> &'static str {
    match g {
        Geometry::Square2D => "-2.0 J",
        Geometry::Triangular2D | Geometry::Cubic3D => "-3.0 J",
    }
}

fn expected_tc(g: Geometry) -> &'static str {
    match g {
        Geometry::Square2D => "2.27 J/kB (Onsager exact)",
        Geometry::Triangular2D => "3.64 J/kB",
        Geometry::Cubic3D => "4.51 J/kB",
    }
}

/// Minimal arg parser: --n, --geometry, --j, --h, --warmup, --samples, --seed
fn parse_args(args: &[String]) -> SweepConfig {
    let mut cfg = SweepConfig::default();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => { cfg.n = args[i + 1].parse().unwrap(); i += 2; }
            "--geometry" => {
                cfg.geometry = match args[i + 1].as_str() {
                    "triangular" => Geometry::Triangular2D,
                    "cubic" => Geometry::Cubic3D,
                    _ => Geometry::Square2D,
                };
                i += 2;
            }
            "--j" => { cfg.j = args[i + 1].parse().unwrap(); i += 2; }
            "--h" => { cfg.h = args[i + 1].parse().unwrap(); i += 2; }
            "--warmup" => { cfg.warmup_sweeps = args[i + 1].parse().unwrap(); i += 2; }
            "--samples" => { cfg.sample_sweeps = args[i + 1].parse().unwrap(); i += 2; }
            "--seed" => { cfg.seed = args[i + 1].parse().unwrap(); i += 2; }
            "--tmin" => { cfg.t_min = args[i + 1].parse().unwrap(); i += 2; }
            "--tmax" => { cfg.t_max = args[i + 1].parse().unwrap(); i += 2; }
            "--steps" => { cfg.t_steps = args[i + 1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }
    cfg
}
