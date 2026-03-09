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
use ising::cli::{
    check_help, get_arg, parse_arg, parse_geometry, validate_lattice_size, validate_samples,
    validate_t_steps, validate_temp_range, warn_unknown_flags,
};
use ising::{
    lattice::Geometry,
    sweep::{run, Algorithm, SweepConfig},
};
use std::env;
use std::fs;
use std::path::Path;

const USAGE: &str = "\
sweep — Ising model temperature sweep

USAGE:
    sweep [OPTIONS]

OPTIONS:
    --n <N>              Lattice size per dimension [default: 20]
    --geometry <TYPE>    square, triangular, or cubic [default: square]
    --j <J>              Coupling constant [default: 1.0]
    --h <H>              External field [default: 0.0]
    --tmin <T>           Minimum temperature [default: 0.5]
    --tmax <T>           Maximum temperature [default: 5.0]
    --steps <N>          Number of temperature points (>=2) [default: 46]
    --warmup <N>         Warmup sweeps per temperature [default: 2000]
    --samples <N>        Measurement sweeps per temperature [default: 500]
    --seed <N>           RNG seed [default: 42]
    --wolff              Use Wolff cluster algorithm instead of Metropolis
    --outdir <DIR>       Output directory [default: .]
    --save-snapshots     Save spin snapshots to CSV
    --help, -h           Show this help message";

const KNOWN_FLAGS: &[&str] = &[
    "--n",
    "--geometry",
    "--j",
    "--h",
    "--tmin",
    "--tmax",
    "--steps",
    "--warmup",
    "--samples",
    "--seed",
    "--wolff",
    "--outdir",
    "--save-snapshots",
    "--help",
];

fn main() {
    let args: Vec<String> = env::args().collect();
    check_help(&args, USAGE);
    warn_unknown_flags(&args, KNOWN_FLAGS);

    let (config, outdir, save_snapshots) = parse_args(&args);

    let algo_name = match config.algorithm {
        Algorithm::Metropolis => "Metropolis",
        Algorithm::Wolff => "Wolff",
    };
    eprintln!(
        "Running {} sweep [{}]: N={}, J={}, h={}, T=[{:.1}..{:.1}], warmup={}, samples={}",
        geometry_name(config.geometry),
        algo_name,
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
            obs.temperature, obs.energy, obs.magnetisation, obs.heat_capacity, obs.susceptibility,
        );
    }

    // Summary to stderr
    let tc = estimate_curie(&results);
    let ground_e = results.first().map(|o| o.energy).unwrap_or(0.0);
    eprintln!(
        "Ground state E/spin ≈ {:.4}  (expected: {})",
        ground_e,
        expected_ground(config.geometry)
    );
    eprintln!(
        "Estimated Tc        ≈ {:.2}  (expected: {})",
        tc,
        expected_tc(config.geometry)
    );

    if save_snapshots {
        use ising::lattice::Lattice;
        use ising::metropolis::warm_up;
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let snap_fname = format!("snapshots_N{}.csv", config.n);
        let snap_path = Path::new(&outdir).join(&snap_fname);
        let n2 = config.n * config.n;

        let mut snap_csv = String::new();
        let header: Vec<String> = (0..n2).map(|idx| format!("s{idx}")).collect();
        snap_csv.push_str(&header.join(","));
        snap_csv.push_str(",temperature\n");

        let t_values: Vec<f64> = (0..config.t_steps)
            .map(|k| {
                config.t_min
                    + (config.t_max - config.t_min) * k as f64 / (config.t_steps - 1) as f64
            })
            .collect();

        for &t in &t_values {
            let beta = 1.0 / t;
            let mut rng =
                Xoshiro256PlusPlus::seed_from_u64(config.seed.wrapping_add((t * 1000.0) as u64));
            let mut lattice = Lattice::new(config.n, config.geometry);
            lattice.randomise(&mut rng);
            warm_up(
                &mut lattice,
                beta,
                config.j,
                0.0,
                config.warmup_sweeps,
                &mut rng,
            );

            // Take 10 snapshots per temperature, spaced 100 sweeps apart
            for _ in 0..10 {
                warm_up(&mut lattice, beta, config.j, 0.0, 100, &mut rng);
                // Take z=0 slice (first N*N spins in cubic lattice layout)
                let slice: Vec<String> =
                    lattice.spins[..n2].iter().map(|&s| s.to_string()).collect();
                snap_csv.push_str(&slice.join(","));
                snap_csv.push_str(&format!(",{t:.4}\n"));
            }
        }

        fs::write(&snap_path, &snap_csv).expect("failed to write snapshots");
        eprintln!("Wrote snapshots to {}", snap_path.display());
    }
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
        Geometry::Mesh => "Mesh",
    }
}

fn expected_ground(g: Geometry) -> &'static str {
    match g {
        Geometry::Square2D => "-2.0 J",
        Geometry::Triangular2D | Geometry::Cubic3D => "-3.0 J",
        Geometry::Mesh => "N/A",
    }
}

fn expected_tc(g: Geometry) -> &'static str {
    match g {
        Geometry::Square2D => "2.27 J/kB (Onsager exact)",
        Geometry::Triangular2D => "3.64 J/kB",
        Geometry::Cubic3D => "4.51 J/kB",
        Geometry::Mesh => "N/A",
    }
}

fn parse_args(args: &[String]) -> (SweepConfig, String, bool) {
    let mut cfg = SweepConfig::default();
    let mut outdir = String::from(".");
    let mut save_snapshots = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => {
                cfg.n = parse_arg(args, i, "--n");
                i += 2;
            }
            "--geometry" => {
                cfg.geometry = parse_geometry(args, i);
                i += 2;
            }
            "--j" => {
                cfg.j = parse_arg(args, i, "--j");
                i += 2;
            }
            "--h" => {
                cfg.h = parse_arg(args, i, "--h");
                i += 2;
            }
            "--warmup" => {
                cfg.warmup_sweeps = parse_arg(args, i, "--warmup");
                i += 2;
            }
            "--samples" => {
                cfg.sample_sweeps = parse_arg(args, i, "--samples");
                i += 2;
            }
            "--seed" => {
                cfg.seed = parse_arg(args, i, "--seed");
                i += 2;
            }
            "--tmin" => {
                cfg.t_min = parse_arg(args, i, "--tmin");
                i += 2;
            }
            "--tmax" => {
                cfg.t_max = parse_arg(args, i, "--tmax");
                i += 2;
            }
            "--steps" => {
                cfg.t_steps = parse_arg(args, i, "--steps");
                i += 2;
            }
            "--wolff" => {
                cfg.algorithm = Algorithm::Wolff;
                i += 1;
            }
            "--outdir" => {
                outdir = get_arg(args, i, "--outdir");
                i += 2;
            }
            "--save-snapshots" => {
                save_snapshots = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    validate_lattice_size(cfg.n);
    validate_t_steps(cfg.t_steps);
    validate_temp_range(cfg.t_min, cfg.t_max);
    validate_samples(cfg.sample_sweeps, "--samples");
    validate_samples(cfg.warmup_sweeps, "--warmup");

    (cfg, outdir, save_snapshots)
}
