/// CLI: temperature sweep on an arbitrary graph loaded from file.
///
/// Usage:
///   cargo run --release --bin mesh_sweep -- --graph graphs/diluted.json --j 1.0
///
/// Output columns: T,E,M,M2,M4,Cv,chi
use std::env;
use std::fs;
use std::path::Path;

use ising::cli::{
    check_help, get_arg, parse_arg, validate_samples, validate_t_steps, validate_temp_range,
    warn_unknown_flags,
};
use ising::graph::GraphDef;
use ising::metropolis::warm_up;
use ising::observables::measure;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const USAGE: &str = "\
mesh_sweep — Temperature sweep on an arbitrary graph

USAGE:
    mesh_sweep --graph <PATH> [OPTIONS]

OPTIONS:
    --graph <PATH>       Path to graph file (JSON or edge CSV) [required]
    --j <J>              Coupling constant [default: 1.0]
    --tmin <T>           Minimum temperature [default: 3.5]
    --tmax <T>           Maximum temperature [default: 5.5]
    --steps <N>          Number of temperature points (>=2) [default: 41]
    --warmup <N>         Warmup sweeps per temperature [default: 2000]
    --samples <N>        Measurement sweeps per temperature [default: 1000]
    --seed <N>           RNG seed [default: 42]
    --outdir <DIR>       Output directory [default: analysis/data]
    --prefix <STR>       Output filename prefix [default: graph filename stem]
    --help, -h           Show this help message";

const KNOWN_FLAGS: &[&str] = &[
    "--graph",
    "--j",
    "--tmin",
    "--tmax",
    "--steps",
    "--warmup",
    "--samples",
    "--seed",
    "--outdir",
    "--prefix",
    "--help",
];

fn main() {
    let args: Vec<String> = env::args().collect();
    check_help(&args, USAGE);
    warn_unknown_flags(&args, KNOWN_FLAGS);

    let mut graph_path = String::new();
    let mut j: f64 = 1.0;
    let mut t_min: f64 = 3.5;
    let mut t_max: f64 = 5.5;
    let mut t_steps: usize = 41;
    let mut warmup: usize = 2000;
    let mut samples: usize = 1000;
    let mut seed: u64 = 42;
    let mut outdir = String::from("analysis/data");
    let mut out_prefix = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--graph" => {
                graph_path = get_arg(&args, i, "--graph");
                i += 2;
            }
            "--j" => {
                j = parse_arg(&args, i, "--j");
                i += 2;
            }
            "--tmin" => {
                t_min = parse_arg(&args, i, "--tmin");
                i += 2;
            }
            "--tmax" => {
                t_max = parse_arg(&args, i, "--tmax");
                i += 2;
            }
            "--steps" => {
                t_steps = parse_arg(&args, i, "--steps");
                i += 2;
            }
            "--warmup" => {
                warmup = parse_arg(&args, i, "--warmup");
                i += 2;
            }
            "--samples" => {
                samples = parse_arg(&args, i, "--samples");
                i += 2;
            }
            "--seed" => {
                seed = parse_arg(&args, i, "--seed");
                i += 2;
            }
            "--outdir" => {
                outdir = get_arg(&args, i, "--outdir");
                i += 2;
            }
            "--prefix" => {
                out_prefix = get_arg(&args, i, "--prefix");
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    // Validation
    if graph_path.is_empty() {
        eprintln!("Error: --graph <path> is required (try --help for usage)");
        std::process::exit(1);
    }
    validate_t_steps(t_steps);
    validate_temp_range(t_min, t_max);
    validate_samples(samples, "--samples");
    validate_samples(warmup, "--warmup");

    let content = fs::read_to_string(&graph_path).unwrap_or_else(|e| {
        eprintln!("Cannot read {graph_path}: {e}");
        std::process::exit(1);
    });
    let gdef = if graph_path.ends_with(".json") {
        GraphDef::from_json(&content)
    } else {
        GraphDef::from_edge_csv(&content)
    }
    .unwrap_or_else(|e| {
        eprintln!("Parse error: {e}");
        std::process::exit(1);
    });

    let n_nodes = gdef.n_nodes;
    eprintln!("Graph: {n_nodes} nodes, {} edges", gdef.edges.len());

    let mut lattice = gdef.into_lattice();

    fs::create_dir_all(&outdir).expect("failed to create outdir");
    let prefix = if out_prefix.is_empty() {
        Path::new(&graph_path)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string()
    } else {
        out_prefix
    };
    let path = Path::new(&outdir).join(format!("{prefix}_sweep.csv"));

    // Single RNG seeded once; randomise lattice once before annealing.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    lattice.randomise(&mut rng);

    // Anneal from high to low temperature (keeps spin correlations between steps).
    let mut rows: Vec<String> = Vec::with_capacity(t_steps);
    for step in 0..t_steps {
        let t = t_max - (t_max - t_min) * step as f64 / (t_steps - 1) as f64;
        let beta = 1.0 / t;
        warm_up(&mut lattice, beta, j, 0.0, warmup, &mut rng);
        let obs = measure(&mut lattice, beta, j, 0.0, samples, &mut rng);
        rows.push(format!(
            "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            t, obs.energy, obs.magnetisation, obs.m2, obs.m4, obs.heat_capacity, obs.susceptibility
        ));
        eprintln!("T={t:.3}: M={:.4}", obs.magnetisation);
    }

    // Write CSV sorted low-to-high T (reverse of annealing order).
    rows.reverse();
    let mut csv = String::from("T,E,M,M2,M4,Cv,chi\n");
    for row in &rows {
        csv.push_str(row);
        csv.push('\n');
    }

    fs::write(&path, &csv).expect("failed to write CSV");
    eprintln!("Wrote {}", path.display());
}
