/// CLI: temperature sweep on an arbitrary graph loaded from file.
///
/// Usage:
///   cargo run --release --bin mesh_sweep -- --graph graphs/diluted.json --j 1.0
///
/// Output columns: T,E,M,M2,M4,Cv,chi
use std::env;
use std::fs;
use std::path::Path;

use ising::graph::GraphDef;
use ising::metropolis::warm_up;
use ising::observables::measure;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn get_arg(args: &[String], i: usize, flag: &str) -> String {
    if i + 1 >= args.len() {
        eprintln!("Error: {} requires a value", flag);
        std::process::exit(1);
    }
    args[i + 1].clone()
}

fn main() {
    let args: Vec<String> = env::args().collect();

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
                j = get_arg(&args, i, "--j").parse().unwrap();
                i += 2;
            }
            "--tmin" => {
                t_min = get_arg(&args, i, "--tmin").parse().unwrap();
                i += 2;
            }
            "--tmax" => {
                t_max = get_arg(&args, i, "--tmax").parse().unwrap();
                i += 2;
            }
            "--steps" => {
                t_steps = get_arg(&args, i, "--steps").parse().unwrap();
                i += 2;
            }
            "--warmup" => {
                warmup = get_arg(&args, i, "--warmup").parse().unwrap();
                i += 2;
            }
            "--samples" => {
                samples = get_arg(&args, i, "--samples").parse().unwrap();
                i += 2;
            }
            "--seed" => {
                seed = get_arg(&args, i, "--seed").parse().unwrap();
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

    if graph_path.is_empty() {
        eprintln!("Error: --graph <path> is required");
        std::process::exit(1);
    }

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

    let mut csv = String::from("T,E,M,M2,M4,Cv,chi\n");

    for step in 0..t_steps {
        let t = t_min + (t_max - t_min) * step as f64 / (t_steps - 1) as f64;
        let beta = 1.0 / t;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add((t * 1000.0) as u64));
        lattice.randomise(&mut rng);
        warm_up(&mut lattice, beta, j, 0.0, warmup, &mut rng);
        let obs = measure(&mut lattice, beta, j, 0.0, samples, &mut rng);
        csv.push_str(&format!(
            "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            t, obs.energy, obs.magnetisation, obs.m2, obs.m4, obs.heat_capacity, obs.susceptibility
        ));
        eprintln!("T={t:.3}: M={:.4}", obs.magnetisation);
    }

    fs::write(&path, &csv).expect("failed to write CSV");
    eprintln!("Wrote {}", path.display());
}
