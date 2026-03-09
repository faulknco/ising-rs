use ising::cli::{get_arg, parse_arg, validate_samples, validate_t_steps, validate_temp_range};
use ising::graph::GraphDef;
use ising::heisenberg::{observables::measure, HeisenbergLattice};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
/// CLI: run Heisenberg temperature sweep on a graph loaded from JSON.
///
/// Usage:
///   cargo run --release --bin heisenberg_jfit -- \
///     --graph analysis/graphs/bcc_N8.json \
///     --tmin 4.0 --tmax 9.0 --steps 41 \
///     --warmup 500 --samples 500 \
///     --outdir analysis/data
///
/// Output: <outdir>/heisenberg_jfit_<graphname>.csv
/// Columns: T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut graph_path = String::new();
    let mut outdir = String::from("analysis/data");
    let mut t_min = 4.0_f64;
    let mut t_max = 9.0_f64;
    let mut t_steps = 41usize;
    let mut warmup = 500usize;
    let mut samples = 500usize;
    let mut n_overrelax = 5usize;
    let mut delta = 0.5_f64;
    let mut j = 1.0_f64;
    let mut seed = 42u64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--graph" => {
                graph_path = get_arg(&args, i, "--graph");
                i += 2;
            }
            "--outdir" => {
                outdir = get_arg(&args, i, "--outdir");
                i += 2;
            }
            "--tmin" => {
                t_min = parse_arg::<f64>(&args, i, "--tmin");
                i += 2;
            }
            "--tmax" => {
                t_max = parse_arg::<f64>(&args, i, "--tmax");
                i += 2;
            }
            "--steps" => {
                t_steps = parse_arg::<usize>(&args, i, "--steps");
                i += 2;
            }
            "--warmup" => {
                warmup = parse_arg::<usize>(&args, i, "--warmup");
                i += 2;
            }
            "--samples" => {
                samples = parse_arg::<usize>(&args, i, "--samples");
                i += 2;
            }
            "--overrelax" => {
                n_overrelax = parse_arg::<usize>(&args, i, "--overrelax");
                i += 2;
            }
            "--delta" => {
                delta = parse_arg::<f64>(&args, i, "--delta");
                i += 2;
            }
            "--j" => {
                j = parse_arg::<f64>(&args, i, "--j");
                i += 2;
            }
            "--seed" => {
                seed = parse_arg::<u64>(&args, i, "--seed");
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    if graph_path.is_empty() {
        eprintln!("Error: --graph <path.json> is required");
        std::process::exit(1);
    }

    validate_t_steps(t_steps);
    validate_temp_range(t_min, t_max);
    validate_samples(warmup, "--warmup");
    validate_samples(samples, "--samples");

    let content = fs::read_to_string(&graph_path).unwrap_or_else(|e| {
        eprintln!("Error: failed to read graph file {graph_path}: {e}");
        std::process::exit(1);
    });
    let graph = GraphDef::from_json(&content).expect("failed to parse graph JSON");

    let graph_name = Path::new(&graph_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| {
            eprintln!("Error: could not determine graph name from path: {graph_path}");
            std::process::exit(1);
        })
        .to_string();

    eprintln!(
        "Heisenberg jfit: graph={graph_name}, N={}, T={t_min}..{t_max}",
        graph.n_nodes
    );

    // Build adjacency list from edge list
    let mut neighbours: Vec<Vec<usize>> = vec![vec![]; graph.n_nodes];
    for (a, b) in &graph.edges {
        neighbours[*a].push(*b);
        neighbours[*b].push(*a);
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut lat = HeisenbergLattice::new(neighbours);
    lat.randomise(&mut rng);

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    let path = Path::new(&outdir).join(format!("heisenberg_jfit_{graph_name}.csv"));
    let mut csv = String::from("T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err\n");

    for step in 0..t_steps {
        let t = t_min + (t_max - t_min) * step as f64 / (t_steps - 1) as f64;
        let beta = 1.0 / t;
        let obs = measure(
            &mut lat,
            beta,
            j,
            delta,
            n_overrelax,
            warmup,
            samples,
            &mut rng,
        );
        csv.push_str(&format!(
            "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            obs.temperature,
            obs.energy,
            obs.energy_err,
            obs.magnetisation,
            obs.magnetisation_err,
            obs.m2,
            obs.m2_err,
            obs.m4,
            obs.m4_err,
            obs.heat_capacity,
            obs.heat_capacity_err,
            obs.susceptibility,
            obs.susceptibility_err,
        ));
        eprintln!(
            "  T={t:.3} M={:.4}±{:.4}",
            obs.magnetisation, obs.magnetisation_err
        );
    }

    fs::write(&path, &csv).expect("failed to write CSV");
    eprintln!("Wrote {}", path.display());
}
