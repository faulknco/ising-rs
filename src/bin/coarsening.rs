/// CLI: run a quench experiment and output domain wall density vs time.
///
/// Usage:
///   cargo run --release --bin coarsening
///   cargo run --release --bin coarsening -- --n 30 --t-quench 0.5 --steps 50000
///
/// Output columns: t,rho

use std::env;
use std::fs;
use std::path::Path;
use ising::coarsening::{CoarseningConfig, run_coarsening};
use ising::lattice::Geometry;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = CoarseningConfig::default();
    let mut outdir = String::from("analysis/data");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n"            => { config.n = args[i+1].parse().unwrap(); i += 2; }
            "--geometry"     => {
                config.geometry = match args[i+1].as_str() {
                    "cubic"      => Geometry::Cubic3D,
                    "triangular" => Geometry::Triangular2D,
                    _            => Geometry::Square2D,
                };
                i += 2;
            }
            "--j"            => { config.j = args[i+1].parse().unwrap(); i += 2; }
            "--t-high"       => { config.t_high = args[i+1].parse().unwrap(); i += 2; }
            "--t-quench"     => { config.t_quench = args[i+1].parse().unwrap(); i += 2; }
            "--warmup"       => { config.warmup_sweeps = args[i+1].parse().unwrap(); i += 2; }
            "--steps"        => { config.total_steps = args[i+1].parse().unwrap(); i += 2; }
            "--sample-every" => { config.sample_every = args[i+1].parse().unwrap(); i += 2; }
            "--seed"         => { config.seed = args[i+1].parse().unwrap(); i += 2; }
            "--outdir"       => { outdir = args[i+1].clone(); i += 2; }
            _                => { i += 1; }
        }
    }

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    eprintln!(
        "Coarsening: N={}, geometry={:?}, T_quench={}, steps={}",
        config.n, config.geometry, config.t_quench, config.total_steps
    );

    let results = run_coarsening(&config);

    let fname = format!("coarsening_N{}_T{:.2}.csv", config.n, config.t_quench);
    let path = Path::new(&outdir).join(&fname);
    let mut csv = String::from("t,rho\n");
    for pt in &results {
        csv.push_str(&format!("{},{:.8}\n", pt.step, pt.rho));
    }
    fs::write(&path, &csv).expect("failed to write CSV");
    eprintln!("Wrote {}", path.display());
}
