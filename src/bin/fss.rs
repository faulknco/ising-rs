/// CLI: run finite-size scaling sweeps for multiple lattice sizes.
///
/// Usage:
///   cargo run --release --bin fss
///   cargo run --release --bin fss -- --sizes 8,12,16,20 --wolff --outdir analysis/data
///
/// Output: one CSV per size at <outdir>/fss_N<n>.csv
/// Columns: T,E,M,M2,M4,Cv,chi

use std::env;
use std::fs;
use std::path::Path;
use ising::fss::{FssConfig, run_fss};
use ising::lattice::Geometry;
use ising::sweep::Algorithm;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = FssConfig::default();
    let mut outdir = String::from("analysis/data");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sizes" => {
                config.sizes = args[i+1].split(',')
                    .filter_map(|s| s.parse().ok())
                    .collect();
                i += 2;
            }
            "--geometry" => {
                config.geometry = match args[i+1].as_str() {
                    "cubic"      => Geometry::Cubic3D,
                    "triangular" => Geometry::Triangular2D,
                    _            => Geometry::Square2D,
                };
                i += 2;
            }
            "--j"       => { config.j = args[i+1].parse().unwrap(); i += 2; }
            "--warmup"  => { config.warmup_sweeps = args[i+1].parse().unwrap(); i += 2; }
            "--samples" => { config.sample_sweeps = args[i+1].parse().unwrap(); i += 2; }
            "--tmin"    => { config.t_min = args[i+1].parse().unwrap(); i += 2; }
            "--tmax"    => { config.t_max = args[i+1].parse().unwrap(); i += 2; }
            "--steps"   => { config.t_steps = args[i+1].parse().unwrap(); i += 2; }
            "--seed"    => { config.seed = args[i+1].parse().unwrap(); i += 2; }
            "--wolff"   => { config.algorithm = Algorithm::Wolff; i += 1; }
            "--outdir"  => { outdir = args[i+1].clone(); i += 2; }
            _           => { i += 1; }
        }
    }

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    let results = run_fss(&config);

    for (n, obs_list) in &results {
        let path = Path::new(&outdir).join(format!("fss_N{n}.csv"));
        let mut csv = String::from("T,E,M,M2,M4,Cv,chi\n");
        for o in obs_list {
            csv.push_str(&format!(
                "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                o.temperature, o.energy, o.magnetisation,
                o.m2, o.m4, o.heat_capacity, o.susceptibility
            ));
        }
        fs::write(&path, &csv).expect("failed to write CSV");
        eprintln!("Wrote {}", path.display());
    }
}
