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

fn get_arg(args: &[String], i: usize, flag: &str) -> String {
    if i + 1 >= args.len() {
        eprintln!("Error: {} requires a value", flag);
        std::process::exit(1);
    }
    args[i + 1].clone()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = FssConfig::default();
    let mut outdir = String::from("analysis/data");

    let mut use_gpu = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sizes" => {
                config.sizes = get_arg(&args, i, "--sizes").split(',')
                    .filter_map(|s| s.parse().ok())
                    .collect();
                i += 2;
            }
            "--geometry" => {
                config.geometry = match get_arg(&args, i, "--geometry").as_str() {
                    "cubic"      => Geometry::Cubic3D,
                    "triangular" => Geometry::Triangular2D,
                    _            => Geometry::Square2D,
                };
                i += 2;
            }
            "--j"       => { config.j = get_arg(&args, i, "--j").parse().unwrap(); i += 2; }
            "--warmup"  => { config.warmup_sweeps = get_arg(&args, i, "--warmup").parse().unwrap(); i += 2; }
            "--samples" => { config.sample_sweeps = get_arg(&args, i, "--samples").parse().unwrap(); i += 2; }
            "--tmin"    => { config.t_min = get_arg(&args, i, "--tmin").parse().unwrap(); i += 2; }
            "--tmax"    => { config.t_max = get_arg(&args, i, "--tmax").parse().unwrap(); i += 2; }
            "--steps"   => { config.t_steps = get_arg(&args, i, "--steps").parse().unwrap(); i += 2; }
            "--seed"    => { config.seed = get_arg(&args, i, "--seed").parse().unwrap(); i += 2; }
            "--wolff"   => { config.algorithm = Algorithm::Wolff; i += 1; }
            "--outdir"  => { outdir = get_arg(&args, i, "--outdir"); i += 2; }
            "--gpu"     => { use_gpu = true; i += 1; }
            _           => { i += 1; }
        }
    }

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    let results = run_fss(&config);

    #[cfg(feature = "cuda")]
    if use_gpu {
        eprintln!("GPU mode: CUDA checkerboard Metropolis (RTX 2060 target)");
        eprintln!("Note: full GPU FSS path not yet implemented — using CPU");
    }
    #[cfg(not(feature = "cuda"))]
    if use_gpu {
        eprintln!("Warning: --gpu specified but binary was not compiled with --features cuda");
    }

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
