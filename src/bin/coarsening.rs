/// CLI: run a quench experiment and output domain wall density vs time.
///
/// Usage:
///   cargo run --release --bin coarsening
///   cargo run --release --bin coarsening -- --n 30 --t-quench 2.5 --steps 200000
///   cargo run --release --features cuda --bin coarsening -- --n 30 --t-quench 2.5 --steps 200000 --gpu
///
/// Output columns: t,rho
use ising::cli::{
    check_help, get_arg, parse_arg, parse_geometry, validate_lattice_size, warn_unknown_flags,
};
use ising::coarsening::{run_coarsening, CoarseningConfig};
use std::env;
use std::fs;
use std::path::Path;

const USAGE: &str = "\
coarsening — Domain coarsening after quench

USAGE:
    coarsening [OPTIONS]

OPTIONS:
    --n <N>              Lattice size per dimension [default: 30]
    --geometry <TYPE>    square, triangular, or cubic [default: cubic]
    --j <J>              Coupling constant [default: 1.0]
    --t-high <T>         Initial high temperature [default: 10.0]
    --t-quench <T>       Quench temperature [default: 2.5]
    --warmup <N>         Warmup sweeps at T_high [default: 500]
    --steps <N>          Total Monte Carlo steps after quench [default: 100000]
    --sample-every <N>   Measure domain walls every N steps [default: 100]
    --seed <N>           RNG seed [default: 42]
    --gpu                Use GPU acceleration (requires --features cuda)
    --outdir <DIR>       Output directory [default: analysis/data]
    --help, -h           Show this help message";

const KNOWN_FLAGS: &[&str] = &[
    "--n",
    "--geometry",
    "--j",
    "--t-high",
    "--t-quench",
    "--warmup",
    "--steps",
    "--sample-every",
    "--seed",
    "--gpu",
    "--outdir",
    "--help",
];

fn main() {
    let args: Vec<String> = env::args().collect();
    check_help(&args, USAGE);
    warn_unknown_flags(&args, KNOWN_FLAGS);

    let mut config = CoarseningConfig::default();
    let mut outdir = String::from("analysis/data");

    let mut use_gpu = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => {
                config.n = parse_arg(&args, i, "--n");
                i += 2;
            }
            "--geometry" => {
                config.geometry = parse_geometry(&args, i);
                i += 2;
            }
            "--j" => {
                config.j = parse_arg(&args, i, "--j");
                i += 2;
            }
            "--t-high" => {
                config.t_high = parse_arg(&args, i, "--t-high");
                i += 2;
            }
            "--t-quench" => {
                config.t_quench = parse_arg(&args, i, "--t-quench");
                i += 2;
            }
            "--warmup" => {
                config.warmup_sweeps = parse_arg(&args, i, "--warmup");
                i += 2;
            }
            "--steps" => {
                config.total_steps = parse_arg(&args, i, "--steps");
                i += 2;
            }
            "--sample-every" => {
                config.sample_every = parse_arg(&args, i, "--sample-every");
                i += 2;
            }
            "--seed" => {
                config.seed = parse_arg(&args, i, "--seed");
                i += 2;
            }
            "--outdir" => {
                outdir = get_arg(&args, i, "--outdir");
                i += 2;
            }
            "--gpu" => {
                use_gpu = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    // Validation
    validate_lattice_size(config.n);
    if config.t_quench <= 0.0 || config.t_high <= 0.0 {
        eprintln!("Error: temperatures must be positive");
        std::process::exit(1);
    }
    if config.total_steps == 0 {
        eprintln!("Error: --steps must be at least 1");
        std::process::exit(1);
    }
    if config.sample_every == 0 {
        eprintln!("Error: --sample-every must be at least 1");
        std::process::exit(1);
    }

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    eprintln!(
        "Coarsening: N={}, geometry={:?}, T_quench={}, steps={}{}",
        config.n,
        config.geometry,
        config.t_quench,
        config.total_steps,
        if use_gpu { " [GPU]" } else { "" }
    );

    let results = if use_gpu {
        run_coarsening_with_gpu(&config)
    } else {
        run_coarsening(&config)
    };

    let fname = format!("coarsening_N{}_T{:.2}.csv", config.n, config.t_quench);
    let path = Path::new(&outdir).join(&fname);
    let mut csv = String::from("t,rho\n");
    for pt in &results {
        csv.push_str(&format!("{},{:.8}\n", pt.step, pt.rho));
    }
    fs::write(&path, &csv).expect("failed to write CSV");
    eprintln!("Wrote {}", path.display());
}

fn run_coarsening_with_gpu(config: &CoarseningConfig) -> Vec<ising::coarsening::CoarseningPoint> {
    #[cfg(feature = "cuda")]
    {
        eprintln!("GPU mode: CUDA checkerboard Metropolis");
        ising::cuda::coarsening_gpu::run_coarsening_gpu(config)
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("Warning: --gpu specified but binary was not compiled with --features cuda");
        eprintln!("Falling back to CPU");
        ising::coarsening::run_coarsening(config)
    }
}
