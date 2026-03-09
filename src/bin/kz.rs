/// CLI: run Kibble-Zurek quench experiment.
///
/// Usage:
///   cargo run --release --bin kz -- --n 20 --trials 10 \
///     --tau-min 100 --tau-max 100000 --tau-steps 20
///
/// Output columns: tau_q,rho,rho_err,n_trials
use std::env;
use std::fs;
use std::path::Path;

use ising::cli::{
    check_help, get_arg, parse_arg, parse_geometry, validate_lattice_size, warn_unknown_flags,
};
use ising::kibble_zurek::{run_kz_sweep, KzProtocol};
use ising::lattice::Geometry;

const USAGE: &str = "\
kz — Kibble-Zurek quench experiment

USAGE:
    kz [OPTIONS]

OPTIONS:
    --n <N>              Lattice size per dimension [default: 20]
    --geometry <TYPE>    square, triangular, or cubic [default: cubic]
    --j <J>              Coupling constant [default: 1.0]
    --t-start <T>        Starting temperature (high T) [default: 6.0]
    --t-end <T>          Ending temperature (low T) [default: 1.0]
    --tau-min <N>        Minimum quench time [default: 100]
    --tau-max <N>        Maximum quench time [default: 100000]
    --tau-steps <N>      Number of tau_q values (>=2, log-spaced) [default: 20]
    --trials <N>         Number of independent trials per tau_q [default: 5]
    --seed <N>           RNG seed [default: 42]
    --gpu                Use GPU acceleration (requires --features cuda)
    --outdir <DIR>       Output directory [default: analysis/data]
    --help, -h           Show this help message";

const KNOWN_FLAGS: &[&str] = &[
    "--n",
    "--geometry",
    "--j",
    "--t-start",
    "--t-end",
    "--tau-min",
    "--tau-max",
    "--tau-steps",
    "--trials",
    "--seed",
    "--gpu",
    "--outdir",
    "--help",
];

fn main() {
    let args: Vec<String> = env::args().collect();
    check_help(&args, USAGE);
    warn_unknown_flags(&args, KNOWN_FLAGS);

    let mut n: usize = 20;
    let mut geometry = Geometry::Cubic3D;
    let mut j: f64 = 1.0;
    let mut t_start: f64 = 6.0;
    let mut t_end: f64 = 1.0;
    let mut tau_min: usize = 100;
    let mut tau_max: usize = 100_000;
    let mut tau_steps: usize = 20;
    let mut n_trials: usize = 5;
    let mut warmup_sweeps: usize = 200;
    let mut freeze_sweeps: usize = 0;
    let mut freeze_temperature: f64 = 0.01;
    let mut seed: u64 = 42;
    let mut outdir = String::from("analysis/data");
    let mut use_gpu = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => {
                n = parse_arg(&args, i, "--n");
                i += 2;
            }
            "--geometry" => {
                geometry = parse_geometry(&args, i);
                i += 2;
            }
            "--j" => {
                j = parse_arg(&args, i, "--j");
                i += 2;
            }
            "--t-start" => {
                t_start = parse_arg(&args, i, "--t-start");
                i += 2;
            }
            "--t-end" => {
                t_end = parse_arg(&args, i, "--t-end");
                i += 2;
            }
            "--tau-min" => {
                tau_min = parse_arg(&args, i, "--tau-min");
                i += 2;
            }
            "--tau-max" => {
                tau_max = parse_arg(&args, i, "--tau-max");
                i += 2;
            }
            "--tau-steps" => {
                tau_steps = parse_arg(&args, i, "--tau-steps");
                i += 2;
            }
            "--trials" => {
                n_trials = parse_arg(&args, i, "--trials");
                i += 2;
            }
            "--warmup-sweeps" => {
                warmup_sweeps = parse_arg(&args, i, "--warmup-sweeps");
                i += 2;
            }
            "--freeze-sweeps" => {
                freeze_sweeps = parse_arg(&args, i, "--freeze-sweeps");
                i += 2;
            }
            "--freeze-temperature" => {
                freeze_temperature = parse_arg(&args, i, "--freeze-temperature");
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
    validate_lattice_size(n);
    if tau_steps == 0 {
        eprintln!("Error: --tau-steps must be at least 1");
        std::process::exit(1);
    }
    if n_trials == 0 {
        eprintln!("Error: --trials must be at least 1");
        std::process::exit(1);
    }
    if t_start <= 0.0 || t_end <= 0.0 || freeze_temperature <= 0.0 {
        eprintln!("Error: temperatures must be positive");
        std::process::exit(1);
    }
    if tau_min == 0 || tau_max == 0 || tau_min > tau_max {
        eprintln!("Error: --tau-min must be >= 1 and <= --tau-max");
        std::process::exit(1);
    }
    if use_gpu && geometry != Geometry::Cubic3D {
        eprintln!("Error: --gpu KZ currently supports only --geometry cubic");
        std::process::exit(1);
    }

    // Build log-spaced tau_q values
    let tau_q_values: Vec<usize> = if tau_steps == 1 {
        vec![tau_min]
    } else {
        (0..tau_steps)
            .map(|k| {
                let log_min = (tau_min as f64).ln();
                let log_max = (tau_max as f64).ln();
                let log_t = log_min + (log_max - log_min) * k as f64 / (tau_steps - 1) as f64;
                log_t.exp().round() as usize
            })
            .collect()
    };

    let protocol = KzProtocol {
        warmup_sweeps,
        freeze_sweeps,
        freeze_temperature,
    };

    let backend = if use_gpu { "GPU" } else { "CPU" };
    eprintln!(
        "KZ sweep [{backend}]: N={n}, tau_q={tau_min}..{tau_max} ({tau_steps} steps, log-spaced), {n_trials} trials each"
    );
    eprintln!(
        "  Ramp: T_start={t_start} -> T_end={t_end}; {}",
        protocol.describe()
    );

    let results = if use_gpu {
        #[cfg(feature = "cuda")]
        {
            ising::cuda::kz_gpu::run_kz_sweep_gpu(
                n,
                geometry,
                j,
                t_start,
                t_end,
                &tau_q_values,
                n_trials,
                protocol,
                seed,
            )
            .unwrap_or_else(|err| {
                eprintln!("Error: {err}");
                std::process::exit(1);
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("Error: --gpu requires the 'cuda' feature");
            std::process::exit(1);
        }
    } else {
        run_kz_sweep(
            n,
            geometry,
            j,
            t_start,
            t_end,
            &tau_q_values,
            n_trials,
            protocol,
            seed,
        )
    };

    fs::create_dir_all(&outdir).expect("failed to create outdir");
    let fname = format!("kz_N{n}.csv");
    let path = Path::new(&outdir).join(&fname);
    let mut csv = String::from("tau_q,rho,rho_err,n_trials\n");
    for point in &results {
        csv.push_str(&format!(
            "{},{:.8},{:.8},{}\n",
            point.tau_q, point.rho, point.rho_err, point.n_trials
        ));
    }
    fs::write(&path, &csv).expect("failed to write CSV");
    eprintln!("Wrote {}", path.display());
}
