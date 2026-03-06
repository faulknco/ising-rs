/// CLI: run Kibble-Zurek quench experiment.
///
/// Usage:
///   cargo run --release --bin kz -- --n 20 --trials 10 \
///     --tau-min 100 --tau-max 100000 --tau-steps 20
///
/// Output columns: tau_q,rho
use std::env;
use std::fs;
use std::path::Path;

use ising::kibble_zurek::run_kz_sweep;
use ising::lattice::Geometry;

fn get_arg(args: &[String], i: usize, flag: &str) -> String {
    if i + 1 >= args.len() {
        eprintln!("Error: {} requires a value", flag);
        std::process::exit(1);
    }
    args[i + 1].clone()
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut n: usize = 20;
    let mut geometry = Geometry::Cubic3D;
    let mut j: f64 = 1.0;
    let mut t_start: f64 = 6.0;
    let mut t_end: f64 = 1.0;
    let mut tau_min: usize = 100;
    let mut tau_max: usize = 100_000;
    let mut tau_steps: usize = 20;
    let mut n_trials: usize = 5;
    let mut seed: u64 = 42;
    let mut outdir = String::from("analysis/data");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n"         => { n = get_arg(&args, i, "--n").parse().unwrap(); i += 2; }
            "--geometry"  => {
                geometry = match get_arg(&args, i, "--geometry").as_str() {
                    "cubic"      => Geometry::Cubic3D,
                    "triangular" => Geometry::Triangular2D,
                    _            => Geometry::Square2D,
                };
                i += 2;
            }
            "--j"         => { j = get_arg(&args, i, "--j").parse().unwrap(); i += 2; }
            "--t-start"   => { t_start = get_arg(&args, i, "--t-start").parse().unwrap(); i += 2; }
            "--t-end"     => { t_end = get_arg(&args, i, "--t-end").parse().unwrap(); i += 2; }
            "--tau-min"   => { tau_min = get_arg(&args, i, "--tau-min").parse().unwrap(); i += 2; }
            "--tau-max"   => { tau_max = get_arg(&args, i, "--tau-max").parse().unwrap(); i += 2; }
            "--tau-steps" => { tau_steps = get_arg(&args, i, "--tau-steps").parse().unwrap(); i += 2; }
            "--trials"    => { n_trials = get_arg(&args, i, "--trials").parse().unwrap(); i += 2; }
            "--seed"      => { seed = get_arg(&args, i, "--seed").parse().unwrap(); i += 2; }
            "--outdir"    => { outdir = get_arg(&args, i, "--outdir"); i += 2; }
            _             => { i += 1; }
        }
    }

    // Build log-spaced tau_q values
    let tau_q_values: Vec<usize> = (0..tau_steps).map(|k| {
        let log_min = (tau_min as f64).ln();
        let log_max = (tau_max as f64).ln();
        let log_t = log_min + (log_max - log_min) * k as f64 / (tau_steps - 1) as f64;
        log_t.exp().round() as usize
    }).collect();

    eprintln!(
        "KZ sweep: N={n}, tau_q={tau_min}..{tau_max} ({tau_steps} steps, log-spaced), {n_trials} trials each"
    );

    let results = run_kz_sweep(n, geometry, j, t_start, t_end, &tau_q_values, n_trials, seed);

    fs::create_dir_all(&outdir).expect("failed to create outdir");
    let fname = format!("kz_N{n}.csv");
    let path = Path::new(&outdir).join(&fname);
    let mut csv = String::from("tau_q,rho\n");
    for (tau_q, rho) in &results {
        csv.push_str(&format!("{tau_q},{rho:.8}\n"));
    }
    fs::write(&path, &csv).expect("failed to write CSV");
    eprintln!("Wrote {}", path.display());
}
