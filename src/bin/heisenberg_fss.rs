use ising::heisenberg::fss::{run_heisenberg_fss, HeisFssConfig};
/// CLI: run Heisenberg FSS for multiple lattice sizes.
///
/// Usage:
///   cargo run --release --bin heisenberg_fss
///   cargo run --release --bin heisenberg_fss -- --sizes 8,12,16,20 --outdir analysis/data
///
/// Output: one CSV per size at <outdir>/heisenberg_fss_N<n>.csv
/// Columns: T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err
use std::env;
use std::fs;
use std::path::Path;

fn get_arg(args: &[String], i: usize, flag: &str) -> String {
    if i + 1 >= args.len() {
        eprintln!("Error: {flag} requires a value");
        std::process::exit(1);
    }
    args[i + 1].clone()
}

fn parse_flag<T: std::str::FromStr>(args: &[String], i: usize, flag: &str) -> T
where
    T::Err: std::fmt::Display,
{
    match get_arg(args, i, flag).parse::<T>() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: invalid value for {flag}: {e}");
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = HeisFssConfig::default();
    let mut outdir = String::from("analysis/data");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sizes" => {
                config.sizes = get_arg(&args, i, "--sizes")
                    .split(',')
                    .map(|s| {
                        s.parse::<usize>().unwrap_or_else(|_| {
                            eprintln!("Error: invalid size value: {s}");
                            std::process::exit(1);
                        })
                    })
                    .collect();
                i += 2;
            }
            "--tmin" => {
                config.t_min = parse_flag::<f64>(&args, i, "--tmin");
                i += 2;
            }
            "--tmax" => {
                config.t_max = parse_flag::<f64>(&args, i, "--tmax");
                i += 2;
            }
            "--steps" => {
                config.t_steps = parse_flag::<usize>(&args, i, "--steps");
                i += 2;
            }
            "--warmup" => {
                config.warmup_sweeps = parse_flag::<usize>(&args, i, "--warmup");
                i += 2;
            }
            "--samples" => {
                config.sample_sweeps = parse_flag::<usize>(&args, i, "--samples");
                i += 2;
            }
            "--overrelax" => {
                config.n_overrelax = parse_flag::<usize>(&args, i, "--overrelax");
                i += 2;
            }
            "--delta" => {
                config.delta = parse_flag::<f64>(&args, i, "--delta");
                i += 2;
            }
            "--seed" => {
                config.seed = parse_flag::<u64>(&args, i, "--seed");
                i += 2;
            }
            "--j" => {
                config.j = parse_flag::<f64>(&args, i, "--j");
                i += 2;
            }
            "--outdir" => {
                outdir = get_arg(&args, i, "--outdir");
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    let results = run_heisenberg_fss(&config);

    for (n, obs_list) in &results {
        let path = Path::new(&outdir).join(format!("heisenberg_fss_N{n}.csv"));
        let mut csv = String::from("T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err\n");
        for o in obs_list {
            csv.push_str(&format!(
                "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                o.temperature,
                o.energy,
                o.energy_err,
                o.magnetisation,
                o.magnetisation_err,
                o.m2,
                o.m2_err,
                o.m4,
                o.m4_err,
                o.heat_capacity,
                o.heat_capacity_err,
                o.susceptibility,
                o.susceptibility_err,
            ));
        }
        fs::write(&path, &csv).expect("failed to write CSV");
        eprintln!("Wrote {}", path.display());
    }
}
