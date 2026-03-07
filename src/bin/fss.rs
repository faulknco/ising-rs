use ising::fss::{run_fss, FssConfig};
use ising::sweep::{run_raw, Algorithm, SweepConfig};
/// CLI: run finite-size scaling sweeps for multiple lattice sizes.
///
/// Usage:
///   cargo run --release --bin fss
///   cargo run --release --bin fss -- --sizes 8,12,16,20 --wolff --outdir analysis/data
///   cargo run --release --features cuda --bin fss -- --sizes 8,12,16,20 --gpu --outdir analysis/data
///
/// Output: one CSV per size at <outdir>/fss_N<n>.csv
/// Columns: T,E,M,M2,M4,Cv,chi
use ising::cli::{get_arg, parse_arg, parse_geometry};
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = FssConfig::default();
    let mut outdir = String::from("analysis/data");

    let mut use_gpu = false;
    let mut raw_mode = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sizes" => {
                config.sizes = get_arg(&args, i, "--sizes")
                    .split(',')
                    .filter_map(|s| s.parse().ok())
                    .collect();
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
            "--warmup" => {
                config.warmup_sweeps = parse_arg(&args, i, "--warmup");
                i += 2;
            }
            "--samples" => {
                config.sample_sweeps = parse_arg(&args, i, "--samples");
                i += 2;
            }
            "--tmin" => {
                config.t_min = parse_arg(&args, i, "--tmin");
                i += 2;
            }
            "--tmax" => {
                config.t_max = parse_arg(&args, i, "--tmax");
                i += 2;
            }
            "--steps" => {
                config.t_steps = parse_arg(&args, i, "--steps");
                i += 2;
            }
            "--seed" => {
                config.seed = parse_arg(&args, i, "--seed");
                i += 2;
            }
            "--wolff" => {
                config.algorithm = Algorithm::Wolff;
                i += 1;
            }
            "--outdir" => {
                outdir = get_arg(&args, i, "--outdir");
                i += 2;
            }
            "--gpu" => {
                use_gpu = true;
                i += 1;
            }
            "--raw" => {
                raw_mode = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    // Raw time-series mode for histogram reweighting
    if raw_mode {
        for &n in &config.sizes {
            eprintln!("FSS raw: N={n}");
            let sweep_cfg = SweepConfig {
                n,
                geometry: config.geometry,
                j: config.j,
                h: config.h,
                t_min: config.t_min,
                t_max: config.t_max,
                t_steps: config.t_steps,
                warmup_sweeps: config.warmup_sweeps,
                sample_sweeps: config.sample_sweeps,
                seed: config.seed.wrapping_add(n as u64),
                algorithm: Algorithm::Wolff,
            };
            let raw_data = run_raw(&sweep_cfg);
            let path = Path::new(&outdir).join(format!("fss_raw_N{n}.csv"));
            let mut csv = String::from("T,sample,e,m_abs,m_signed\n");
            for raw in &raw_data {
                for (i, ((e, ma), ms)) in raw
                    .e_per_spin
                    .iter()
                    .zip(raw.m_abs.iter())
                    .zip(raw.m_signed.iter())
                    .enumerate()
                {
                    csv.push_str(&format!(
                        "{:.6},{},{:.8},{:.8},{:.8}\n",
                        raw.temperature, i, e, ma, ms
                    ));
                }
            }
            fs::write(&path, &csv).expect("failed to write raw CSV");
            eprintln!("Wrote {}", path.display());
        }
        return;
    }

    let results = if use_gpu {
        run_fss_with_gpu(&config)
    } else {
        run_fss(&config)
    };

    for (n, obs_list) in &results {
        let path = Path::new(&outdir).join(format!("fss_N{n}.csv"));
        let mut csv = String::from("T,E,M,M2,M4,Cv,chi\n");
        for o in obs_list {
            csv.push_str(&format!(
                "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                o.temperature,
                o.energy,
                o.magnetisation,
                o.m2,
                o.m4,
                o.heat_capacity,
                o.susceptibility
            ));
        }
        fs::write(&path, &csv).expect("failed to write CSV");
        eprintln!("Wrote {}", path.display());
    }
}

fn run_fss_with_gpu(config: &FssConfig) -> Vec<(usize, Vec<ising::observables::Observables>)> {
    #[cfg(feature = "cuda")]
    {
        eprintln!("GPU mode: CUDA checkerboard Metropolis");
        ising::cuda::fss_gpu::run_fss_gpu(config)
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("Warning: --gpu specified but binary was not compiled with --features cuda");
        eprintln!("Falling back to CPU");
        ising::fss::run_fss(config)
    }
}
