/// GPU FSS with parallel tempering for Ising, XY, and Heisenberg models.
///
/// Usage:
///   cargo run --release --features cuda --bin gpu_fss -- \
///     --model ising --sizes 8,16,32,64 \
///     --tmin 4.4 --tmax 4.6 --replicas 20 \
///     --warmup 5000 --samples 100000 \
///     --exchange-every 10 --seed 42 \
///     --outdir analysis/data
///
/// Output per size:
///   gpu_fss_{model}_N{n}_summary.csv    — pre-averaged observables per temperature
///   gpu_fss_{model}_N{n}_timeseries.csv  — raw E, |M| per sweep per replica
use std::env;
use std::fs;
use std::path::Path;

use ising::cli::{get_arg, parse_arg, validate_samples, validate_temp_range};

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut model = String::from("ising");
    let mut algorithm = String::from("metropolis");
    let mut sizes_str = String::from("8,16,32,64");
    let mut t_min = 4.4_f64;
    let mut t_max = 4.6_f64;
    let mut n_replicas = 20usize;
    let mut warmup = 5000usize;
    let mut samples = 100000usize;
    let mut exchange_every = 10usize;
    let mut seed = 42u64;
    let mut outdir = String::from("analysis/data");
    let mut delta = 0.5_f64;
    let mut n_overrelax = 5usize;
    let mut measure_every = 1usize;
    let mut wolff_every = 10usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = get_arg(&args, i, "--model");
                i += 2;
            }
            "--algorithm" => {
                algorithm = get_arg(&args, i, "--algorithm");
                i += 2;
            }
            "--sizes" => {
                sizes_str = get_arg(&args, i, "--sizes");
                i += 2;
            }
            "--tmin" => {
                t_min = parse_arg(&args, i, "--tmin");
                i += 2;
            }
            "--tmax" => {
                t_max = parse_arg(&args, i, "--tmax");
                i += 2;
            }
            "--replicas" => {
                n_replicas = parse_arg(&args, i, "--replicas");
                i += 2;
            }
            "--warmup" => {
                warmup = parse_arg(&args, i, "--warmup");
                i += 2;
            }
            "--samples" => {
                samples = parse_arg(&args, i, "--samples");
                i += 2;
            }
            "--exchange-every" => {
                exchange_every = parse_arg(&args, i, "--exchange-every");
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
            "--delta" => {
                delta = parse_arg(&args, i, "--delta");
                i += 2;
            }
            "--overrelax" => {
                n_overrelax = parse_arg(&args, i, "--overrelax");
                i += 2;
            }
            "--measure-every" => {
                measure_every = parse_arg(&args, i, "--measure-every");
                i += 2;
            }
            "--wolff-every" => {
                wolff_every = parse_arg(&args, i, "--wolff-every");
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    let sizes: Vec<usize> = sizes_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    validate_temp_range(t_min, t_max);
    validate_samples(samples, "--samples");
    validate_samples(warmup, "--warmup");
    if measure_every == 0 {
        eprintln!("Error: --measure-every must be at least 1");
        std::process::exit(1);
    }

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    eprintln!("GPU FSS: model={model}, sizes={sizes:?}, T={t_min}..{t_max}, replicas={n_replicas}, algorithm={algorithm}");

    match model.as_str() {
        "ising" => run_ising_fss(
            &sizes,
            t_min,
            t_max,
            n_replicas,
            warmup,
            samples,
            exchange_every,
            seed,
            &outdir,
            measure_every,
            &algorithm,
        ),
        "xy" => run_continuous_fss(
            &sizes,
            2,
            t_min,
            t_max,
            n_replicas,
            warmup,
            samples,
            exchange_every,
            seed,
            &outdir,
            delta as f32,
            n_overrelax,
            measure_every,
            wolff_every,
        ),
        "heisenberg" => run_continuous_fss(
            &sizes,
            3,
            t_min,
            t_max,
            n_replicas,
            warmup,
            samples,
            exchange_every,
            seed,
            &outdir,
            delta as f32,
            n_overrelax,
            measure_every,
            wolff_every,
        ),
        _ => {
            eprintln!("Error: --model must be ising, xy, or heisenberg");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "cuda")]
fn run_ising_fss(
    sizes: &[usize],
    t_min: f64,
    t_max: f64,
    n_replicas: usize,
    warmup: usize,
    samples: usize,
    exchange_every: usize,
    seed: u64,
    outdir: &str,
    measure_every: usize,
    algorithm: &str,
) {
    match algorithm {
        "metropolis" | "auto" => {
            if algorithm == "auto" {
                eprintln!(
                    "  auto algorithm: using metropolis (auto-selection not yet implemented)"
                );
            }
            run_ising_fss_metropolis(
                sizes,
                t_min,
                t_max,
                n_replicas,
                warmup,
                samples,
                exchange_every,
                seed,
                outdir,
                measure_every,
            );
        }
        "msc" => {
            run_ising_fss_msc(
                sizes,
                t_min,
                t_max,
                n_replicas,
                warmup,
                samples,
                exchange_every,
                seed,
                outdir,
                measure_every,
            );
        }
        "wolff" => {
            run_ising_fss_wolff(
                sizes,
                t_min,
                t_max,
                n_replicas,
                warmup,
                samples,
                exchange_every,
                seed,
                outdir,
                measure_every,
            );
        }
        _ => {
            eprintln!("Error: --algorithm must be metropolis, msc, wolff, or auto");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "cuda")]
fn run_ising_fss_metropolis(
    sizes: &[usize],
    t_min: f64,
    t_max: f64,
    n_replicas: usize,
    warmup: usize,
    samples: usize,
    exchange_every: usize,
    seed: u64,
    outdir: &str,
    measure_every: usize,
) {
    use ising::cuda::lattice_gpu::LatticeGpu;
    use ising::cuda::parallel_tempering::{linspace_temperatures, replica_exchange};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::io::{BufWriter, Write as IoWrite};

    let temperatures = linspace_temperatures(t_min, t_max, n_replicas);

    for &n in sizes {
        eprintln!("  N={n}: creating {n_replicas} metropolis replicas...");
        let mut replicas: Vec<LatticeGpu> = (0..n_replicas)
            .map(|r| {
                LatticeGpu::new(n, seed.wrapping_add(r as u64 * 1000 + n as u64))
                    .expect("failed to create GPU lattice")
            })
            .collect();

        let mut replica_to_temp: Vec<usize> = (0..n_replicas).collect();
        let mut temp_to_replica: Vec<usize> = (0..n_replicas).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(n as u64));

        // Warmup
        eprintln!("  N={n}: warming up {warmup} sweeps...");
        for w in 0..warmup {
            for (r, lat) in replicas.iter_mut().enumerate() {
                let t_idx = replica_to_temp[r];
                let beta = 1.0 / temperatures[t_idx];
                lat.step(beta as f32, 1.0, 0.0).expect("GPU step failed");
            }
            if (w + 1) % 1000 == 0 {
                eprintln!("    warmup {}/{warmup}", w + 1);
            }
        }

        // Open timeseries file for streaming writes
        let model_name = "ising";
        let ts_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_timeseries.csv"));
        let ts_file = fs::File::create(&ts_path).expect("create timeseries file failed");
        let mut ts_writer = BufWriter::new(ts_file);
        writeln!(ts_writer, "temp_idx,E,M").expect("write header failed");

        // Sampling with parallel tempering
        let mut sum_e = vec![0.0_f64; n_replicas];
        let mut sum_e2 = vec![0.0_f64; n_replicas];
        let mut sum_m = vec![0.0_f64; n_replicas];
        let mut sum_m2 = vec![0.0_f64; n_replicas];
        let mut sum_m4 = vec![0.0_f64; n_replicas];
        let mut count = vec![0usize; n_replicas];
        // Keep ts_data in memory for jackknife at the end
        let mut ts_data: Vec<Vec<(f64, f64)>> = (0..n_replicas)
            .map(|_| Vec::with_capacity(samples / measure_every + 1))
            .collect();
        let mut energies = vec![0.0_f64; n_replicas]; // reused every sweep

        let n3 = (n * n * n) as f64;

        eprintln!("  N={n}: sampling {samples} sweeps with PT exchange every {exchange_every}...");
        for sweep in 0..samples {
            // Sweep all replicas
            for (r, lat) in replicas.iter_mut().enumerate() {
                let t_idx = replica_to_temp[r];
                let beta = 1.0 / temperatures[t_idx];
                lat.step(beta as f32, 1.0, 0.0).expect("GPU step failed");
            }

            // Measure and accumulate (GPU-resident — no spin transfer)
            let do_measure = (sweep + 1) % measure_every == 0;
            let do_exchange = (sweep + 1) % exchange_every == 0;

            if do_measure || do_exchange {
                for (r, lat) in replicas.iter_mut().enumerate() {
                    let t_idx = replica_to_temp[r];
                    let (e_per, m_per) = lat.measure_gpu(1.0).expect("GPU measure failed");
                    energies[r] = e_per * n3;

                    if do_measure {
                        sum_e[t_idx] += e_per;
                        sum_e2[t_idx] += e_per * e_per;
                        sum_m[t_idx] += m_per;
                        sum_m2[t_idx] += m_per * m_per;
                        sum_m4[t_idx] += m_per.powi(4);
                        count[t_idx] += 1;

                        ts_data[t_idx].push((e_per, m_per));

                        // Stream to disk immediately
                        writeln!(ts_writer, "{t_idx},{e_per:.8},{m_per:.8}")
                            .expect("write timeseries row failed");
                    }
                }
            }

            // Replica exchange
            if do_exchange {
                replica_exchange(
                    &temperatures,
                    &energies,
                    &mut replica_to_temp,
                    &mut temp_to_replica,
                    &mut rng,
                    sweep / exchange_every,
                );
            }

            // Progress logging
            if (sweep + 1) % 10000 == 0 {
                eprintln!("    sweep {}/{samples}", sweep + 1);
            }
        }

        // Flush timeseries
        ts_writer.flush().expect("flush timeseries failed");
        drop(ts_writer);
        eprintln!("  Wrote {}", ts_path.display());

        // Write summary CSV with jackknife error bars
        let summary_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_summary.csv"));
        let mut csv = String::from("T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err\n");
        for t_idx in 0..n_replicas {
            if ts_data[t_idx].is_empty() {
                continue;
            }
            let t = temperatures[t_idx];
            let beta = 1.0 / t;
            let n_blocks = 20.min(ts_data[t_idx].len());
            let obs = jackknife_observables(&ts_data[t_idx], beta, n3, n_blocks);
            csv.push_str(&format!(
                "{t:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                obs[0],
                obs[1],
                obs[2],
                obs[3],
                obs[4],
                obs[5],
                obs[6],
                obs[7],
                obs[8],
                obs[9],
                obs[10],
                obs[11],
            ));
        }
        fs::write(&summary_path, &csv).expect("write summary failed");
        eprintln!("  Wrote {}", summary_path.display());
    }
}

#[cfg(feature = "cuda")]
fn run_ising_fss_msc(
    sizes: &[usize],
    t_min: f64,
    t_max: f64,
    n_replicas: usize,
    warmup: usize,
    samples: usize,
    exchange_every: usize,
    seed: u64,
    outdir: &str,
    measure_every: usize,
) {
    use cudarc::driver::CudaDevice;
    use ising::cuda::msc_lattice::BatchedMscLattice;
    use ising::cuda::parallel_tempering::{linspace_temperatures, replica_exchange};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::io::{BufWriter, Write as IoWrite};

    let temperatures = linspace_temperatures(t_min, t_max, n_replicas);
    let device = CudaDevice::new(0).expect("CUDA init failed");

    // Helper: build betas array from current replica-to-temp mapping
    let build_betas = |replica_to_temp: &[usize], temps: &[f64]| -> Vec<f32> {
        replica_to_temp
            .iter()
            .map(|&t_idx| (1.0 / temps[t_idx]) as f32)
            .collect()
    };

    for &n in sizes {
        if n % 32 != 0 {
            eprintln!("Error: --algorithm msc requires N to be a multiple of 32, got {n}");
            std::process::exit(1);
        }

        eprintln!("  N={n}: creating batched MSC lattice ({n_replicas} replicas, single kernel launch)...");
        let mut lattice = BatchedMscLattice::new(n, n_replicas, seed.wrapping_add(n as u64), device.clone())
            .expect("failed to create batched MSC lattice");

        let mut replica_to_temp: Vec<usize> = (0..n_replicas).collect();
        let mut temp_to_replica: Vec<usize> = (0..n_replicas).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(n as u64));

        // Set initial Boltzmann tables for all replicas in one upload
        let betas = build_betas(&replica_to_temp, &temperatures);
        lattice.set_temperatures(&betas, 1.0).expect("set_temperatures failed");

        // Warmup — single kernel launch per sweep for all replicas
        eprintln!("  N={n}: warming up {warmup} sweeps...");
        for w in 0..warmup {
            lattice.step_all().expect("batched MSC step failed");
            if (w + 1) % 1000 == 0 {
                eprintln!("    warmup {}/{warmup}", w + 1);
            }
        }

        // Open timeseries file for streaming writes
        let model_name = "ising";
        let ts_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_timeseries.csv"));
        let ts_file = fs::File::create(&ts_path).expect("create timeseries file failed");
        let mut ts_writer = BufWriter::new(ts_file);
        writeln!(ts_writer, "temp_idx,E,M").expect("write header failed");

        // Sampling with parallel tempering
        let mut ts_data: Vec<Vec<(f64, f64)>> = (0..n_replicas)
            .map(|_| Vec::with_capacity(samples / measure_every + 1))
            .collect();
        let mut energies = vec![0.0_f64; n_replicas];

        let n3 = (n * n * n) as f64;

        eprintln!("  N={n}: sampling {samples} sweeps with PT exchange every {exchange_every}...");
        for sweep in 0..samples {
            lattice.step_all().expect("batched MSC step failed");

            let do_measure = (sweep + 1) % measure_every == 0;
            let do_exchange = (sweep + 1) % exchange_every == 0;

            if do_measure || do_exchange {
                for r in 0..n_replicas {
                    let t_idx = replica_to_temp[r];
                    let (e_per, m_per) = lattice.measure_replica(r, 1.0).expect("MSC measure failed");
                    energies[r] = e_per * n3;

                    if do_measure {
                        ts_data[t_idx].push((e_per, m_per));
                        writeln!(ts_writer, "{t_idx},{e_per:.8},{m_per:.8}")
                            .expect("write timeseries row failed");
                    }
                }
            }

            if do_exchange {
                replica_exchange(
                    &temperatures,
                    &energies,
                    &mut replica_to_temp,
                    &mut temp_to_replica,
                    &mut rng,
                    sweep / exchange_every,
                );
                // Update all Boltzmann tables in one upload after PT exchange
                let betas = build_betas(&replica_to_temp, &temperatures);
                lattice.set_temperatures(&betas, 1.0).expect("set_temperatures failed");
            }

            if (sweep + 1) % 10000 == 0 {
                eprintln!("    sweep {}/{samples}", sweep + 1);
            }
        }

        ts_writer.flush().expect("flush timeseries failed");
        drop(ts_writer);
        eprintln!("  Wrote {}", ts_path.display());

        let summary_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_summary.csv"));
        let mut csv = String::from("T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err\n");
        for t_idx in 0..n_replicas {
            if ts_data[t_idx].is_empty() {
                continue;
            }
            let t = temperatures[t_idx];
            let beta = 1.0 / t;
            let n_blocks = 20.min(ts_data[t_idx].len());
            let obs = jackknife_observables(&ts_data[t_idx], beta, n3, n_blocks);
            csv.push_str(&format!(
                "{t:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                obs[0],
                obs[1],
                obs[2],
                obs[3],
                obs[4],
                obs[5],
                obs[6],
                obs[7],
                obs[8],
                obs[9],
                obs[10],
                obs[11],
            ));
        }
        fs::write(&summary_path, &csv).expect("write summary failed");
        eprintln!("  Wrote {}", summary_path.display());
    }
}

#[cfg(feature = "cuda")]
fn run_ising_fss_wolff(
    sizes: &[usize],
    t_min: f64,
    t_max: f64,
    n_replicas: usize,
    warmup: usize,
    samples: usize,
    exchange_every: usize,
    seed: u64,
    outdir: &str,
    measure_every: usize,
) {
    use cudarc::driver::CudaDevice;
    use ising::cuda::parallel_tempering::{linspace_temperatures, replica_exchange};
    use ising::cuda::wolff_gpu::WolffGpuLattice;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::io::{BufWriter, Write as IoWrite};

    let temperatures = linspace_temperatures(t_min, t_max, n_replicas);
    let device = CudaDevice::new(0).expect("CUDA init failed");

    for &n in sizes {
        eprintln!("  N={n}: creating {n_replicas} Wolff replicas...");
        let mut replicas: Vec<WolffGpuLattice> = (0..n_replicas)
            .map(|r| {
                WolffGpuLattice::new(
                    n,
                    seed.wrapping_add(r as u64 * 1000 + n as u64),
                    device.clone(),
                )
                .expect("failed to create Wolff GPU lattice")
            })
            .collect();

        let mut replica_to_temp: Vec<usize> = (0..n_replicas).collect();
        let mut temp_to_replica: Vec<usize> = (0..n_replicas).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(n as u64));

        // Warmup
        eprintln!("  N={n}: warming up {warmup} sweeps...");
        for w in 0..warmup {
            for (r, lat) in replicas.iter_mut().enumerate() {
                let t_idx = replica_to_temp[r];
                let beta = 1.0 / temperatures[t_idx];
                lat.step(beta as f32, 1.0, &mut rng)
                    .expect("Wolff step failed");
            }
            if (w + 1) % 1000 == 0 {
                eprintln!("    warmup {}/{warmup}", w + 1);
            }
        }

        // Open timeseries file for streaming writes
        let model_name = "ising";
        let ts_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_timeseries.csv"));
        let ts_file = fs::File::create(&ts_path).expect("create timeseries file failed");
        let mut ts_writer = BufWriter::new(ts_file);
        writeln!(ts_writer, "temp_idx,E,M").expect("write header failed");

        // Sampling with parallel tempering
        let mut ts_data: Vec<Vec<(f64, f64)>> = (0..n_replicas)
            .map(|_| Vec::with_capacity(samples / measure_every + 1))
            .collect();
        let mut energies = vec![0.0_f64; n_replicas];

        let n3 = (n * n * n) as f64;

        eprintln!("  N={n}: sampling {samples} sweeps with PT exchange every {exchange_every}...");
        for sweep in 0..samples {
            for (r, lat) in replicas.iter_mut().enumerate() {
                let t_idx = replica_to_temp[r];
                let beta = 1.0 / temperatures[t_idx];
                lat.step(beta as f32, 1.0, &mut rng)
                    .expect("Wolff step failed");
            }

            let do_measure = (sweep + 1) % measure_every == 0;
            let do_exchange = (sweep + 1) % exchange_every == 0;

            if do_measure || do_exchange {
                for (r, lat) in replicas.iter_mut().enumerate() {
                    let t_idx = replica_to_temp[r];
                    let (e_per, m_per) = lat.measure_gpu(1.0).expect("Wolff measure failed");
                    energies[r] = e_per * n3;

                    if do_measure {
                        ts_data[t_idx].push((e_per, m_per));
                        writeln!(ts_writer, "{t_idx},{e_per:.8},{m_per:.8}")
                            .expect("write timeseries row failed");
                    }
                }
            }

            if do_exchange {
                replica_exchange(
                    &temperatures,
                    &energies,
                    &mut replica_to_temp,
                    &mut temp_to_replica,
                    &mut rng,
                    sweep / exchange_every,
                );
            }

            if (sweep + 1) % 10000 == 0 {
                eprintln!("    sweep {}/{samples}", sweep + 1);
            }
        }

        ts_writer.flush().expect("flush timeseries failed");
        drop(ts_writer);
        eprintln!("  Wrote {}", ts_path.display());

        let summary_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_summary.csv"));
        let mut csv = String::from("T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err\n");
        for t_idx in 0..n_replicas {
            if ts_data[t_idx].is_empty() {
                continue;
            }
            let t = temperatures[t_idx];
            let beta = 1.0 / t;
            let n_blocks = 20.min(ts_data[t_idx].len());
            let obs = jackknife_observables(&ts_data[t_idx], beta, n3, n_blocks);
            csv.push_str(&format!(
                "{t:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                obs[0],
                obs[1],
                obs[2],
                obs[3],
                obs[4],
                obs[5],
                obs[6],
                obs[7],
                obs[8],
                obs[9],
                obs[10],
                obs[11],
            ));
        }
        fs::write(&summary_path, &csv).expect("write summary failed");
        eprintln!("  Wrote {}", summary_path.display());
    }
}

/// Jackknife analysis of time series data.
/// Returns: (E, E_err, M, M_err, M2, M2_err, M4, M4_err, Cv, Cv_err, chi, chi_err)
fn jackknife_observables(ts: &[(f64, f64)], beta: f64, n3: f64, n_blocks: usize) -> [f64; 12] {
    let n = ts.len();
    if n < n_blocks || n_blocks < 2 {
        return [0.0; 12];
    }
    let block_size = n / n_blocks;
    let n_used = block_size * n_blocks;

    // Full-sample averages
    let (sum_e, sum_e2, sum_m, sum_m2, sum_m4) = ts[..n_used].iter().fold(
        (0.0, 0.0, 0.0, 0.0, 0.0),
        |(se, se2, sm, sm2, sm4), &(e, m)| {
            (se + e, se2 + e * e, sm + m, sm2 + m * m, sm4 + m.powi(4))
        },
    );
    let s = n_used as f64;
    let avg_e = sum_e / s;
    let avg_e2 = sum_e2 / s;
    let avg_m = sum_m / s;
    let avg_m2 = sum_m2 / s;
    let avg_m4 = sum_m4 / s;
    let cv = beta * beta * (avg_e2 - avg_e * avg_e) * n3;
    let chi = beta * (avg_m2 - avg_m * avg_m) * n3;

    // Jackknife: leave-one-block-out
    let mut jk_e = Vec::with_capacity(n_blocks);
    let mut jk_m = Vec::with_capacity(n_blocks);
    let mut jk_m2 = Vec::with_capacity(n_blocks);
    let mut jk_m4 = Vec::with_capacity(n_blocks);
    let mut jk_cv = Vec::with_capacity(n_blocks);
    let mut jk_chi = Vec::with_capacity(n_blocks);

    for b in 0..n_blocks {
        let start = b * block_size;
        let end = start + block_size;
        let mut be = 0.0;
        let mut be2 = 0.0;
        let mut bm = 0.0;
        let mut bm2 = 0.0;
        let mut bm4 = 0.0;
        for i in 0..n_used {
            if i >= start && i < end {
                continue;
            }
            let (e, m) = ts[i];
            be += e;
            be2 += e * e;
            bm += m;
            bm2 += m * m;
            bm4 += m.powi(4);
        }
        let sj = (n_used - block_size) as f64;
        let je = be / sj;
        let je2 = be2 / sj;
        let jm = bm / sj;
        let jm2 = bm2 / sj;
        let jm4 = bm4 / sj;
        jk_e.push(je);
        jk_m.push(jm);
        jk_m2.push(jm2);
        jk_m4.push(jm4);
        jk_cv.push(beta * beta * (je2 - je * je) * n3);
        jk_chi.push(beta * (jm2 - jm * jm) * n3);
    }

    let jk_err = |full: f64, jk: &[f64]| -> f64 {
        let nb = jk.len() as f64;
        let var: f64 = jk.iter().map(|&x| (x - full).powi(2)).sum::<f64>() * (nb - 1.0) / nb;
        var.sqrt()
    };

    [
        avg_e,
        jk_err(avg_e, &jk_e),
        avg_m,
        jk_err(avg_m, &jk_m),
        avg_m2,
        jk_err(avg_m2, &jk_m2),
        avg_m4,
        jk_err(avg_m4, &jk_m4),
        cv,
        jk_err(cv, &jk_cv),
        chi,
        jk_err(chi, &jk_chi),
    ]
}

#[cfg(feature = "cuda")]
fn run_continuous_fss(
    sizes: &[usize],
    n_comp: usize,
    t_min: f64,
    t_max: f64,
    n_replicas: usize,
    warmup: usize,
    samples: usize,
    exchange_every: usize,
    seed: u64,
    outdir: &str,
    delta: f32,
    n_overrelax: usize,
    measure_every: usize,
    wolff_every: usize,
) {
    use cudarc::driver::CudaDevice;
    use ising::cuda::gpu_lattice_continuous::ContinuousGpuLattice;
    use ising::cuda::parallel_tempering::{linspace_temperatures, replica_exchange};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::io::{BufWriter, Write as IoWrite};

    let model_name = if n_comp == 2 { "xy" } else { "heisenberg" };
    let temperatures = linspace_temperatures(t_min, t_max, n_replicas);
    let device = CudaDevice::new(0).expect("failed to init CUDA device");

    for &n in sizes {
        eprintln!("  N={n}: creating {n_replicas} {model_name} replicas...");
        let mut replicas: Vec<ContinuousGpuLattice> = (0..n_replicas)
            .map(|r| {
                ContinuousGpuLattice::new(
                    n,
                    n_comp,
                    seed.wrapping_add(r as u64 * 1000 + n as u64),
                    device.clone(),
                )
                .expect("failed to create continuous GPU lattice")
            })
            .collect();

        let mut replica_to_temp: Vec<usize> = (0..n_replicas).collect();
        let mut temp_to_replica: Vec<usize> = (0..n_replicas).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(n as u64));

        let n3 = (n * n * n) as f64;

        // Warmup
        eprintln!("  N={n}: warming up {warmup} sweeps...");
        for w in 0..warmup {
            for (r, lat) in replicas.iter_mut().enumerate() {
                let t_idx = replica_to_temp[r];
                let beta = (1.0 / temperatures[t_idx]) as f32;
                lat.sweep(beta, 1.0, delta, n_overrelax)
                    .expect("sweep failed");
            }
            if (w + 1) % 1000 == 0 {
                eprintln!("    warmup {}/{warmup}", w + 1);
            }
        }

        // Open timeseries file for streaming writes
        let ts_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_timeseries.csv"));
        let ts_file = fs::File::create(&ts_path).expect("create timeseries file failed");
        let mut ts_writer = BufWriter::new(ts_file);
        writeln!(ts_writer, "temp_idx,E,M").expect("write header failed");

        // Accumulators
        let mut sum_e = vec![0.0_f64; n_replicas];
        let mut sum_e2 = vec![0.0_f64; n_replicas];
        let mut sum_m = vec![0.0_f64; n_replicas];
        let mut sum_m2 = vec![0.0_f64; n_replicas];
        let mut sum_m4 = vec![0.0_f64; n_replicas];
        let mut count = vec![0usize; n_replicas];
        let mut ts_data: Vec<Vec<(f64, f64)>> = (0..n_replicas)
            .map(|_| Vec::with_capacity(samples / measure_every + 1))
            .collect();
        let mut energies = vec![0.0_f64; n_replicas]; // reused every sweep

        if wolff_every > 0 {
            eprintln!("  N={n}: Wolff embedding every {wolff_every} sweeps");
        }
        eprintln!("  N={n}: sampling {samples} sweeps with PT (measure every {measure_every})...");
        for sweep in 0..samples {
            // Sweep all replicas (Metropolis + over-relaxation)
            for (r, lat) in replicas.iter_mut().enumerate() {
                let t_idx = replica_to_temp[r];
                let beta = (1.0 / temperatures[t_idx]) as f32;
                lat.sweep(beta, 1.0, delta, n_overrelax)
                    .expect("sweep failed");
            }

            // Wolff embedding cluster step for global decorrelation
            if wolff_every > 0 && (sweep + 1) % wolff_every == 0 {
                for (r, lat) in replicas.iter_mut().enumerate() {
                    let t_idx = replica_to_temp[r];
                    let beta = (1.0 / temperatures[t_idx]) as f32;
                    lat.wolff_step(beta, 1.0, &mut rng)
                        .expect("Wolff step failed");
                }
            }

            let do_measure = (sweep + 1) % measure_every == 0;
            let do_exchange = (sweep + 1) % exchange_every == 0;

            if do_measure || do_exchange {
                for (r, lat) in replicas.iter_mut().enumerate() {
                    let t_idx = replica_to_temp[r];
                    let (e, mx, my, mz) = lat.measure_gpu(1.0).expect("GPU measure failed");
                    let e_per = e / n3;
                    let m_abs = ((mx * mx + my * my + mz * mz).sqrt()) / n3;
                    energies[r] = e;

                    if do_measure {
                        sum_e[t_idx] += e_per;
                        sum_e2[t_idx] += e_per * e_per;
                        sum_m[t_idx] += m_abs;
                        sum_m2[t_idx] += m_abs * m_abs;
                        sum_m4[t_idx] += m_abs.powi(4);
                        count[t_idx] += 1;

                        ts_data[t_idx].push((e_per, m_abs));

                        // Stream to disk immediately
                        writeln!(ts_writer, "{t_idx},{e_per:.8},{m_abs:.8}")
                            .expect("write timeseries row failed");
                    }
                }
            }

            if do_exchange {
                replica_exchange(
                    &temperatures,
                    &energies,
                    &mut replica_to_temp,
                    &mut temp_to_replica,
                    &mut rng,
                    sweep / exchange_every,
                );
            }

            if (sweep + 1) % 10000 == 0 {
                eprintln!("    sweep {}/{samples}", sweep + 1);
            }
        }

        // Flush timeseries
        ts_writer.flush().expect("flush timeseries failed");
        drop(ts_writer);
        eprintln!("  Wrote {}", ts_path.display());

        // Write summary CSV with jackknife error bars
        let summary_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_summary.csv"));
        let mut csv = String::from("T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err\n");
        for t_idx in 0..n_replicas {
            if ts_data[t_idx].is_empty() {
                continue;
            }
            let t = temperatures[t_idx];
            let beta = 1.0 / t;
            let n_blocks = 20.min(ts_data[t_idx].len());
            let obs = jackknife_observables(&ts_data[t_idx], beta, n3, n_blocks);
            csv.push_str(&format!(
                "{t:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                obs[0],
                obs[1],
                obs[2],
                obs[3],
                obs[4],
                obs[5],
                obs[6],
                obs[7],
                obs[8],
                obs[9],
                obs[10],
                obs[11],
            ));
        }
        fs::write(&summary_path, &csv).expect("write summary failed");
        eprintln!("  Wrote {}", summary_path.display());
    }
}

#[cfg(not(feature = "cuda"))]
fn run_ising_fss(
    _: &[usize],
    _: f64,
    _: f64,
    _: usize,
    _: usize,
    _: usize,
    _: usize,
    _: u64,
    _: &str,
    _: usize,
    _: &str,
) {
    eprintln!("Error: gpu_fss requires --features cuda");
    std::process::exit(1);
}

#[cfg(not(feature = "cuda"))]
fn run_continuous_fss(
    _: &[usize],
    _: usize,
    _: f64,
    _: f64,
    _: usize,
    _: usize,
    _: usize,
    _: usize,
    _: u64,
    _: &str,
    _: f32,
    _: usize,
    _: usize,
    _: usize,
) {
    eprintln!("Error: gpu_fss requires --features cuda");
    std::process::exit(1);
}
