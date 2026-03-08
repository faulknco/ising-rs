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

    let mut model = String::from("ising");
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

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"          => { model        = get_arg(&args, i, "--model"); i += 2; }
            "--sizes"          => { sizes_str    = get_arg(&args, i, "--sizes"); i += 2; }
            "--tmin"           => { t_min        = parse_flag(&args, i, "--tmin"); i += 2; }
            "--tmax"           => { t_max        = parse_flag(&args, i, "--tmax"); i += 2; }
            "--replicas"       => { n_replicas   = parse_flag(&args, i, "--replicas"); i += 2; }
            "--warmup"         => { warmup       = parse_flag(&args, i, "--warmup"); i += 2; }
            "--samples"        => { samples      = parse_flag(&args, i, "--samples"); i += 2; }
            "--exchange-every" => { exchange_every = parse_flag(&args, i, "--exchange-every"); i += 2; }
            "--seed"           => { seed         = parse_flag(&args, i, "--seed"); i += 2; }
            "--outdir"         => { outdir       = get_arg(&args, i, "--outdir"); i += 2; }
            "--delta"          => { delta        = parse_flag(&args, i, "--delta"); i += 2; }
            "--overrelax"      => { n_overrelax  = parse_flag(&args, i, "--overrelax"); i += 2; }
            _ => { i += 1; }
        }
    }

    let sizes: Vec<usize> = sizes_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    eprintln!("GPU FSS: model={model}, sizes={sizes:?}, T={t_min}..{t_max}, replicas={n_replicas}");

    match model.as_str() {
        "ising"      => run_ising_fss(&sizes, t_min, t_max, n_replicas, warmup, samples, exchange_every, seed, &outdir),
        "xy"         => run_continuous_fss(&sizes, 2, t_min, t_max, n_replicas, warmup, samples, exchange_every, seed, &outdir, delta as f32, n_overrelax),
        "heisenberg" => run_continuous_fss(&sizes, 3, t_min, t_max, n_replicas, warmup, samples, exchange_every, seed, &outdir, delta as f32, n_overrelax),
        _ => {
            eprintln!("Error: --model must be ising, xy, or heisenberg");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "cuda")]
fn run_ising_fss(
    sizes: &[usize], t_min: f64, t_max: f64, n_replicas: usize,
    warmup: usize, samples: usize, exchange_every: usize, seed: u64, outdir: &str,
) {
    use ising::cuda::lattice_gpu::LatticeGpu;
    use ising::cuda::parallel_tempering::{linspace_temperatures, replica_exchange};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let temperatures = linspace_temperatures(t_min, t_max, n_replicas);

    for &n in sizes {
        eprintln!("  N={n}: creating {n_replicas} replicas...");
        let mut replicas: Vec<LatticeGpu> = (0..n_replicas)
            .map(|r| LatticeGpu::new(n, seed.wrapping_add(r as u64 * 1000 + n as u64))
                .expect("failed to create GPU lattice"))
            .collect();

        let mut replica_to_temp: Vec<usize> = (0..n_replicas).collect();
        let mut temp_to_replica: Vec<usize> = (0..n_replicas).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(n as u64));

        // Warmup
        eprintln!("  N={n}: warming up {warmup} sweeps...");
        for _ in 0..warmup {
            for (r, lat) in replicas.iter_mut().enumerate() {
                let t_idx = replica_to_temp[r];
                let beta = 1.0 / temperatures[t_idx];
                lat.step(beta as f32, 1.0, 0.0).expect("GPU step failed");
            }
        }

        // Sampling with parallel tempering
        let mut sum_e    = vec![0.0_f64; n_replicas];
        let mut sum_e2   = vec![0.0_f64; n_replicas];
        let mut sum_m    = vec![0.0_f64; n_replicas];
        let mut sum_m2   = vec![0.0_f64; n_replicas];
        let mut sum_m4   = vec![0.0_f64; n_replicas];
        let mut count    = vec![0usize; n_replicas];
        let mut ts_data: Vec<Vec<(f64, f64)>> = vec![vec![]; n_replicas];

        let n3 = (n * n * n) as f64;

        eprintln!("  N={n}: sampling {samples} sweeps with PT exchange every {exchange_every}...");
        for sweep in 0..samples {
            // Sweep all replicas
            for (r, lat) in replicas.iter_mut().enumerate() {
                let t_idx = replica_to_temp[r];
                let beta = 1.0 / temperatures[t_idx];
                lat.step(beta as f32, 1.0, 0.0).expect("GPU step failed");
            }

            // Measure and accumulate
            let mut energies = vec![0.0_f64; n_replicas];
            for (r, lat) in replicas.iter().enumerate() {
                let t_idx = replica_to_temp[r];
                let spins = lat.get_spins().expect("get_spins failed");
                let (e, m) = ising_e_m_host(&spins, n);
                let e_per = e / n3;
                let m_per = (m / n3).abs();
                energies[r] = e;

                sum_e[t_idx]  += e_per;
                sum_e2[t_idx] += e_per * e_per;
                sum_m[t_idx]  += m_per;
                sum_m2[t_idx] += m_per * m_per;
                sum_m4[t_idx] += m_per.powi(4);
                count[t_idx]  += 1;

                ts_data[t_idx].push((e_per, m_per));
            }

            // Replica exchange
            if (sweep + 1) % exchange_every == 0 {
                replica_exchange(
                    &temperatures, &energies,
                    &mut replica_to_temp, &mut temp_to_replica,
                    &mut rng, sweep / exchange_every,
                );
            }
        }

        // Write summary CSV
        let model_name = "ising";
        let summary_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_summary.csv"));
        let mut csv = String::from("T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err\n");
        for t_idx in 0..n_replicas {
            let s = count[t_idx] as f64;
            if s == 0.0 { continue; }
            let t = temperatures[t_idx];
            let beta = 1.0 / t;
            let avg_e = sum_e[t_idx] / s;
            let avg_e2 = sum_e2[t_idx] / s;
            let avg_m = sum_m[t_idx] / s;
            let avg_m2 = sum_m2[t_idx] / s;
            let avg_m4 = sum_m4[t_idx] / s;
            let cv = beta * beta * (avg_e2 - avg_e * avg_e) * n3;
            csv.push_str(&format!(
                "{t:.6},{avg_e:.6},0.0,{avg_m:.6},0.0,{avg_m2:.6},0.0,{avg_m4:.6},0.0,{cv:.6},0.0,0.0,0.0\n"
            ));
        }
        fs::write(&summary_path, &csv).expect("write summary failed");
        eprintln!("  Wrote {}", summary_path.display());

        // Write time series CSV for reweighting
        let ts_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_timeseries.csv"));
        let mut ts_csv = String::from("temp_idx,E,M\n");
        for (t_idx, data) in ts_data.iter().enumerate() {
            for &(e, m) in data {
                ts_csv.push_str(&format!("{t_idx},{e:.8},{m:.8}\n"));
            }
        }
        fs::write(&ts_path, &ts_csv).expect("write timeseries failed");
        eprintln!("  Wrote {}", ts_path.display());
    }
}

fn ising_e_m_host(spins: &[i8], n: usize) -> (f64, f64) {
    let mut e = 0.0_f64;
    let mut m = 0.0_f64;
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let idx = iz * n * n + iy * n + ix;
                let s = spins[idx] as f64;
                let fwd = [
                    iz * n * n + iy * n + (ix + 1) % n,
                    iz * n * n + ((iy + 1) % n) * n + ix,
                    ((iz + 1) % n) * n * n + iy * n + ix,
                ];
                for &nb in &fwd {
                    e -= s * spins[nb] as f64;
                }
                m += s;
            }
        }
    }
    (e, m)
}

#[cfg(feature = "cuda")]
fn run_continuous_fss(
    sizes: &[usize], n_comp: usize, t_min: f64, t_max: f64,
    n_replicas: usize, warmup: usize, samples: usize,
    exchange_every: usize, seed: u64, outdir: &str,
    delta: f32, n_overrelax: usize,
) {
    use ising::cuda::gpu_lattice_continuous::ContinuousGpuLattice;
    use ising::cuda::parallel_tempering::{linspace_temperatures, replica_exchange};
    use cudarc::driver::CudaDevice;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let model_name = if n_comp == 2 { "xy" } else { "heisenberg" };
    let temperatures = linspace_temperatures(t_min, t_max, n_replicas);
    let device = CudaDevice::new(0).expect("failed to init CUDA device");

    for &n in sizes {
        eprintln!("  N={n}: creating {n_replicas} {model_name} replicas...");
        let mut replicas: Vec<ContinuousGpuLattice> = (0..n_replicas)
            .map(|r| {
                ContinuousGpuLattice::new(
                    n, n_comp,
                    seed.wrapping_add(r as u64 * 1000 + n as u64),
                    device.clone(),
                ).expect("failed to create continuous GPU lattice")
            })
            .collect();

        let mut replica_to_temp: Vec<usize> = (0..n_replicas).collect();
        let mut temp_to_replica: Vec<usize> = (0..n_replicas).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(n as u64));

        let n3 = (n * n * n) as f64;

        // Warmup
        eprintln!("  N={n}: warming up {warmup} sweeps...");
        for _ in 0..warmup {
            for (r, lat) in replicas.iter_mut().enumerate() {
                let t_idx = replica_to_temp[r];
                let beta = (1.0 / temperatures[t_idx]) as f32;
                lat.sweep(beta, 1.0, delta, n_overrelax).expect("sweep failed");
            }
        }

        // Accumulators
        let mut sum_e  = vec![0.0_f64; n_replicas];
        let mut sum_e2 = vec![0.0_f64; n_replicas];
        let mut sum_m  = vec![0.0_f64; n_replicas];
        let mut sum_m2 = vec![0.0_f64; n_replicas];
        let mut sum_m4 = vec![0.0_f64; n_replicas];
        let mut count  = vec![0usize; n_replicas];
        let mut ts_data: Vec<Vec<(f64, f64)>> = vec![vec![]; n_replicas];

        eprintln!("  N={n}: sampling {samples} sweeps with PT...");
        for sweep in 0..samples {
            let mut energies = vec![0.0_f64; n_replicas];

            for (r, lat) in replicas.iter_mut().enumerate() {
                let t_idx = replica_to_temp[r];
                let beta = (1.0 / temperatures[t_idx]) as f32;
                lat.sweep(beta, 1.0, delta, n_overrelax).expect("sweep failed");

                let (e, mx, my, mz) = lat.measure_raw().expect("measure failed");
                let e_per = e / n3;
                let m_abs = ((mx * mx + my * my + mz * mz).sqrt()) / n3;
                energies[r] = e;

                sum_e[t_idx]  += e_per;
                sum_e2[t_idx] += e_per * e_per;
                sum_m[t_idx]  += m_abs;
                sum_m2[t_idx] += m_abs * m_abs;
                sum_m4[t_idx] += m_abs.powi(4);
                count[t_idx]  += 1;

                ts_data[t_idx].push((e_per, m_abs));
            }

            if (sweep + 1) % exchange_every == 0 {
                replica_exchange(
                    &temperatures, &energies,
                    &mut replica_to_temp, &mut temp_to_replica,
                    &mut rng, sweep / exchange_every,
                );
            }

            if (sweep + 1) % 10000 == 0 {
                eprintln!("    sweep {}/{samples}", sweep + 1);
            }
        }

        // Write summary CSV
        let summary_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_summary.csv"));
        let mut csv = String::from("T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err\n");
        for t_idx in 0..n_replicas {
            let s = count[t_idx] as f64;
            if s == 0.0 { continue; }
            let t = temperatures[t_idx];
            let beta = 1.0 / t;
            let avg_e = sum_e[t_idx] / s;
            let avg_e2 = sum_e2[t_idx] / s;
            let avg_m = sum_m[t_idx] / s;
            let avg_m2 = sum_m2[t_idx] / s;
            let avg_m4 = sum_m4[t_idx] / s;
            let cv = beta * beta * (avg_e2 - avg_e * avg_e) * n3;
            csv.push_str(&format!(
                "{t:.6},{avg_e:.6},0.0,{avg_m:.6},0.0,{avg_m2:.6},0.0,{avg_m4:.6},0.0,{cv:.6},0.0,0.0,0.0\n"
            ));
        }
        fs::write(&summary_path, &csv).expect("write summary failed");
        eprintln!("  Wrote {}", summary_path.display());

        // Write time series
        let ts_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_timeseries.csv"));
        let mut ts_csv = String::from("temp_idx,E,M\n");
        for (t_idx, data) in ts_data.iter().enumerate() {
            for &(e, m) in data {
                ts_csv.push_str(&format!("{t_idx},{e:.8},{m:.8}\n"));
            }
        }
        fs::write(&ts_path, &ts_csv).expect("write timeseries failed");
        eprintln!("  Wrote {}", ts_path.display());
    }
}

#[cfg(not(feature = "cuda"))]
fn run_ising_fss(_: &[usize], _: f64, _: f64, _: usize, _: usize, _: usize, _: usize, _: u64, _: &str) {
    eprintln!("Error: gpu_fss requires --features cuda");
    std::process::exit(1);
}

#[cfg(not(feature = "cuda"))]
fn run_continuous_fss(_: &[usize], _: usize, _: f64, _: f64, _: usize, _: usize, _: usize, _: usize, _: u64, _: &str, _: f32, _: usize) {
    eprintln!("Error: gpu_fss requires --features cuda");
    std::process::exit(1);
}
