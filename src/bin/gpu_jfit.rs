/// CPU parallel tempering for crystal graph J-fitting (Ising, XY, Heisenberg).
///
/// Uses the existing CPU lattice types with replica exchange across temperatures.
/// GPU kernels only support cubic lattices; crystal graphs (BCC/FCC) are small (N=4..12)
/// so CPU is fast enough — the benefit comes from parallel tempering statistics.
///
/// Usage:
///   cargo run --release --bin gpu_jfit -- \
///     --model ising --graph analysis/graphs/bcc_N8.json \
///     --tmin 6.0 --tmax 6.7 --replicas 20 \
///     --warmup 5000 --samples 50000 \
///     --exchange-every 10 --seed 42 \
///     --outdir analysis/data
use std::env;
use std::fs;
use std::path::Path;

use ising::graph::GraphDef;
use ising::parallel_tempering::{linspace_temperatures, replica_exchange};

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
    let mut graph_path = String::new();
    let mut outdir = String::from("analysis/data");
    let mut t_min = 6.0_f64;
    let mut t_max = 6.7_f64;
    let mut n_replicas = 20usize;
    let mut warmup = 5000usize;
    let mut samples = 50000usize;
    let mut exchange_every = 10usize;
    let mut seed = 42u64;
    let mut j = 1.0_f64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"          => { model        = get_arg(&args, i, "--model"); i += 2; }
            "--graph"          => { graph_path   = get_arg(&args, i, "--graph"); i += 2; }
            "--outdir"         => { outdir       = get_arg(&args, i, "--outdir"); i += 2; }
            "--tmin"           => { t_min        = parse_flag(&args, i, "--tmin"); i += 2; }
            "--tmax"           => { t_max        = parse_flag(&args, i, "--tmax"); i += 2; }
            "--replicas"       => { n_replicas   = parse_flag(&args, i, "--replicas"); i += 2; }
            "--warmup"         => { warmup       = parse_flag(&args, i, "--warmup"); i += 2; }
            "--samples"        => { samples      = parse_flag(&args, i, "--samples"); i += 2; }
            "--exchange-every" => { exchange_every = parse_flag(&args, i, "--exchange-every"); i += 2; }
            "--seed"           => { seed         = parse_flag(&args, i, "--seed"); i += 2; }
            "--j"              => { j            = parse_flag(&args, i, "--j"); i += 2; }
            _ => { i += 1; }
        }
    }

    if graph_path.is_empty() {
        eprintln!("Error: --graph <path.json> is required");
        std::process::exit(1);
    }

    let content = fs::read_to_string(&graph_path).unwrap_or_else(|e| {
        eprintln!("Error: failed to read graph file {graph_path}: {e}");
        std::process::exit(1);
    });
    let graph = GraphDef::from_json(&content).expect("failed to parse graph JSON");

    let graph_name = Path::new(&graph_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("graph")
        .to_string();

    let mut neighbours: Vec<Vec<usize>> = vec![vec![]; graph.n_nodes];
    for (a, b) in &graph.edges {
        neighbours[*a].push(*b);
        neighbours[*b].push(*a);
    }

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    eprintln!(
        "jfit: model={model}, graph={graph_name}, N={}, T={t_min}..{t_max}, replicas={n_replicas}, exchange_every={exchange_every}",
        graph.n_nodes
    );

    match model.as_str() {
        "ising"      => run_ising_jfit(&neighbours, &graph_name, t_min, t_max, n_replicas, warmup, samples, exchange_every, seed, j, &outdir),
        "xy"         => run_xy_jfit(&neighbours, &graph_name, t_min, t_max, n_replicas, warmup, samples, exchange_every, seed, j, &outdir),
        "heisenberg" => run_heisenberg_jfit(&neighbours, &graph_name, t_min, t_max, n_replicas, warmup, samples, exchange_every, seed, j, &outdir),
        _ => {
            eprintln!("Error: --model must be ising, xy, or heisenberg");
            std::process::exit(1);
        }
    }
}

fn run_ising_jfit(
    neighbours: &[Vec<usize>], graph_name: &str,
    t_min: f64, t_max: f64, n_replicas: usize,
    warmup: usize, samples: usize, exchange_every: usize,
    seed: u64, j: f64, outdir: &str,
) {
    use ising::lattice::{Lattice, Geometry};
    use ising::metropolis::sweep;
    use ising::observables::energy_magnetisation;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let temperatures = linspace_temperatures(t_min, t_max, n_replicas);
    let n_nodes = neighbours.len();
    let n_f64 = n_nodes as f64;

    // Create R replicas
    let mut replicas: Vec<Lattice> = (0..n_replicas)
        .map(|r| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(r as u64 * 1000));
            let mut lat = Lattice {
                n: n_nodes,
                spins: vec![1i8; n_nodes],
                neighbours: neighbours.to_vec(),
                geometry: Geometry::Cubic3D,
            };
            lat.randomise(&mut rng);
            lat
        })
        .collect();

    let mut replica_to_temp: Vec<usize> = (0..n_replicas).collect();
    let mut temp_to_replica: Vec<usize> = (0..n_replicas).collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    // Warmup
    for _ in 0..warmup {
        for r in 0..n_replicas {
            let t_idx = replica_to_temp[r];
            let beta = 1.0 / temperatures[t_idx];
            sweep(&mut replicas[r], beta, j, 0.0, &mut rng);
        }
    }

    // Accumulators per temperature index
    let mut sum_e  = vec![0.0_f64; n_replicas];
    let mut sum_e2 = vec![0.0_f64; n_replicas];
    let mut sum_m  = vec![0.0_f64; n_replicas];
    let mut sum_m2 = vec![0.0_f64; n_replicas];
    let mut sum_m4 = vec![0.0_f64; n_replicas];
    let mut count  = vec![0usize; n_replicas];
    let mut n_exchanges = 0usize;

    for s in 0..samples {
        let mut energies = vec![0.0_f64; n_replicas];

        for r in 0..n_replicas {
            let t_idx = replica_to_temp[r];
            let beta = 1.0 / temperatures[t_idx];
            sweep(&mut replicas[r], beta, j, 0.0, &mut rng);

            let (e, m) = energy_magnetisation(&replicas[r], j, 0.0);
            let e_per = e / n_f64;
            let m_per = (m / n_f64).abs();
            energies[r] = e;

            sum_e[t_idx]  += e_per;
            sum_e2[t_idx] += e_per * e_per;
            sum_m[t_idx]  += m_per;
            sum_m2[t_idx] += m_per * m_per;
            sum_m4[t_idx] += m_per.powi(4);
            count[t_idx]  += 1;
        }

        if (s + 1) % exchange_every == 0 {
            n_exchanges += replica_exchange(
                &temperatures, &energies,
                &mut replica_to_temp, &mut temp_to_replica,
                &mut rng, s / exchange_every,
            );
        }
    }

    let path = Path::new(outdir).join(format!("gpu_jfit_ising_{graph_name}.csv"));
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
        let cv = beta * beta * (avg_e2 - avg_e * avg_e) * n_f64;
        let chi = beta * (avg_m2 - avg_m * avg_m) * n_f64;
        csv.push_str(&format!(
            "{:.4},{:.6},0.0,{:.6},0.0,{:.6},0.0,{:.6},0.0,{:.6},0.0,{:.6},0.0\n",
            t, avg_e, avg_m, avg_m2, avg_m4, cv, chi,
        ));
        eprintln!("  T={t:.3} M={avg_m:.4}");
    }

    eprintln!("  PT exchanges accepted: {n_exchanges}");
    fs::write(&path, &csv).expect("failed to write CSV");
    eprintln!("Wrote {}", path.display());
}

fn run_xy_jfit(
    neighbours: &[Vec<usize>], graph_name: &str,
    t_min: f64, t_max: f64, n_replicas: usize,
    warmup: usize, samples: usize, exchange_every: usize,
    seed: u64, j: f64, outdir: &str,
) {
    use ising::xy::{XyLattice, energy_magnetisation};
    use ising::xy::wolff::sweep;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let temperatures = linspace_temperatures(t_min, t_max, n_replicas);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let n_nodes = neighbours.len();
    let n_f64 = n_nodes as f64;

    let mut replicas: Vec<XyLattice> = (0..n_replicas)
        .map(|_| {
            let mut lat = XyLattice::new(neighbours.to_vec());
            lat.randomise(&mut rng);
            lat
        })
        .collect();

    let mut replica_to_temp: Vec<usize> = (0..n_replicas).collect();
    let mut temp_to_replica: Vec<usize> = (0..n_replicas).collect();

    // Warmup
    for _ in 0..warmup {
        for r in 0..n_replicas {
            let t_idx = replica_to_temp[r];
            let beta = 1.0 / temperatures[t_idx];
            sweep(&mut replicas[r], beta, j, &mut rng);
        }
    }

    let mut sum_e  = vec![0.0_f64; n_replicas];
    let mut sum_e2 = vec![0.0_f64; n_replicas];
    let mut sum_m  = vec![0.0_f64; n_replicas];
    let mut sum_m2 = vec![0.0_f64; n_replicas];
    let mut sum_m4 = vec![0.0_f64; n_replicas];
    let mut count  = vec![0usize; n_replicas];
    let mut n_exchanges = 0usize;

    for s in 0..samples {
        let mut energies = vec![0.0_f64; n_replicas];

        for r in 0..n_replicas {
            let t_idx = replica_to_temp[r];
            let beta = 1.0 / temperatures[t_idx];
            sweep(&mut replicas[r], beta, j, &mut rng);

            let (e, mag) = energy_magnetisation(&replicas[r], j);
            let e_per = e / n_f64;
            let m_abs = (mag[0] * mag[0] + mag[1] * mag[1]).sqrt() / n_f64;
            energies[r] = e;

            sum_e[t_idx]  += e_per;
            sum_e2[t_idx] += e_per * e_per;
            sum_m[t_idx]  += m_abs;
            sum_m2[t_idx] += m_abs * m_abs;
            sum_m4[t_idx] += m_abs.powi(4);
            count[t_idx]  += 1;
        }

        if (s + 1) % exchange_every == 0 {
            n_exchanges += replica_exchange(
                &temperatures, &energies,
                &mut replica_to_temp, &mut temp_to_replica,
                &mut rng, s / exchange_every,
            );
        }
    }

    let path = Path::new(outdir).join(format!("gpu_jfit_xy_{graph_name}.csv"));
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
        let cv = beta * beta * (avg_e2 - avg_e * avg_e) * n_f64;
        let chi = beta * (avg_m2 - avg_m * avg_m) * n_f64;
        csv.push_str(&format!(
            "{:.4},{:.6},0.0,{:.6},0.0,{:.6},0.0,{:.6},0.0,{:.6},0.0,{:.6},0.0\n",
            t, avg_e, avg_m, avg_m2, avg_m4, cv, chi,
        ));
        eprintln!("  T={t:.3} M={avg_m:.4}");
    }

    eprintln!("  PT exchanges accepted: {n_exchanges}");
    fs::write(&path, &csv).expect("failed to write CSV");
    eprintln!("Wrote {}", path.display());
}

fn run_heisenberg_jfit(
    neighbours: &[Vec<usize>], graph_name: &str,
    t_min: f64, t_max: f64, n_replicas: usize,
    warmup: usize, samples: usize, exchange_every: usize,
    seed: u64, j: f64, outdir: &str,
) {
    use ising::heisenberg::{HeisenbergLattice, energy_magnetisation};
    use ising::heisenberg::metropolis::sweep as heisenberg_sweep;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let temperatures = linspace_temperatures(t_min, t_max, n_replicas);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let n_nodes = neighbours.len();
    let n_f64 = n_nodes as f64;

    let mut replicas: Vec<HeisenbergLattice> = (0..n_replicas)
        .map(|_| {
            let mut lat = HeisenbergLattice::new(neighbours.to_vec());
            lat.randomise(&mut rng);
            lat
        })
        .collect();

    let mut replica_to_temp: Vec<usize> = (0..n_replicas).collect();
    let mut temp_to_replica: Vec<usize> = (0..n_replicas).collect();

    // Warmup
    for _ in 0..warmup {
        for r in 0..n_replicas {
            let t_idx = replica_to_temp[r];
            let beta = 1.0 / temperatures[t_idx];
            heisenberg_sweep(&mut replicas[r], beta, j, 0.5, &mut rng);
        }
    }

    let mut sum_e  = vec![0.0_f64; n_replicas];
    let mut sum_e2 = vec![0.0_f64; n_replicas];
    let mut sum_m  = vec![0.0_f64; n_replicas];
    let mut sum_m2 = vec![0.0_f64; n_replicas];
    let mut sum_m4 = vec![0.0_f64; n_replicas];
    let mut count  = vec![0usize; n_replicas];
    let mut n_exchanges = 0usize;

    for s in 0..samples {
        let mut energies = vec![0.0_f64; n_replicas];

        for r in 0..n_replicas {
            let t_idx = replica_to_temp[r];
            let beta = 1.0 / temperatures[t_idx];
            heisenberg_sweep(&mut replicas[r], beta, j, 0.5, &mut rng);

            let (e, mag) = energy_magnetisation(&replicas[r], j);
            let e_per = e / n_f64;
            let m_abs = (mag[0] * mag[0] + mag[1] * mag[1] + mag[2] * mag[2]).sqrt() / n_f64;
            energies[r] = e;

            sum_e[t_idx]  += e_per;
            sum_e2[t_idx] += e_per * e_per;
            sum_m[t_idx]  += m_abs;
            sum_m2[t_idx] += m_abs * m_abs;
            sum_m4[t_idx] += m_abs.powi(4);
            count[t_idx]  += 1;
        }

        if (s + 1) % exchange_every == 0 {
            n_exchanges += replica_exchange(
                &temperatures, &energies,
                &mut replica_to_temp, &mut temp_to_replica,
                &mut rng, s / exchange_every,
            );
        }
    }

    let path = Path::new(outdir).join(format!("gpu_jfit_heisenberg_{graph_name}.csv"));
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
        let cv = beta * beta * (avg_e2 - avg_e * avg_e) * n_f64;
        let chi = beta * (avg_m2 - avg_m * avg_m) * n_f64;
        csv.push_str(&format!(
            "{:.4},{:.6},0.0,{:.6},0.0,{:.6},0.0,{:.6},0.0,{:.6},0.0,{:.6},0.0\n",
            t, avg_e, avg_m, avg_m2, avg_m4, cv, chi,
        ));
        eprintln!("  T={t:.3} M={avg_m:.4}");
    }

    eprintln!("  PT exchanges accepted: {n_exchanges}");
    fs::write(&path, &csv).expect("failed to write CSV");
    eprintln!("Wrote {}", path.display());
}
