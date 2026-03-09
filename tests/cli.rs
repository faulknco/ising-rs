//! Integration tests for CLI binaries.
//!
//! Each test runs a binary with minimal parameters (small N, few samples)
//! and validates the output format and basic correctness.

use std::process::Command;

fn cargo_bin(name: &str) -> Command {
    // Use the debug binary built alongside the test suite to avoid
    // lock contention from a competing `cargo run --release`.
    let test_exe = std::env::current_exe().expect("cannot find test exe");
    let deps_dir = test_exe.parent().expect("no parent for test exe");
    let bin_path = deps_dir.parent().unwrap().join(format!("{name}.exe"));
    // Fallback for unix-style paths without .exe
    let bin_path = if bin_path.exists() {
        bin_path
    } else {
        deps_dir.parent().unwrap().join(name)
    };
    let mut cmd = Command::new(bin_path);
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
    cmd
}

// ---------------------------------------------------------------------------
// sweep binary
// ---------------------------------------------------------------------------

#[test]
fn sweep_outputs_csv_header_and_data() {
    let output = cargo_bin("sweep")
        .args([
            "--n",
            "4",
            "--geometry",
            "cubic",
            "--samples",
            "10",
            "--warmup",
            "10",
            "--tmin",
            "3.0",
            "--tmax",
            "5.0",
            "--steps",
            "3",
            "--seed",
            "1",
        ])
        .output()
        .expect("failed to run sweep");

    assert!(output.status.success(), "sweep exited with error");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.trim().lines().collect();

    // Header check
    assert_eq!(lines[0], "T,E,M,Cv,chi", "unexpected CSV header");

    // Should have exactly 3 data rows (--steps 3)
    assert_eq!(
        lines.len(),
        4,
        "expected header + 3 data rows, got {}",
        lines.len()
    );

    // Each data row should have 5 comma-separated fields that parse as f64
    for line in &lines[1..] {
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(
            fields.len(),
            5,
            "expected 5 columns, got {}: {line}",
            fields.len()
        );
        for (j, f) in fields.iter().enumerate() {
            f.parse::<f64>().unwrap_or_else(|e| {
                panic!("field {j} '{f}' is not a valid f64: {e}");
            });
        }
    }

    // Temperature should span from tmin to tmax
    let t_first: f64 = lines[1].split(',').next().unwrap().parse().unwrap();
    let t_last: f64 = lines[3].split(',').next().unwrap().parse().unwrap();
    assert!(
        (t_first - 3.0).abs() < 0.01,
        "first T should be ~3.0, got {t_first}"
    );
    assert!(
        (t_last - 5.0).abs() < 0.01,
        "last T should be ~5.0, got {t_last}"
    );
}

#[test]
fn sweep_wolff_flag_works() {
    let output = cargo_bin("sweep")
        .args([
            "--n",
            "4",
            "--geometry",
            "cubic",
            "--wolff",
            "--samples",
            "10",
            "--warmup",
            "10",
            "--tmin",
            "3.0",
            "--tmax",
            "5.0",
            "--steps",
            "2",
            "--seed",
            "1",
        ])
        .output()
        .expect("failed to run sweep --wolff");

    assert!(output.status.success(), "sweep --wolff exited with error");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("T,E,M,Cv,chi"), "missing CSV header");
}

// ---------------------------------------------------------------------------
// fss binary
// ---------------------------------------------------------------------------

#[test]
fn fss_writes_csv_files() {
    let tmp = std::env::temp_dir().join("ising_test_fss");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let output = cargo_bin("fss")
        .args([
            "--sizes",
            "4,6",
            "--geometry",
            "cubic",
            "--wolff",
            "--samples",
            "10",
            "--warmup",
            "10",
            "--tmin",
            "3.0",
            "--tmax",
            "5.0",
            "--steps",
            "3",
            "--seed",
            "1",
            "--outdir",
            tmp.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run fss");

    assert!(
        output.status.success(),
        "fss exited with error: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Should produce one CSV per size
    for n in [4, 6] {
        let path = tmp.join(format!("fss_N{n}.csv"));
        assert!(path.exists(), "missing {}", path.display());
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines[0], "T,E,M,M2,M4,Cv,chi", "bad header in fss_N{n}.csv");
        assert_eq!(lines.len(), 4, "expected header + 3 rows in fss_N{n}.csv");

        // 7 columns per row
        for line in &lines[1..] {
            let fields: Vec<&str> = line.split(',').collect();
            assert_eq!(fields.len(), 7, "expected 7 columns in fss_N{n}.csv");
        }
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn fss_raw_mode_writes_timeseries() {
    let tmp = std::env::temp_dir().join("ising_test_fss_raw");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let output = cargo_bin("fss")
        .args([
            "--sizes",
            "4",
            "--raw",
            "--wolff",
            "--samples",
            "5",
            "--warmup",
            "5",
            "--tmin",
            "4.0",
            "--tmax",
            "5.0",
            "--steps",
            "2",
            "--seed",
            "1",
            "--outdir",
            tmp.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run fss --raw");

    assert!(
        output.status.success(),
        "fss --raw exited with error: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let path = tmp.join("fss_raw_N4.csv");
    assert!(path.exists(), "missing {}", path.display());
    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.trim().lines().collect();
    assert_eq!(
        lines[0], "T,sample,e,m_abs,m_signed",
        "bad header in raw CSV"
    );
    // 2 temperatures * 5 samples = 10 data rows
    assert_eq!(
        lines.len(),
        11,
        "expected header + 10 rows, got {}",
        lines.len()
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// mesh_sweep binary
// ---------------------------------------------------------------------------

#[test]
fn mesh_sweep_with_json_graph() {
    let tmp = std::env::temp_dir().join("ising_test_mesh");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // Create a minimal 4-node ring graph
    let graph_path = tmp.join("ring4.json");
    std::fs::write(
        &graph_path,
        r#"{
        "n_nodes": 4,
        "edges": [[0,1],[1,2],[2,3],[3,0]]
    }"#,
    )
    .unwrap();

    let output = cargo_bin("mesh_sweep")
        .args([
            "--graph",
            graph_path.to_str().unwrap(),
            "--j",
            "1.0",
            "--tmin",
            "1.0",
            "--tmax",
            "3.0",
            "--steps",
            "3",
            "--warmup",
            "10",
            "--samples",
            "10",
            "--seed",
            "1",
            "--outdir",
            tmp.to_str().unwrap(),
            "--prefix",
            "ring4",
        ])
        .output()
        .expect("failed to run mesh_sweep");

    assert!(
        output.status.success(),
        "mesh_sweep exited with error: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let csv_path = tmp.join("ring4_sweep.csv");
    assert!(csv_path.exists(), "missing {}", csv_path.display());
    let content = std::fs::read_to_string(&csv_path).unwrap();
    let lines: Vec<&str> = content.trim().lines().collect();
    assert_eq!(lines[0], "T,E,M,M2,M4,Cv,chi");
    assert_eq!(lines.len(), 4, "expected header + 3 rows");

    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// kz binary
// ---------------------------------------------------------------------------

#[test]
fn kz_writes_csv() {
    let tmp = std::env::temp_dir().join("ising_test_kz");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let output = cargo_bin("kz")
        .args([
            "--n",
            "4",
            "--geometry",
            "cubic",
            "--tau-min",
            "10",
            "--tau-max",
            "100",
            "--tau-steps",
            "3",
            "--trials",
            "2",
            "--seed",
            "1",
            "--outdir",
            tmp.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run kz");

    assert!(
        output.status.success(),
        "kz exited with error: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let path = tmp.join("kz_N4.csv");
    assert!(path.exists(), "missing {}", path.display());
    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.trim().lines().collect();
    assert_eq!(
        lines[0], "tau_q,rho,rho_err,n_trials",
        "bad header in kz CSV"
    );
    assert_eq!(lines.len(), 4, "expected header + 3 rows");

    // rho and rho_err should be between 0 and 1
    for line in &lines[1..] {
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(
            fields.len(),
            4,
            "expected 4 columns, got {}: {line}",
            fields.len()
        );
        let rho: f64 = fields[1].parse().unwrap();
        let rho_err: f64 = fields[2].parse().unwrap();
        let n_trials: usize = fields[3].parse().unwrap();
        assert!((0.0..=1.0).contains(&rho), "rho={rho} out of range [0,1]");
        assert!(
            (0.0..=1.0).contains(&rho_err),
            "rho_err={rho_err} out of range [0,1]"
        );
        assert_eq!(n_trials, 2, "unexpected trial count in KZ CSV");
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn kz_supports_single_tau_step() {
    let tmp = std::env::temp_dir().join("ising_test_kz_single_step");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let output = cargo_bin("kz")
        .args([
            "--n",
            "4",
            "--geometry",
            "cubic",
            "--tau-min",
            "25",
            "--tau-max",
            "25",
            "--tau-steps",
            "1",
            "--trials",
            "1",
            "--seed",
            "2",
            "--outdir",
            tmp.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run kz with one tau step");

    assert!(
        output.status.success(),
        "kz single-step exited with error: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let path = tmp.join("kz_N4.csv");
    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.trim().lines().collect();
    assert_eq!(lines.len(), 2, "expected header + 1 row");
    assert!(lines[1].starts_with("25,"), "expected tau_q=25 row");

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn kz_gpu_rejects_non_cubic_geometry() {
    let output = cargo_bin("kz")
        .args(["--geometry", "square", "--gpu"])
        .output()
        .expect("failed to run kz --gpu --geometry square");

    assert!(
        !output.status.success(),
        "expected kz to reject non-cubic GPU geometry"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("supports only --geometry cubic"),
        "unexpected stderr: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// coarsening binary
// ---------------------------------------------------------------------------

#[test]
fn coarsening_writes_csv() {
    let tmp = std::env::temp_dir().join("ising_test_coarsening");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let output = cargo_bin("coarsening")
        .args([
            "--n",
            "4",
            "--geometry",
            "cubic",
            "--t-quench",
            "3.0",
            "--steps",
            "50",
            "--sample-every",
            "25",
            "--warmup",
            "10",
            "--seed",
            "1",
            "--outdir",
            tmp.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run coarsening");

    assert!(
        output.status.success(),
        "coarsening exited with error: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let path = tmp.join("coarsening_N4_T3.00.csv");
    assert!(path.exists(), "missing {}", path.display());
    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.trim().lines().collect();
    assert_eq!(lines[0], "t,rho", "bad header in coarsening CSV");
    assert!(lines.len() >= 2, "expected at least header + 1 data row");

    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// heisenberg_fss binary
// ---------------------------------------------------------------------------

#[test]
fn heisenberg_fss_smoke() {
    let outdir = "/tmp/heis_test_fss";
    let status = cargo_bin("heisenberg_fss")
        .args([
            "--sizes",
            "4",
            "--tmin",
            "1.0",
            "--tmax",
            "2.0",
            "--steps",
            "3",
            "--warmup",
            "20",
            "--samples",
            "40",
            "--outdir",
            outdir,
        ])
        .status()
        .expect("failed to run heisenberg_fss");
    assert!(
        status.success(),
        "heisenberg_fss exited with non-zero status"
    );

    let csv = std::fs::read_to_string(format!("{outdir}/heisenberg_fss_N4.csv"))
        .expect("CSV not written");
    let rows: Vec<&str> = csv.lines().collect();
    assert_eq!(
        rows.len(),
        4,
        "expected header + 3 data rows, got {}",
        rows.len()
    );
    assert!(
        rows[0].contains("E_err"),
        "CSV header missing error columns"
    );
    assert!(!csv.contains("NaN"), "CSV contains NaN values");
}

// ---------------------------------------------------------------------------
// heisenberg_jfit binary
// ---------------------------------------------------------------------------

#[test]
fn heisenberg_jfit_smoke() {
    let outdir = "/tmp/heis_test_jfit";
    let status = cargo_bin("heisenberg_jfit")
        .args([
            "--graph",
            "analysis/graphs/bcc_N4.json",
            "--tmin",
            "5.0",
            "--tmax",
            "8.0",
            "--steps",
            "3",
            "--warmup",
            "20",
            "--samples",
            "40",
            "--outdir",
            outdir,
        ])
        .status()
        .expect("failed to run heisenberg_jfit");
    assert!(
        status.success(),
        "heisenberg_jfit exited with non-zero status"
    );

    let csv = std::fs::read_to_string(format!("{outdir}/heisenberg_jfit_bcc_N4.csv"))
        .expect("CSV not written");
    let rows: Vec<&str> = csv.lines().collect();
    assert_eq!(
        rows.len(),
        4,
        "expected header + 3 data rows, got {}",
        rows.len()
    );
    assert!(
        rows[0].starts_with("T,E,"),
        "unexpected CSV header: {}",
        rows[0]
    );
    assert!(!csv.contains("NaN"), "CSV contains NaN values");
}

// ---------------------------------------------------------------------------
// xy_fss binary
// ---------------------------------------------------------------------------

#[test]
fn xy_fss_smoke() {
    let outdir = "/tmp/xy_test_fss";
    let status = cargo_bin("xy_fss")
        .args([
            "--sizes",
            "4",
            "--tmin",
            "1.0",
            "--tmax",
            "2.0",
            "--steps",
            "5",
            "--warmup",
            "50",
            "--samples",
            "100",
            "--seed",
            "1",
            "--outdir",
            outdir,
        ])
        .status()
        .expect("failed to run xy_fss");
    assert!(status.success(), "xy_fss exited with non-zero status");

    let csv = std::fs::read_to_string(format!("{outdir}/xy_fss_N4.csv")).expect("CSV not written");
    let rows: Vec<&str> = csv.lines().collect();
    assert_eq!(
        rows.len(),
        6,
        "expected header + 5 data rows, got {}",
        rows.len()
    );
    assert_eq!(
        rows[0], "T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err",
        "unexpected CSV header: {}",
        rows[0]
    );
    assert!(!csv.contains("NaN"), "CSV contains NaN values");
}

// ---------------------------------------------------------------------------
// xy_jfit binary
// ---------------------------------------------------------------------------

#[test]
fn xy_jfit_smoke() {
    let outdir = "/tmp/xy_test_jfit";
    let status = cargo_bin("xy_jfit")
        .args([
            "--graph",
            "analysis/graphs/bcc_N4.json",
            "--tmin",
            "2.3",
            "--tmax",
            "3.5",
            "--steps",
            "5",
            "--warmup",
            "50",
            "--samples",
            "100",
            "--seed",
            "1",
            "--outdir",
            outdir,
        ])
        .status()
        .expect("failed to run xy_jfit");
    assert!(status.success(), "xy_jfit exited with non-zero status");

    let csv =
        std::fs::read_to_string(format!("{outdir}/xy_jfit_bcc_N4.csv")).expect("CSV not written");
    let rows: Vec<&str> = csv.lines().collect();
    assert_eq!(
        rows.len(),
        6,
        "expected header + 5 data rows, got {}",
        rows.len()
    );
    assert_eq!(
        rows[0], "T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err",
        "unexpected CSV header: {}",
        rows[0]
    );
    assert!(!csv.contains("NaN"), "CSV contains NaN values");
}

// ---------------------------------------------------------------------------
// gpu_fss binary (requires --features cuda and a GPU)
// ---------------------------------------------------------------------------

#[test]
#[ignore] // Requires --features cuda and a GPU
fn gpu_fss_ising_smoke() {
    let dir = std::env::temp_dir().join("ising_test_gpu_fss");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let status = std::process::Command::new("cargo")
        .args([
            "run",
            "--release",
            "--features",
            "cuda",
            "--bin",
            "gpu_fss",
            "--",
            "--model",
            "ising",
            "--sizes",
            "4",
            "--tmin",
            "4.4",
            "--tmax",
            "4.6",
            "--replicas",
            "4",
            "--warmup",
            "50",
            "--samples",
            "100",
            "--exchange-every",
            "10",
            "--outdir",
            dir.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run gpu_fss");
    assert!(status.success());

    let summary = dir.join("gpu_fss_ising_N4_summary.csv");
    assert!(summary.exists(), "summary CSV missing");

    let content = std::fs::read_to_string(&summary).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert!(lines.len() >= 2, "summary CSV should have header + data");
    assert!(lines[0].starts_with("T,E,"));

    let ts = dir.join("gpu_fss_ising_N4_timeseries.csv");
    assert!(ts.exists(), "timeseries CSV missing");

    let _ = std::fs::remove_dir_all(&dir);
}
