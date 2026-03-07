use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only compile CUDA kernels if the cuda feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc");
    let kernel_src = PathBuf::from("src/cuda/kernels.cu");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_out = out_dir.join("kernels.ptx");

    let status = Command::new(&nvcc)
        .args([
            "-ptx",
            "-O3",
            "--allow-unsupported-compiler",
            "--generate-code",
            "arch=compute_75,code=sm_75",
            "-I",
            &format!("{}/include", cuda_path),
            kernel_src.to_str().unwrap(),
            "-o",
            ptx_out.to_str().unwrap(),
        ])
        .status()
        .expect("nvcc not found — is CUDA_PATH set and CUDA Toolkit installed?");

    assert!(status.success(), "nvcc compilation failed");

    println!("cargo:rerun-if-changed=src/cuda/kernels.cu");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
