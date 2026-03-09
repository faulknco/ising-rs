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
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let kernel_files = [
        "src/cuda/kernels.cu",
        "src/cuda/continuous_spin_kernel.cu",
        "src/cuda/reduce_kernel.cu",
        "src/cuda/msc_kernel.cu",
    ];

    for src in &kernel_files {
        let src_path = PathBuf::from(src);
        let stem = src_path.file_stem().unwrap().to_str().unwrap();
        let ptx_out = out_dir.join(format!("{stem}.ptx"));

        let status = Command::new(&nvcc)
            .args([
                "-ptx",
                "-O3",
                "--allow-unsupported-compiler",
                "--generate-code",
                "arch=compute_75,code=sm_75",
                "-I",
                &format!("{}/include", cuda_path),
                src,
                "-o",
                ptx_out.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|_| panic!("nvcc not found — is CUDA_PATH set?"));

        assert!(status.success(), "nvcc compilation failed for {src}");
        println!("cargo:rerun-if-changed={src}");
    }
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
