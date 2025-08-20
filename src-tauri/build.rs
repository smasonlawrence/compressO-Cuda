use std::env;
use std::path::PathBuf;

fn main() {
    // Configure Tauri build first
    tauri_build::build();

    // Only build CUDA components if the feature is enabled
    if cfg!(feature = "cuda") {
        println!("cargo:rerun-if-changed=cuda/");
        
        match build_cuda_components() {
            Ok(_) => println!("CUDA components built successfully"),
            Err(e) => {
                eprintln!("Warning: Failed to build CUDA components: {}", e);
                eprintln!("Continuing with CPU-only build...");
                // Don't fail the build, just continue without CUDA
            }
        }
    } else {
        println!("CUDA feature disabled, building CPU-only version");
    }
}

fn build_cuda_components() -> Result<(), Box<dyn std::error::Error>> {
    let cuda_path = find_cuda_path()?;
    println!("Found CUDA at: {}", cuda_path);
    
    // Set up CUDA library search paths
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    println!("cargo:rustc-link-search=native={}/lib", cuda_path);

    // Link CUDA libraries
    link_cuda_libraries();

    // Compile CUDA kernels
    compile_cuda_kernels(&cuda_path)?;

    // Generate bindings
    generate_cuda_bindings(&cuda_path)?;

    Ok(())
}

fn compile_cuda_kernels(cuda_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Check if nvcc is available
    if std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .is_err()
    {
        return Err("nvcc (CUDA compiler) not found in PATH".into());
    }

    let mut cuda_build = cc::Build::new();
    
    cuda_build
        .cuda(true)
        .flag("-std=c++14")  // Use C++14 for broader compatibility
        .flag("-O3")
        .flag("--expt-relaxed-constexpr")
        .flag("--use_fast_math")
        .flag("--ptxas-options=-v")
        // GPU architectures - covering most common cards
        .flag("-gencode=arch=compute_60,code=sm_60")  // Pascal (GTX 1060, 1070, 1080)
        .flag("-gencode=arch=compute_61,code=sm_61")  // Pascal (GTX 1050, 1080 Ti)
        .flag("-gencode=arch=compute_70,code=sm_70")  // Volta (Tesla V100)
        .flag("-gencode=arch=compute_75,code=sm_75")  // Turing (RTX 2060, 2070, 2080)
        .flag("-gencode=arch=compute_80,code=sm_80")  // Ampere (RTX 3060, 3070, 3080)
        .flag("-gencode=arch=compute_86,code=sm_86")  // Ampere (RTX 3060 Ti, 3070 Ti, 3080 Ti, 3090)
        .flag("-gencode=arch=compute_89,code=sm_89")  // Ada Lovelace (RTX 4090)
        .include("cuda")
        .include(&format!("{}/include", cuda_path));

    // Add CUDA source files
    let cuda_files = [
        "cuda/kernels.cu",
        "cuda/preprocessing.cu", 
        "cuda/colorspace.cu",
        "cuda/scaling.cu",
    ];

    for file in &cuda_files {
        if std::path::Path::new(file).exists() {
            cuda_build.file(file);
        } else {
            println!("cargo:warning=CUDA file {} not found, skipping", file);
        }
    }

    cuda_build.compile("compresso_cuda_kernels");
    Ok(())
}

fn link_cuda_libraries() {
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    
    // Optional libraries - don't fail if not available
    let optional_libs = ["curand", "cublas", "cufft", "nppi", "nppig", "nppc"];
    for lib in &optional_libs {
        println!("cargo:rustc-link-lib=dylib={}", lib);
    }
}

fn generate_cuda_bindings(cuda_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let header_path = "cuda/kernels.h";
    
    if !std::path::Path::new(header_path).exists() {
        println!("cargo:warning=CUDA header {} not found, skipping bindings generation", header_path);
        return Ok(());
    }

    let bindings = bindgen::Builder::default()
        .header(header_path)
        .clang_arg(&format!("-I{}/include", cuda_path))
        .clang_arg("-I./cuda")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .allowlist_function("launch_.*")
        .allowlist_function("cuda_.*")
        .allowlist_function("get_cuda_.*")
        .allowlist_type("CudaDeviceInfo")
        .allowlist_var("cudaError_.*")
        .generate()?;

    let out_path = PathBuf::from(env::var("OUT_DIR")?);
    bindings.write_to_file(out_path.join("cuda_bindings.rs"))?;

    Ok(())
}

fn find_cuda_path() -> Result<String, Box<dyn std::error::Error>> {
    // Try environment variables first
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        if std::path::Path::new(&cuda_path).exists() {
            return Ok(cuda_path);
        }
    }

    if let Ok(cuda_root) = env::var("CUDA_ROOT") {
        if std::path::Path::new(&cuda_root).exists() {
            return Ok(cuda_root);
        }
    }

    // Try common installation paths
    let possible_paths = vec![
        "/usr/local/cuda",
        "/opt/cuda", 
        "/usr/cuda",
        // Windows paths
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
        // macOS paths (if using NVIDIA eGPU)
        "/usr/local/cuda-12.3",
        "/usr/local/cuda-12.2",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-12.0",
    ];

    for path in possible_paths {
        if std::path::Path::new(path).exists() {
            // Verify it's a valid CUDA installation by checking for nvcc
            let nvcc_path = if cfg!(windows) {
                format!("{}\\bin\\nvcc.exe", path)
            } else {
                format!("{}/bin/nvcc", path)
            };
            
            if std::path::Path::new(&nvcc_path).exists() {
                return Ok(path.to_string());
            }
        }
    }

    Err("CUDA installation not found. Please install CUDA toolkit and set CUDA_PATH environment variable.".into())
}
