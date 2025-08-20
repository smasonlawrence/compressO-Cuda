// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{Manager, State};

// Simple application state for now
#[derive(Default)]
struct AppState {
    active_jobs: Arc<Mutex<HashMap<String, CompressionJobHandle>>>,
}

#[derive(Debug, Clone, Serialize)]
struct CompressionJobHandle {
    id: String,
    input_path: PathBuf,
    output_path: PathBuf,
    status: JobStatus,
    created_at: String, // Using String to avoid chrono complexity for now
    updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Serialize)]
struct SystemInfo {
    cuda_available: bool,
    ffmpeg_available: bool,
    temp_dir: String,
    version: String,
}

#[derive(Debug, Serialize)]
struct FileInfo {
    filename: String,
    size: u64,
    extension: String,
    modified: Option<String>,
}

// Basic Tauri commands
#[tauri::command]
async fn get_system_info() -> Result<SystemInfo, String> {
    println!("Getting system information");

    // Check FFmpeg availability
    let ffmpeg_available = std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    let temp_dir = std::env::temp_dir()
        .join("compresso_cuda")
        .to_string_lossy()
        .to_string();

    Ok(SystemInfo {
        cuda_available: false, // Disabled for now
        ffmpeg_available,
        temp_dir,
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[tauri::command]
async fn validate_file_path(path: String) -> Result<bool, String> {
    let path = PathBuf::from(path);
    Ok(path.exists() && path.is_file())
}

#[tauri::command]
async fn get_file_info(path: String) -> Result<FileInfo, String> {
    let path = PathBuf::from(path);
    
    if !path.exists() {
        return Err("File does not exist".to_string());
    }

    let metadata = std::fs::metadata(&path)
        .map_err(|e| format!("Failed to get file metadata: {}", e))?;

    let size = metadata.len();
    let modified = metadata.modified()
        .ok()
        .map(|_| "Unknown".to_string()); // Simplified for now

    let extension = path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_string();

    let filename = path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_string();

    Ok(FileInfo {
        filename,
        size,
        extension,
        modified,
    })
}

#[tauri::command]
async fn get_compression_jobs(
    state: State<'_, AppState>,
) -> Result<Vec<CompressionJobHandle>, String> {
    let jobs = state.active_jobs.lock().unwrap();
    let job_list: Vec<CompressionJobHandle> = jobs.values().cloned().collect();
    Ok(job_list)
}

#[tauri::command]
async fn open_file_location(path: String) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .args(["/select,", &path])
            .spawn()
            .map_err(|e| format!("Failed to open file location: {}", e))?;
    }
    
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .args(["-R", &path])
            .spawn()
            .map_err(|e| format!("Failed to open file location: {}", e))?;
    }
    
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(std::path::Path::new(&path).parent().unwrap_or(std::path::Path::new("/")))
            .spawn()
            .map_err(|e| format!("Failed to open file location: {}", e))?;
    }
    
    Ok(())
}

fn main() {
    println!("Starting CompressO-Cuda v{}", env!("CARGO_PKG_VERSION"));

    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            get_system_info,
            validate_file_path,
            get_file_info,
            get_compression_jobs,
            open_file_location,
        ])
        .setup(|_app| {
            println!("Application setup completed");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
