import React, { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { open } from '@tauri-apps/api/dialog';
import { appWindow } from '@tauri-apps/api/window';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { 
  Upload, 
  Settings, 
  Activity, 
  Zap, 
  FileVideo, 
  Download,
  Monitor,
  Info,
  AlertTriangle,
  CheckCircle,
  Sparkles,
  Github,
  ExternalLink
} from 'lucide-react';

// Import our custom components
import { CudaSettings, type CudaSettings as CudaSettingsType } from '@/components/CudaSettings';
import { CompressionSettings, type CompressionSettings as CompressionSettingsType } from '@/components/CompressionSettings';
import { CompressionProgress } from '@/components/CompressionProgress';
import { JobQueueManager } from '@/components/JobQueueManager';

interface VideoFileInfo {
  filename: string;
  size: number;
  extension: string;
  modified?: string;
}

interface VideoInfo {
  width: number;
  height: number;
  fps: number;
  duration: number;
  codec: string;
}

interface SystemInfo {
  cuda: {
    available: boolean;
    devices: any[];
    best_device: number | null;
  };
  ffmpeg_available: boolean;
  temp_dir: string;
  version: string;
}

function App() {
  // Main application state
  const [activeTab, setActiveTab] = useState('compress');
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [outputPath, setOutputPath] = useState<string | null>(null);
  const [fileInfo, setFileInfo] = useState<VideoFileInfo | null>(null);
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [isCompressing, setIsCompressing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Settings state
  const [cudaSettings, setCudaSettings] = useState<CudaSettingsType>({
    use_cuda: false,
    cuda_device_id: null,
    preprocessing: {
      brightness: 0.0,
      contrast: 1.0,
      gamma: 1.0,
      denoise: false,
    },
  });

  const [compressionSettings, setCompressionSettings] = useState<CompressionSettingsType>({
    quality: 'Medium',
    mute_audio: false,
    audio_bitrate: 128,
    codec: 'H264',
    preset: 'Fast',
    profile: 'High',
    level: 'L4_1',
    use_cuda: false,
    preprocessing: {
      brightness: 0.0,
      contrast: 1.0,
      gamma: 1.0,
      denoise: false,
    },
    container_format: 'MP4',
    two_pass: false,
  });

  // Load system info on startup
  useEffect(() => {
    loadSystemInfo();
    initializeWindow();
  }, []);

  // Sync CUDA settings between components
  useEffect(() => {
    setCompressionSettings(prev => ({
      ...prev,
      use_cuda: cudaSettings.use_cuda,
      cuda_device_id: cudaSettings.cuda_device_id,
      preprocessing: cudaSettings.preprocessing,
    }));
  }, [cudaSettings]);

  const initializeWindow = async () => {
    try {
      await appWindow.setTitle('CompressO-Cuda - GPU Accelerated Video Compression');
      await appWindow.center();
    } catch (err) {
      console.warn('Failed to initialize window:', err);
    }
  };

  const loadSystemInfo = async () => {
    try {
      const info = await invoke<SystemInfo>('get_system_info');
      setSystemInfo(info);
      
      // Initialize compression engine if CUDA is available
      if (info.cuda.available) {
        await invoke('initialize_compression_engine', { 
          settings: {
            ...compressionSettings,
            use_cuda: true,
            cuda_device_id: info.cuda.best_device,
          }
        });
      }
    } catch (err) {
      setError(`Failed to load system info: ${err}`);
    }
  };

  const selectInputFile = async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [
          {
            name: 'Video Files',
            extensions: [
              'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 
              'm4v', '3gp', 'ts', 'mts', 'm2ts', 'vob', 'asf'
            ]
          }
        ]
      });

      if (selected && typeof selected === 'string') {
        setSelectedFile(selected);
        await loadFileInfo(selected);
        generateOutputPath(selected);
        setError(null);
      }
    } catch (err) {
      setError(`Failed to select file: ${err}`);
    }
  };

  const loadFileInfo = async (filePath: string) => {
    try {
      const info = await invoke<VideoFileInfo>('get_file_info', { path: filePath });
      setFileInfo(info);
      
      // You might want to add a function to get video-specific info
      // const videoInfo = await invoke<VideoInfo>('get_video_info', { path: filePath });
      // setVideoInfo(videoInfo);
    } catch (err) {
      console.warn('Failed to load file info:', err);
    }
  };

  const generateOutputPath = (inputPath: string) => {
    const pathParts = inputPath.split('.');
    const extension = compressionSettings.container_format.toLowerCase();
    const outputPath = `${pathParts.slice(0, -1).join('.')}_compressed.${extension}`;
    setOutputPath(outputPath);
  };

  const selectOutputPath = async () => {
    try {
      const selected = await open({
        multiple: false,
        directory: true,
      });

      if (selected && typeof selected === 'string') {
        const filename = fileInfo?.filename || 'output';
        const extension = compressionSettings.container_format.toLowerCase();
        setOutputPath(`${selected}/${filename}_compressed.${extension}`);
      }
    } catch (err) {
      setError(`Failed to select output path: ${err}`);
    }
  };

  const startCompression = async () => {
    if (!selectedFile || !outputPath) {
      setError('Please select input file and output path');
      return;
    }

    try {
      setIsCompressing(true);
      setError(null);

      // Initialize compression engine with current settings
      await invoke('initialize_compression_engine', { settings: compressionSettings });

      // Start compression
      const jobId = await invoke<string>('start_compression', {
        inputPath: selectedFile,
        outputPath: outputPath,
        settings: compressionSettings,
      });

      setCurrentJobId(jobId);
      setActiveTab('progress');
    } catch (err) {
      setError(`Failed to start compression: ${err}`);
      setIsCompressing(false);
    }
  };

  const handleJobComplete = useCallback((success: boolean, outputPath?: string) => {
    setIsCompressing(false);
    if (success) {
      setActiveTab('queue');
    }
  }, []);

  const handleJobCancel = useCallback(() => {
    setIsCompressing(false);
    setCurrentJobId(null);
  }, []);

  const formatFileSize = (bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
                <Zap className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  CompressO-Cuda
                </h1>
                <p className="text-muted-foreground">
                  GPU-Accelerated Video Compression {systemInfo?.version && `v${systemInfo.version}`}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {systemInfo?.cuda.available && (
                <Badge variant="default" className="bg-green-100 text-green-800">
                  <Zap className="h-3 w-3 mr-1" />
                  CUDA Ready
                </Badge>
              )}
              {systemInfo?.ffmpeg_available && (
                <Badge variant="outline">
                  <FileVideo className="h-3 w-3 mr-1" />
                  FFmpeg
                </Badge>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => window.open('https://github.com/smasonlawrence/compressO-Cuda')}
              >
                <Github className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* System Status */}
          {!systemInfo?.cuda.available && (
            <Alert className="mt-4">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                CUDA is not available. The application will use CPU encoding which may be slower.
                For GPU acceleration, ensure you have an NVIDIA GPU with updated drivers and CUDA toolkit installed.
              </AlertDescription>
            </Alert>
          )}

          {!systemInfo?.ffmpeg_available && (
            <Alert variant="destructive" className="mt-4">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                FFmpeg is not found. Please install FFmpeg to use video compression features.
              </AlertDescription>
            </Alert>
          )}
        </div>

        {/* Main Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="compress" className="flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Compress
            </TabsTrigger>
            <TabsTrigger value="settings" className="flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Settings
            </TabsTrigger>
            <TabsTrigger value="progress" className="flex items-center gap-2" disabled={!currentJobId}>
              <Activity className="h-4 w-4" />
              Progress
            </TabsTrigger>
            <TabsTrigger value="queue" className="flex items-center gap-2">
              <Monitor className="h-4 w-4" />
              Queue
            </TabsTrigger>
          </TabsList>

          {/* Compression Tab */}
          <TabsContent value="compress" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* File Selection */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Upload className="h-5 w-5" />
                    Input File
                  </CardTitle>
                  <CardDescription>
                    Select a video file to compress
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Button
                    onClick={selectInputFile}
                    className="w-full h-24 border-dashed border-2 bg-muted/50 hover:bg-muted"
                    variant="outline"
                  >
                    <div className="flex flex-col items-center gap-2">
                      <Upload className="h-8 w-8 text-muted-foreground" />
                      <span>Click to select video file</span>
                    </div>
                  </Button>

                  {selectedFile && fileInfo && (
                    <div className="p-4 bg-muted rounded-lg space-y-2">
                      <div className="flex items-center gap-2">
                        <FileVideo className="h-4 w-4" />
                        <span className="font-medium">{fileInfo.filename}</span>
                      </div>
                      <div className="text-sm text-muted-foreground">
                        Size: {formatFileSize(fileInfo.size)} • 
                        Format: {fileInfo.extension.toUpperCase()}
                        {fileInfo.modified && (
                          <> • Modified: {new Date(fileInfo.modified).toLocaleDateString()}</>
                        )}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Output Settings */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Download className="h-5 w-5" />
                    Output Settings
                  </CardTitle>
                  <CardDescription>
                    Configure output file and location
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Output Path</label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={outputPath || ''}
                        onChange={(e) => setOutputPath(e.target.value)}
                        placeholder="Select output location..."
                        className="flex-1 px-3 py-2 border rounded-md text-sm"
                      />
                      <Button variant="outline" onClick={selectOutputPath}>
                        Browse
                      </Button>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <Button
                      onClick={startCompression}
                      disabled={!selectedFile || !outputPath || isCompressing || !systemInfo?.ffmpeg_available}
                      className="flex-1"
                    >
                      {isCompressing ? (
                        <>
                          <Activity className="h-4 w-4 mr-2 animate-spin" />
                          Compressing...
                        </>
                      ) : (
                        <>
                          <Sparkles className="h-4 w-4 mr-2" />
                          Start Compression
                        </>
                      )}
                    </Button>
                  </div>

                  {systemInfo?.cuda.available && compressionSettings.use_cuda && (
                    <div className="p-3 bg-green-50 dark:bg-green-950/20 rounded-lg border border-green-200 dark:border-green-800">
                      <div className="flex items-center gap-2 text-green-700 dark:text-green-300">
                        <Zap className="h-4 w-4" />
                        <span className="text-sm font-medium">CUDA Acceleration Enabled</span>
                      </div>
                      <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                        GPU acceleration will significantly speed up compression
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Quick Compression Settings */}
            <CompressionSettings
              onSettingsChange={setCompressionSettings}
              initialSettings={compressionSettings}
              cudaAvailable={systemInfo?.cuda.available || false}
              videoInfo={videoInfo || undefined}
            />
          </TabsContent>

          {/* Settings Tab */}
          <TabsContent value="settings" className="space-y-6">
            <CudaSettings
              onSettingsChange={setCudaSettings}
              initialSettings={cudaSettings}
            />
          </TabsContent>

          {/* Progress Tab */}
          <TabsContent value="progress">
            {currentJobId ? (
              <CompressionProgress
                jobId={currentJobId}
                onJobComplete={handleJobComplete}
                onJobCancel={handleJobCancel}
                showPerformanceMetrics={true}
              />
            ) : (
              <Card>
                <CardContent className="text-center py-12">
                  <Activity className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-medium mb-2">No Active Compression</h3>
                  <p className="text-muted-foreground mb-4">
                    Start a compression job to monitor progress here
                  </p>
                  <Button onClick={() => setActiveTab('compress')}>
                    Start Compression
                  </Button>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Queue Tab */}
          <TabsContent value="queue">
            <JobQueueManager
              onJobSelect={(jobId) => {
                setCurrentJobId(jobId);
                setActiveTab('progress');
              }}
              onNewJob={() => setActiveTab('compress')}
              showCudaMetrics={systemInfo?.cuda.available || false}
            />
          </TabsContent>
        </Tabs>

        {/* Global Error Display */}
        {error && (
          <Alert variant="destructive" className="mt-6">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Footer */}
        <footer className="mt-12 pt-6 border-t border-border/40">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <div className="flex items-center gap-4">
              <span>© 2024 CompressO-Cuda</span>
              <Separator orientation="vertical" className="h-4" />
              <a 
                href="https://github.com/smasonlawrence/compressO-Cuda" 
                className="flex items-center gap-1 hover:text-foreground transition-colors"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Github className="h-3 w-3" />
                Open Source
              </a>
            </div>
            <div className="flex items-center gap-2">
              <span>Powered by</span>
              <Badge variant="outline" className="text-xs">
                Rust + CUDA + FFmpeg
              </Badge>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
