import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Zap, 
  Cpu, 
  HardDrive, 
  Monitor, 
  AlertTriangle, 
  CheckCircle 
} from 'lucide-react';

interface CudaDeviceInfo {
  device_id: number;
  name: string;
  total_memory: number;
  free_memory: number;
  compute_capability: [number, number];
  multiprocessor_count: number;
  max_threads_per_block: number;
  warp_size: number;
}

interface SystemInfo {
  cuda: {
    available: boolean;
    devices: CudaDeviceInfo[];
    best_device: number | null;
  };
  ffmpeg_available: boolean;
  temp_dir: string;
  version: string;
}

interface CudaSettingsProps {
  onSettingsChange: (settings: CudaSettings) => void;
  initialSettings?: CudaSettings;
}

export interface CudaSettings {
  use_cuda: boolean;
  cuda_device_id: number | null;
  preprocessing: {
    brightness: number;
    contrast: number;
    gamma: number;
    denoise: boolean;
  };
}

export function CudaSettings({ onSettingsChange, initialSettings }: CudaSettingsProps) {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [settings, setSettings] = useState<CudaSettings>(
    initialSettings || {
      use_cuda: false,
      cuda_device_id: null,
      preprocessing: {
        brightness: 0.0,
        contrast: 1.0,
        gamma: 1.0,
        denoise: false,
      },
    }
  );

  useEffect(() => {
    loadSystemInfo();
  }, []);

  useEffect(() => {
    onSettingsChange(settings);
  }, [settings, onSettingsChange]);

  const loadSystemInfo = async () => {
    try {
      setLoading(true);
      const info = await invoke<SystemInfo>('get_system_info');
      setSystemInfo(info);
      
      // Auto-enable CUDA if available and set best device
      if (info.cuda.available && info.cuda.best_device !== null) {
        setSettings(prev => ({
          ...prev,
          use_cuda: true,
          cuda_device_id: info.cuda.best_device,
        }));
      }
      
      setError(null);
    } catch (err) {
      setError(`Failed to load system info: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const formatMemory = (bytes: number): string => {
    const gb = bytes / (1024 * 1024 * 1024);
    return `${gb.toFixed(1)} GB`;
  };

  const formatComputeCapability = (capability: [number, number]): string => {
    return `${capability[0]}.${capability[1]}`;
  };

  const getPerformanceLevel = (device: CudaDeviceInfo): string => {
    const [major, minor] = device.compute_capability;
    const computeScore = major * 10 + minor;
    
    if (computeScore >= 89) return 'Excellent'; // RTX 40 series
    if (computeScore >= 86) return 'Very Good'; // RTX 30 series
    if (computeScore >= 75) return 'Good';      // RTX 20 series
    if (computeScore >= 60) return 'Fair';      // GTX 10 series
    return 'Basic';
  };

  const getPerformanceColor = (level: string): string => {
    switch (level) {
      case 'Excellent': return 'bg-green-100 text-green-800';
      case 'Very Good': return 'bg-blue-100 text-blue-800';
      case 'Good': return 'bg-yellow-100 text-yellow-800';
      case 'Fair': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            CUDA Acceleration
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            <span className="ml-2">Detecting CUDA devices...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-red-600">
            <AlertTriangle className="h-5 w-5" />
            CUDA Error
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* System Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Monitor className="h-5 w-5" />
            System Status
          </CardTitle>
          <CardDescription>
            CompressO-Cuda v{systemInfo?.version} system capabilities
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4" />
              <span className="text-sm">CUDA:</span>
              {systemInfo?.cuda.available ? (
                <Badge variant="default" className="bg-green-100 text-green-800">
                  <CheckCircle className="h-3 w-3 mr-1" />
                  Available
                </Badge>
              ) : (
                <Badge variant="secondary" className="bg-red-100 text-red-800">
                  <AlertTriangle className="h-3 w-3 mr-1" />
                  Not Available
                </Badge>
              )}
            </div>
            
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4" />
              <span className="text-sm">FFmpeg:</span>
              {systemInfo?.ffmpeg_available ? (
                <Badge variant="default" className="bg-green-100 text-green-800">
                  <CheckCircle className="h-3 w-3 mr-1" />
                  Available
                </Badge>
              ) : (
                <Badge variant="secondary" className="bg-red-100 text-red-800">
                  <AlertTriangle className="h-3 w-3 mr-1" />
                  Not Found
                </Badge>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* CUDA Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            CUDA Acceleration
          </CardTitle>
          <CardDescription>
            Configure GPU acceleration for faster video processing
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* CUDA Enable Toggle */}
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="cuda-enable">Enable CUDA Acceleration</Label>
              <p className="text-sm text-muted-foreground">
                Use GPU for faster video processing
              </p>
            </div>
            <Switch
              id="cuda-enable"
              checked={settings.use_cuda}
              onCheckedChange={(checked) =>
                setSettings(prev => ({ ...prev, use_cuda: checked }))
              }
              disabled={!systemInfo?.cuda.available}
            />
          </div>

          {/* Device Selection */}
          {settings.use_cuda && systemInfo?.cuda.available && (
            <div className="space-y-3">
              <Label>GPU Device</Label>
              <Select
                value={settings.cuda_device_id?.toString() || ''}
                onValueChange={(value) =>
                  setSettings(prev => ({ 
                    ...prev, 
                    cuda_device_id: parseInt(value) 
                  }))
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select GPU device" />
                </SelectTrigger>
                <SelectContent>
                  {systemInfo.cuda.devices.map((device) => (
                    <SelectItem 
                      key={device.device_id} 
                      value={device.device_id.toString()}
                    >
                      <div className="flex flex-col">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{device.name}</span>
                          <Badge 
                            variant="outline" 
                            className={getPerformanceColor(getPerformanceLevel(device))}
                          >
                            {getPerformanceLevel(device)}
                          </Badge>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {formatMemory(device.total_memory)} • 
                          Compute {formatComputeCapability(device.compute_capability)} • 
                          {device.multiprocessor_count} SMs
                        </div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {/* Selected Device Details */}
              {settings.cuda_device_id !== null && (
                <div className="mt-4">
                  {(() => {
                    const selectedDevice = systemInfo.cuda.devices.find(
                      d => d.device_id === settings.cuda_device_id
                    );
                    if (!selectedDevice) return null;

                    const memoryUsage = (
                      (selectedDevice.total_memory - selectedDevice.free_memory) / 
                      selectedDevice.total_memory
                    ) * 100;

                    return (
                      <Card className="bg-muted/50">
                        <CardContent className="pt-4">
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium">Memory Usage</span>
                              <span className="text-sm text-muted-foreground">
                                {formatMemory(selectedDevice.total_memory - selectedDevice.free_memory)} / 
                                {formatMemory(selectedDevice.total_memory)}
                              </span>
                            </div>
                            <Progress value={memoryUsage} className="h-2" />
                            
                            <div className="grid grid-cols-2 gap-4 text-xs">
                              <div>
                                <span className="text-muted-foreground">Compute:</span>
                                <span className="ml-1 font-medium">
                                  {formatComputeCapability(selectedDevice.compute_capability)}
                                </span>
                              </div>
                              <div>
                                <span className="text-muted-foreground">Max Threads:</span>
                                <span className="ml-1 font-medium">
                                  {selectedDevice.max_threads_per_block.toLocaleString()}
                                </span>
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })()}
                </div>
              )}
            </div>
          )}

          {/* No CUDA Warning */}
          {!systemInfo?.cuda.available && (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                CUDA is not available on this system. Video processing will use CPU only, 
                which may be slower. To enable CUDA acceleration, ensure you have:
                <ul className="mt-2 ml-4 list-disc text-sm">
                  <li>An NVIDIA GPU with compute capability 6.0 or higher</li>
                  <li>CUDA Toolkit 12.0 or later installed</li>
                  <li>Updated NVIDIA drivers</li>
                </ul>
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Preprocessing Settings */}
      {settings.use_cuda && (
        <Card>
          <CardHeader>
            <CardTitle>GPU Preprocessing</CardTitle>
            <CardDescription>
              Advanced image processing before compression
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Brightness */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="brightness">Brightness</Label>
                <span className="text-sm text-muted-foreground">
                  {settings.preprocessing.brightness > 0 ? '+' : ''}{settings.preprocessing.brightness.toFixed(2)}
                </span>
              </div>
              <input
                id="brightness"
                type="range"
                min="-0.5"
                max="0.5"
                step="0.01"
                value={settings.preprocessing.brightness}
                onChange={(e) =>
                  setSettings(prev => ({
                    ...prev,
                    preprocessing: {
                      ...prev.preprocessing,
                      brightness: parseFloat(e.target.value),
                    },
                  }))
                }
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            {/* Contrast */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="contrast">Contrast</Label>
                <span className="text-sm text-muted-foreground">
                  {settings.preprocessing.contrast.toFixed(2)}x
                </span>
              </div>
              <input
                id="contrast"
                type="range"
                min="0.5"
                max="2.0"
                step="0.01"
                value={settings.preprocessing.contrast}
                onChange={(e) =>
                  setSettings(prev => ({
                    ...prev,
                    preprocessing: {
                      ...prev.preprocessing,
                      contrast: parseFloat(e.target.value),
                    },
                  }))
                }
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            {/* Gamma */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="gamma">Gamma</Label>
                <span className="text-sm text-muted-foreground">
                  {settings.preprocessing.gamma.toFixed(2)}
                </span>
              </div>
              <input
                id="gamma"
                type="range"
                min="0.5"
                max="2.0"
                step="0.01"
                value={settings.preprocessing.gamma}
                onChange={(e) =>
                  setSettings(prev => ({
                    ...prev,
                    preprocessing: {
                      ...prev.preprocessing,
                      gamma: parseFloat(e.target.value),
                    },
                  }))
                }
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            {/* Denoise */}
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label htmlFor="denoise">Noise Reduction</Label>
                <p className="text-sm text-muted-foreground">
                  GPU-accelerated denoising filter
                </p>
              </div>
              <Switch
                id="denoise"
                checked={settings.preprocessing.denoise}
                onCheckedChange={(checked) =>
                  setSettings(prev => ({
                    ...prev,
                    preprocessing: {
                      ...prev.preprocessing,
                      denoise: checked,
                    },
                  }))
                }
              />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
