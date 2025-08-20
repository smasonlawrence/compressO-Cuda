import React, { useState, useEffect } from 'react';
import { listen } from '@tauri-apps/api/event';
import { invoke } from '@tauri-apps/api/tauri';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Play, 
  Pause, 
  Square, 
  Clock, 
  HardDrive, 
  Cpu, 
  Zap, 
  Activity,
  FileVideo,
  CheckCircle,
  XCircle,
  AlertTriangle,
  BarChart3,
  Timer,
  Gauge
} from 'lucide-react';

interface CompressionProgress {
  frame: number;
  total_frames: number;
  fps: number;
  bitrate: number;
  size: number;
  speed: number;
  time_elapsed: number;
  eta: number;
  percentage: number;
}

interface CompressionJob {
  id: string;
  input_path: string;
  output_path: string;
  status: 'Pending' | 'Running' | 'Completed' | 'Failed' | 'Cancelled';
  progress?: CompressionProgress;
  error?: string;
  created_at: string;
  updated_at: string;
}

interface PerformanceMetrics {
  gpu_utilization?: number;
  gpu_memory_used?: number;
  gpu_memory_total?: number;
  gpu_temperature?: number;
  cpu_usage?: number;
  memory_usage?: number;
}

interface CompressionProgressProps {
  jobId?: string;
  onJobComplete?: (success: boolean, outputPath?: string) => void;
  onJobCancel?: () => void;
  showPerformanceMetrics?: boolean;
}

export function CompressionProgress({ 
  jobId, 
  onJobComplete, 
  onJobCancel, 
  showPerformanceMetrics = true 
}: CompressionProgressProps) {
  const [job, setJob] = useState<CompressionJob | null>(null);
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (jobId) {
      loadJobStatus();
      
      // Listen for progress updates
      const progressUnlisten = listen<CompressionProgress>('compression-progress', (event) => {
        setJob(prev => prev ? { ...prev, progress: event.payload } : null);
      });

      // Performance metrics polling
      let metricsInterval: NodeJS.Timeout;
      if (showPerformanceMetrics) {
        metricsInterval = setInterval(updatePerformanceMetrics, 1000);
      }

      return () => {
        progressUnlisten.then(fn => fn());
        if (metricsInterval) clearInterval(metricsInterval);
      };
    }
  }, [jobId, showPerformanceMetrics]);

  useEffect(() => {
    // Check for job completion
    if (job?.status === 'Completed') {
      onJobComplete?.(true, job.output_path);
    } else if (job?.status === 'Failed') {
      onJobComplete?.(false);
    }
  }, [job?.status, job?.output_path, onJobComplete]);

  const loadJobStatus = async () => {
    if (!jobId) return;
    
    try {
      setIsLoading(true);
      const jobs = await invoke<CompressionJob[]>('get_compression_jobs');
      const currentJob = jobs.find(j => j.id === jobId);
      setJob(currentJob || null);
      setError(null);
    } catch (err) {
      setError(`Failed to load job status: ${err}`);
    } finally {
      setIsLoading(false);
    }
  };

  const updatePerformanceMetrics = async () => {
    try {
      // This would be implemented as a separate Tauri command to get system metrics
      // For now, we'll simulate some metrics
      setPerformanceMetrics({
        gpu_utilization: Math.random() * 100,
        gpu_memory_used: 4.2,
        gpu_memory_total: 8.0,
        gpu_temperature: 65 + Math.random() * 15,
        cpu_usage: 20 + Math.random() * 30,
        memory_usage: 60 + Math.random() * 20,
      });
    } catch (err) {
      console.warn('Failed to update performance metrics:', err);
    }
  };

  const cancelJob = async () => {
    if (!jobId) return;
    
    try {
      await invoke('cancel_compression', { jobId });
      onJobCancel?.();
    } catch (err) {
      setError(`Failed to cancel job: ${err}`);
    }
  };

  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    if (minutes < 60) return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return `${hours}h ${remainingMinutes}m`;
  };

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

  const getStatusIcon = (status: CompressionJob['status']) => {
    switch (status) {
      case 'Running':
        return <Activity className="h-5 w-5 text-blue-500 animate-pulse" />;
      case 'Completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'Failed':
        return <XCircle className="h-5 w-5 text-red-500" />;
      case 'Cancelled':
        return <Square className="h-5 w-5 text-gray-500" />;
      default:
        return <Clock className="h-5 w-5 text-yellow-500" />;
    }
  };

  const getStatusColor = (status: CompressionJob['status']) => {
    switch (status) {
      case 'Running':
        return 'bg-blue-100 text-blue-800';
      case 'Completed':
        return 'bg-green-100 text-green-800';
      case 'Failed':
        return 'bg-red-100 text-red-800';
      case 'Cancelled':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-yellow-100 text-yellow-800';
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          <span className="ml-2">Loading job status...</span>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (!job) {
    return (
      <Card>
        <CardContent className="text-center py-8">
          <p className="text-muted-foreground">No compression job found</p>
        </CardContent>
      </Card>
    );
  }

  const progress = job.progress;

  return (
    <div className="space-y-6">
      {/* Main Progress Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {getStatusIcon(job.status)}
              <div>
                <CardTitle>Video Compression</CardTitle>
                <CardDescription>
                  {job.input_path.split('/').pop() || job.input_path}
                </CardDescription>
              </div>
            </div>
            <Badge className={getStatusColor(job.status)}>
              {job.status}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Progress Bar */}
          {progress && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progress</span>
                <span>{progress.percentage.toFixed(1)}%</span>
              </div>
              <Progress value={progress.percentage} className="h-3" />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Frame {progress.frame.toLocaleString()} of {progress.total_frames.toLocaleString()}</span>
                <span>{formatTime(progress.eta)} remaining</span>
              </div>
            </div>
          )}

          {/* Real-time Stats */}
          {progress && job.status === 'Running' && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{progress.fps.toFixed(1)}</div>
                <div className="text-xs text-muted-foreground">FPS</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{progress.speed.toFixed(2)}x</div>
                <div className="text-xs text-muted-foreground">Speed</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">{progress.bitrate.toFixed(0)}</div>
                <div className="text-xs text-muted-foreground">kbps</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">{formatFileSize(progress.size)}</div>
                <div className="text-xs text-muted-foreground">Output Size</div>
              </div>
            </div>
          )}

          {/* Control Buttons */}
          <div className="flex gap-2">
            {job.status === 'Running' && (
              <Button 
                variant="outline" 
                size="sm" 
                onClick={cancelJob}
                className="flex items-center gap-2"
              >
                <Square className="h-4 w-4" />
                Cancel
              </Button>
            )}
            
            {job.status === 'Completed' && (
              <Button 
                variant="default" 
                size="sm"
                onClick={() => {
                  // Open output folder or file
                  invoke('open_file_location', { path: job.output_path });
                }}
                className="flex items-center gap-2"
              >
                <FileVideo className="h-4 w-4" />
                Open Output
              </Button>
            )}

            <Button 
              variant="ghost" 
              size="sm" 
              onClick={loadJobStatus}
              className="flex items-center gap-2"
            >
              <Activity className="h-4 w-4" />
              Refresh
            </Button>
          </div>

          {/* Error Message */}
          {job.status === 'Failed' && job.error && (
            <Alert variant="destructive">
              <XCircle className="h-4 w-4" />
              <AlertDescription>
                Compression failed: {job.error}
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      {showPerformanceMetrics && job.status === 'Running' && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Performance Metrics
            </CardTitle>
            <CardDescription>
              Real-time system performance during compression
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* GPU Metrics */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4 text-yellow-500" />
                  <span className="font-medium">GPU Performance</span>
                </div>
                
                {performanceMetrics.gpu_utilization !== undefined && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>GPU Utilization</span>
                      <span>{performanceMetrics.gpu_utilization.toFixed(0)}%</span>
                    </div>
                    <Progress value={performanceMetrics.gpu_utilization} className="h-2" />
                  </div>
                )}

                {performanceMetrics.gpu_memory_used !== undefined && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>GPU Memory</span>
                      <span>
                        {performanceMetrics.gpu_memory_used.toFixed(1)} / 
                        {performanceMetrics.gpu_memory_total?.toFixed(1)} GB
                      </span>
                    </div>
                    <Progress 
                      value={performanceMetrics.gpu_memory_total ? 
                        (performanceMetrics.gpu_memory_used / performanceMetrics.gpu_memory_total) * 100 : 0
                      } 
                      className="h-2" 
                    />
                  </div>
                )}

                {performanceMetrics.gpu_temperature !== undefined && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm">GPU Temperature</span>
                    <Badge variant={performanceMetrics.gpu_temperature > 80 ? "destructive" : "secondary"}>
                      {performanceMetrics.gpu_temperature.toFixed(0)}°C
                    </Badge>
                  </div>
                )}
              </div>

              {/* CPU/System Metrics */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-blue-500" />
                  <span className="font-medium">System Performance</span>
                </div>

                {performanceMetrics.cpu_usage !== undefined && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>CPU Usage</span>
                      <span>{performanceMetrics.cpu_usage.toFixed(0)}%</span>
                    </div>
                    <Progress value={performanceMetrics.cpu_usage} className="h-2" />
                  </div>
                )}

                {performanceMetrics.memory_usage !== undefined && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>RAM Usage</span>
                      <span>{performanceMetrics.memory_usage.toFixed(0)}%</span>
                    </div>
                    <Progress value={performanceMetrics.memory_usage} className="h-2" />
                  </div>
                )}

                {progress && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Encoding Speed</span>
                    <Badge variant={progress.speed > 1.0 ? "default" : "secondary"}>
                      {progress.speed.toFixed(2)}x realtime
                    </Badge>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Compression Statistics */}
      {job.status === 'Completed' && progress && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Gauge className="h-5 w-5 text-green-500" />
              Compression Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-green-600">{formatFileSize(progress.size)}</div>
                <div className="text-sm text-muted-foreground">Final Size</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-blue-600">{formatTime(progress.time_elapsed)}</div>
                <div className="text-sm text-muted-foreground">Total Time</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-purple-600">{progress.speed.toFixed(2)}x</div>
                <div className="text-sm text-muted-foreground">Avg Speed</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-orange-600">{progress.bitrate.toFixed(0)}</div>
                <div className="text-sm text-muted-foreground">Avg Bitrate (kbps)</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
