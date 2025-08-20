import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { 
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { 
  Play, 
  Pause, 
  Square, 
  Trash2, 
  MoreHorizontal,
  RefreshCw,
  Download,
  FolderOpen,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Activity,
  Zap,
  Cpu,
  Timer,
  FileVideo,
  Plus,
  Settings,
  BarChart3
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
  settings: {
    codec: string;
    quality: string | { Custom: number };
    use_cuda: boolean;
    container_format: string;
  };
  status: 'Pending' | 'Running' | 'Completed' | 'Failed' | 'Cancelled';
  progress?: CompressionProgress;
  error?: string;
  created_at: string;
  updated_at: string;
}

interface QueueStats {
  total_jobs: number;
  completed: number;
  failed: number;
  running: number;
  pending: number;
  total_time_saved?: number; // Time saved with CUDA vs CPU
  avg_speed_improvement?: number; // Average speed improvement with CUDA
}

interface JobQueueManagerProps {
  onJobSelect?: (jobId: string) => void;
  onNewJob?: () => void;
  showCudaMetrics?: boolean;
}

export function JobQueueManager({ 
  onJobSelect, 
  onNewJob, 
  showCudaMetrics = true 
}: JobQueueManagerProps) {
  const [jobs, setJobs] = useState<CompressionJob[]>([]);
  const [queueStats, setQueueStats] = useState<QueueStats>({
    total_jobs: 0,
    completed: 0,
    failed: 0,
    running: 0,
    pending: 0,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'created_at' | 'status' | 'progress'>('created_at');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  useEffect(() => {
    loadJobs();
    
    // Listen for job updates
    const unlistenProgress = listen('compression-progress', () => {
      loadJobs(); // Refresh jobs when progress updates
    });

    // Auto-refresh every 5 seconds
    const interval = setInterval(loadJobs, 5000);

    return () => {
      unlistenProgress.then(fn => fn());
      clearInterval(interval);
    };
  }, []);

  const loadJobs = async () => {
    try {
      setIsLoading(true);
      const jobList = await invoke<CompressionJob[]>('get_compression_jobs');
      setJobs(jobList);
      calculateQueueStats(jobList);
      setError(null);
    } catch (err) {
      setError(`Failed to load jobs: ${err}`);
    } finally {
      setIsLoading(false);
    }
  };

  const calculateQueueStats = (jobList: CompressionJob[]) => {
    const stats: QueueStats = {
      total_jobs: jobList.length,
      completed: jobList.filter(j => j.status === 'Completed').length,
      failed: jobList.filter(j => j.status === 'Failed').length,
      running: jobList.filter(j => j.status === 'Running').length,
      pending: jobList.filter(j => j.status === 'Pending').length,
    };

    // Calculate CUDA performance benefits
    const cudaJobs = jobList.filter(j => j.settings.use_cuda && j.status === 'Completed');
    const cpuJobs = jobList.filter(j => !j.settings.use_cuda && j.status === 'Completed');

    if (cudaJobs.length > 0 && cpuJobs.length > 0) {
      const avgCudaSpeed = cudaJobs.reduce((sum, job) => 
        sum + (job.progress?.speed || 0), 0) / cudaJobs.length;
      const avgCpuSpeed = cpuJobs.reduce((sum, job) => 
        sum + (job.progress?.speed || 0), 0) / cpuJobs.length;
      
      stats.avg_speed_improvement = avgCudaSpeed / avgCpuSpeed;
      
      // Estimate time saved
      const cudaTime = cudaJobs.reduce((sum, job) => 
        sum + (job.progress?.time_elapsed || 0), 0);
      const estimatedCpuTime = cudaTime * (stats.avg_speed_improvement || 1);
      stats.total_time_saved = estimatedCpuTime - cudaTime;
    }

    setQueueStats(stats);
  };

  const cancelJob = async (jobId: string) => {
    try {
      await invoke('cancel_compression', { jobId });
      loadJobs();
    } catch (err) {
      setError(`Failed to cancel job: ${err}`);
    }
  };

  const removeJob = async (jobId: string) => {
    try {
      await invoke('remove_compression_job', { jobId });
      loadJobs();
    } catch (err) {
      setError(`Failed to remove job: ${err}`);
    }
  };

  const clearCompletedJobs = async () => {
    try {
      await invoke('clear_completed_jobs');
      loadJobs();
    } catch (err) {
      setError(`Failed to clear completed jobs: ${err}`);
    }
  };

  const openFileLocation = async (path: string) => {
    try {
      await invoke('open_file_location', { path });
    } catch (err) {
      console.error('Failed to open file location:', err);
    }
  };

  const getStatusIcon = (status: CompressionJob['status']) => {
    switch (status) {
      case 'Running':
        return <Activity className="h-4 w-4 text-blue-500 animate-pulse" />;
      case 'Completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'Failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'Cancelled':
        return <Square className="h-4 w-4 text-gray-500" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-500" />;
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

  const getFileName = (path: string): string => {
    return path.split('/').pop() || path.split('\\').pop() || path;
  };

  const filteredJobs = jobs.filter(job => 
    filterStatus === 'all' || job.status.toLowerCase() === filterStatus
  );

  const sortedJobs = [...filteredJobs].sort((a, b) => {
    switch (sortBy) {
      case 'created_at':
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      case 'status':
        return a.status.localeCompare(b.status);
      case 'progress':
        return (b.progress?.percentage || 0) - (a.progress?.percentage || 0);
      default:
        return 0;
    }
  });

  return (
    <div className="space-y-6">
      {/* Queue Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <FileVideo className="h-5 w-5 text-blue-500" />
              <div>
                <div className="text-2xl font-bold">{queueStats.total_jobs}</div>
                <div className="text-sm text-muted-foreground">Total Jobs</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-green-500" />
              <div>
                <div className="text-2xl font-bold">{queueStats.running}</div>
                <div className="text-sm text-muted-foreground">Running</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <div>
                <div className="text-2xl font-bold">{queueStats.completed}</div>
                <div className="text-sm text-muted-foreground">Completed</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Clock className="h-5 w-5 text-yellow-500" />
              <div>
                <div className="text-2xl font-bold">{queueStats.pending}</div>
                <div className="text-sm text-muted-foreground">Pending</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* CUDA Performance Metrics */}
      {showCudaMetrics && (queueStats.avg_speed_improvement || queueStats.total_time_saved) && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-yellow-500" />
              CUDA Performance Benefits
            </CardTitle>
            <CardDescription>
              Acceleration metrics compared to CPU encoding
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {queueStats.avg_speed_improvement && (
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-600">
                    {queueStats.avg_speed_improvement.toFixed(1)}x
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Average Speed Improvement
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    GPU vs CPU encoding speed
                  </div>
                </div>
              )}

              {queueStats.total_time_saved && (
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-600">
                    {formatTime(queueStats.total_time_saved)}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Total Time Saved
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Cumulative time savings
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Job Queue */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Compression Queue
              </CardTitle>
              <CardDescription>
                Manage and monitor your compression jobs
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={loadJobs}
                disabled={isLoading}
              >
                <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                onClick={clearCompletedJobs}
                disabled={queueStats.completed === 0}
              >
                Clear Completed
              </Button>

              <Button
                variant="default"
                size="sm"
                onClick={onNewJob}
              >
                <Plus className="h-4 w-4 mr-2" />
                New Job
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {/* Filters */}
          <div className="flex gap-2 mb-4">
            <Button
              variant={filterStatus === 'all' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilterStatus('all')}
            >
              All ({jobs.length})
            </Button>
            <Button
              variant={filterStatus === 'running' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilterStatus('running')}
            >
              Running ({queueStats.running})
            </Button>
            <Button
              variant={filterStatus === 'completed' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilterStatus('completed')}
            >
              Completed ({queueStats.completed})
            </Button>
            <Button
              variant={filterStatus === 'failed' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilterStatus('failed')}
            >
              Failed ({queueStats.failed})
            </Button>
          </div>

          {/* Jobs Table */}
          <div className="border rounded-lg">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>File</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Progress</TableHead>
                  <TableHead>Speed</TableHead>
                  <TableHead>Output</TableHead>
                  <TableHead>Settings</TableHead>
                  <TableHead className="w-[100px]">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sortedJobs.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center py-8">
                      <div className="text-muted-foreground">
                        {filterStatus === 'all' ? 'No compression jobs yet' : `No ${filterStatus} jobs`}
                      </div>
                      {filterStatus === 'all' && (
                        <Button
                          variant="link"
                          onClick={onNewJob}
                          className="mt-2"
                        >
                          Start your first compression job
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ) : (
                  sortedJobs.map((job) => (
                    <TableRow 
                      key={job.id} 
                      className="cursor-pointer hover:bg-muted/50"
                      onClick={() => onJobSelect?.(job.id)}
                    >
                      <TableCell>
                        <div className="flex flex-col gap-1">
                          <div className="font-medium truncate max-w-[200px]">
                            {getFileName(job.input_path)}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {new Date(job.created_at).toLocaleString()}
                          </div>
                        </div>
                      </TableCell>

                      <TableCell>
                        <div className="flex items-center gap-2">
                          {getStatusIcon(job.status)}
                          <Badge className={getStatusColor(job.status)}>
                            {job.status}
                          </Badge>
                        </div>
                      </TableCell>

                      <TableCell>
                        {job.progress && job.status === 'Running' ? (
                          <div className="space-y-1">
                            <Progress value={job.progress.percentage} className="h-2 w-20" />
                            <div className="text-xs text-muted-foreground">
                              {job.progress.percentage.toFixed(0)}% • {formatTime(job.progress.eta)} left
                            </div>
                          </div>
                        ) : job.status === 'Completed' ? (
                          <div className="text-sm text-green-600">100%</div>
                        ) : (
                          <div className="text-sm text-muted-foreground">-</div>
                        )}
                      </TableCell>

                      <TableCell>
                        <div className="flex items-center gap-1">
                          {job.progress?.speed && (
                            <>
                              <span className="text-sm font-medium">
                                {job.progress.speed.toFixed(1)}x
                              </span>
                              {job.settings.use_cuda && (
                                <Zap className="h-3 w-3 text-yellow-500" />
                              )}
                            </>
                          )}
                        </div>
                      </TableCell>

                      <TableCell>
                        {job.progress?.size ? (
                          <div className="text-sm">
                            {formatFileSize(job.progress.size)}
                          </div>
                        ) : (
                          <div className="text-sm text-muted-foreground">-</div>
                        )}
                      </TableCell>

                      <TableCell>
                        <div className="flex flex-col gap-1">
                          <div className="flex items-center gap-1">
                            <Badge variant="outline" className="text-xs">
                              {job.settings.codec}
                            </Badge>
                            {job.settings.use_cuda && (
                              <Zap className="h-3 w-3 text-yellow-500" />
                            )}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {job.settings.container_format}
                          </div>
                        </div>
                      </TableCell>

                      <TableCell onClick={(e) => e.stopPropagation()}>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="sm">
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            {job.status === 'Running' && (
                              <DropdownMenuItem onClick={() => cancelJob(job.id)}>
                                <Square className="h-4 w-4 mr-2" />
                                Cancel
                              </DropdownMenuItem>
                            )}
                            
                            {job.status === 'Completed' && (
                              <>
                                <DropdownMenuItem onClick={() => openFileLocation(job.output_path)}>
                                  <FolderOpen className="h-4 w-4 mr-2" />
                                  Open Location
                                </DropdownMenuItem>
                                <DropdownMenuItem onClick={() => openFileLocation(job.output_path)}>
                                  <Download className="h-4 w-4 mr-2" />
                                  Open File
                                </DropdownMenuItem>
                              </>
                            )}

                            {(job.status === 'Completed' || job.status === 'Failed' || job.status === 'Cancelled') && (
                              <DropdownMenuItem 
                                onClick={() => removeJob(job.id)}
                                className="text-red-600"
                              >
                                <Trash2 className="h-4 w-4 mr-2" />
                                Remove
                              </DropdownMenuItem>
                            )}
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}
