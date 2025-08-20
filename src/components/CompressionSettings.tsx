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
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Settings, 
  Zap, 
  Monitor, 
  Volume2, 
  VolumeX,
  Gauge,
  FileVideo,
  Info,
  Sparkles,
  Clock
} from 'lucide-react';

export interface CompressionSettings {
  quality: 'Low' | 'Medium' | 'High' | { Custom: number };
  resolution?: [number, number];
  bitrate?: number;
  framerate?: number;
  mute_audio: boolean;
  audio_bitrate?: number;
  codec: 'H264' | 'H265' | 'AV1' | 'VP9' | 'H264_NVENC' | 'H265_NVENC';
  preset: 'UltraFast' | 'SuperFast' | 'VeryFast' | 'Faster' | 'Fast' | 'Medium' | 'Slow' | 'Slower' | 'VerySlow' | 'P1' | 'P2' | 'P3' | 'P4' | 'P5' | 'P6' | 'P7';
  profile: 'Baseline' | 'Main' | 'High' | 'High10' | 'High422' | 'High444';
  level: 'L3_0' | 'L3_1' | 'L4_0' | 'L4_1' | 'L4_2' | 'L5_0' | 'L5_1' | 'L5_2';
  use_cuda: boolean;
  cuda_device_id?: number;
  preprocessing: {
    brightness: number;
    contrast: number;
    gamma: number;
    denoise: boolean;
  };
  container_format: 'MP4' | 'AVI' | 'MKV' | 'WebM' | 'MOV';
  two_pass: boolean;
}

interface CompressionSettingsProps {
  onSettingsChange: (settings: CompressionSettings) => void;
  initialSettings?: CompressionSettings;
  cudaAvailable: boolean;
  videoInfo?: {
    width: number;
    height: number;
    fps: number;
    duration: number;
    codec: string;
  };
}

interface QualityPreset {
  name: string;
  description: string;
  settings: Partial<CompressionSettings>;
  estimatedSize: string;
  estimatedTime: string;
  icon: React.ReactNode;
}

export function CompressionSettings({ 
  onSettingsChange, 
  initialSettings, 
  cudaAvailable,
  videoInfo 
}: CompressionSettingsProps) {
  const [settings, setSettings] = useState<CompressionSettings>(
    initialSettings || {
      quality: 'Medium',
      mute_audio: false,
      audio_bitrate: 128,
      codec: cudaAvailable ? 'H264_NVENC' : 'H264',
      preset: cudaAvailable ? 'P4' : 'Fast',
      profile: 'High',
      level: 'L4_1',
      use_cuda: cudaAvailable,
      preprocessing: {
        brightness: 0.0,
        contrast: 1.0,
        gamma: 1.0,
        denoise: false,
      },
      container_format: 'MP4',
      two_pass: false,
    }
  );

  const [activeTab, setActiveTab] = useState('basic');
  const [estimatedOutput, setEstimatedOutput] = useState<{
    size: string;
    time: string;
    quality: string;
  } | null>(null);

  useEffect(() => {
    onSettingsChange(settings);
    calculateEstimates();
  }, [settings, onSettingsChange]);

  const qualityPresets: QualityPreset[] = [
    {
      name: 'Fast',
      description: 'Quick compression for previews',
      settings: {
        quality: 'Medium',
        codec: cudaAvailable ? 'H264_NVENC' : 'H264',
        preset: cudaAvailable ? 'P1' : 'VeryFast',
        two_pass: false,
      },
      estimatedSize: '~70% smaller',
      estimatedTime: '2-3x realtime',
      icon: <Clock className="h-4 w-4" />,
    },
    {
      name: 'Balanced',
      description: 'Good quality and reasonable speed',
      settings: {
        quality: 'Medium',
        codec: cudaAvailable ? 'H264_NVENC' : 'H264',
        preset: cudaAvailable ? 'P4' : 'Fast',
        two_pass: false,
      },
      estimatedSize: '~75% smaller',
      estimatedTime: '1-2x realtime',
      icon: <Gauge className="h-4 w-4" />,
    },
    {
      name: 'High Quality',
      description: 'Best quality for archival',
      settings: {
        quality: 'High',
        codec: cudaAvailable ? 'H265_NVENC' : 'H265',
        preset: cudaAvailable ? 'P6' : 'Slow',
        two_pass: true,
      },
      estimatedSize: '~80% smaller',
      estimatedTime: '0.5-1x realtime',
      icon: <Sparkles className="h-4 w-4" />,
    },
    {
      name: 'Small Size',
      description: 'Maximum compression',
      settings: {
        quality: 'Low',
        codec: cudaAvailable ? 'H265_NVENC' : 'H265',
        preset: cudaAvailable ? 'P7' : 'VerySlow',
        two_pass: true,
      },
      estimatedSize: '~85% smaller',
      estimatedTime: '0.3-0.8x realtime',
      icon: <FileVideo className="h-4 w-4" />,
    },
  ];

  const calculateEstimates = () => {
    if (!videoInfo) return;

    // Rough estimation based on codec and quality
    const baseSize = videoInfo.width * videoInfo.height * videoInfo.duration * 0.1; // Base size estimate
    let compressionRatio = 0.7; // Default 70% reduction

    // Adjust for quality
    switch (settings.quality) {
      case 'Low':
        compressionRatio = 0.85;
        break;
      case 'Medium':
        compressionRatio = 0.75;
        break;
      case 'High':
        compressionRatio = 0.65;
        break;
      default:
        if (typeof settings.quality === 'object' && 'Custom' in settings.quality) {
          const crf = settings.quality.Custom;
          compressionRatio = 0.9 - (crf / 51) * 0.4; // CRF 0 = 50% reduction, CRF 51 = 90% reduction
        }
    }

    // Adjust for codec
    if (settings.codec.includes('H265') || settings.codec === 'AV1') {
      compressionRatio += 0.1; // Better compression
    }

    const estimatedSize = (baseSize * (1 - compressionRatio) / 1024 / 1024).toFixed(1);
    
    // Time estimation
    let timeMultiplier = 1.0;
    if (cudaAvailable && settings.use_cuda) {
      timeMultiplier = settings.preset.startsWith('P') ? 
        3.0 - (parseInt(settings.preset.slice(1)) * 0.3) : // P1=2.7x, P7=0.9x realtime
        2.0; // Default CUDA speedup
    } else {
      const cpuSpeeds: { [key: string]: number } = {
        'UltraFast': 4.0, 'SuperFast': 3.0, 'VeryFast': 2.5, 'Faster': 2.0,
        'Fast': 1.5, 'Medium': 1.0, 'Slow': 0.7, 'Slower': 0.5, 'VerySlow': 0.3
      };
      timeMultiplier = cpuSpeeds[settings.preset] || 1.0;
    }

    const estimatedTime = (videoInfo.duration / timeMultiplier / 60).toFixed(1);
    
    setEstimatedOutput({
      size: `${estimatedSize} MB`,
      time: `${estimatedTime} min`,
      quality: typeof settings.quality === 'string' ? settings.quality : 'Custom',
    });
  };

  const applyPreset = (preset: QualityPreset) => {
    setSettings(prev => ({ ...prev, ...preset.settings }));
  };

  const getQualityValue = (): number => {
    if (typeof settings.quality === 'string') {
      switch (settings.quality) {
        case 'High': return 18;
        case 'Medium': return 23;
        case 'Low': return 28;
        default: return 23;
      }
    }
    return settings.quality.Custom;
  };

  const setQualityValue = (value: number) => {
    setSettings(prev => ({ ...prev, quality: { Custom: value } }));
  };

  const getCodecOptions = () => {
    const baseCodecs = [
      { value: 'H264', label: 'H.264 (CPU)', description: 'Universal compatibility' },
      { value: 'H265', label: 'H.265 (CPU)', description: 'Better compression' },
      { value: 'AV1', label: 'AV1 (CPU)', description: 'Next-gen codec' },
      { value: 'VP9', label: 'VP9 (CPU)', description: 'Web optimized' },
    ];

    if (cudaAvailable) {
      return [
        { value: 'H264_NVENC', label: 'H.264 (GPU)', description: 'Fast GPU encoding', recommended: true },
        { value: 'H265_NVENC', label: 'H.265 (GPU)', description: 'GPU with better compression', recommended: true },
        ...baseCodecs,
      ];
    }

    return baseCodecs;
  };

  const getPresetOptions = () => {
    if (cudaAvailable && settings.codec.includes('NVENC')) {
      return [
        { value: 'P1', label: 'P1 (Fastest)', description: 'Maximum speed' },
        { value: 'P2', label: 'P2', description: 'Very fast' },
        { value: 'P3', label: 'P3', description: 'Fast' },
        { value: 'P4', label: 'P4 (Balanced)', description: 'Good speed/quality', recommended: true },
        { value: 'P5', label: 'P5', description: 'Better quality' },
        { value: 'P6', label: 'P6', description: 'High quality' },
        { value: 'P7', label: 'P7 (Best)', description: 'Maximum quality' },
      ];
    }

    return [
      { value: 'UltraFast', label: 'Ultra Fast', description: 'Fastest encoding' },
      { value: 'SuperFast', label: 'Super Fast', description: 'Very fast' },
      { value: 'VeryFast', label: 'Very Fast', description: 'Fast encoding' },
      { value: 'Faster', label: 'Faster', description: 'Faster than fast' },
      { value: 'Fast', label: 'Fast', description: 'Good speed', recommended: true },
      { value: 'Medium', label: 'Medium', description: 'Balanced' },
      { value: 'Slow', label: 'Slow', description: 'Better quality' },
      { value: 'Slower', label: 'Slower', description: 'High quality' },
      { value: 'VerySlow', label: 'Very Slow', description: 'Best quality' },
    ];
  };

  return (
    <div className="space-y-6">
      {/* Quick Presets */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Presets</CardTitle>
          <CardDescription>
            Choose a preset to get started quickly
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-3">
            {qualityPresets.map((preset) => (
              <Button
                key={preset.name}
                variant="outline"
                className="h-auto p-4 flex flex-col items-start gap-2"
                onClick={() => applyPreset(preset)}
              >
                <div className="flex items-center gap-2 w-full">
                  {preset.icon}
                  <span className="font-medium">{preset.name}</span>
                  {cudaAvailable && (
                    <Zap className="h-3 w-3 text-yellow-500 ml-auto" />
                  )}
                </div>
                <p className="text-xs text-muted-foreground text-left">
                  {preset.description}
                </p>
                <div className="flex gap-2 text-xs">
                  <Badge variant="secondary">{preset.estimatedSize}</Badge>
                  <Badge variant="outline">{preset.estimatedTime}</Badge>
                </div>
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Detailed Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Compression Settings
          </CardTitle>
          <CardDescription>
            Fine-tune compression parameters
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="basic">Basic</TabsTrigger>
              <TabsTrigger value="video">Video</TabsTrigger>
              <TabsTrigger value="advanced">Advanced</TabsTrigger>
            </TabsList>

            {/* Basic Settings */}
            <TabsContent value="basic" className="space-y-6 mt-6">
              {/* Quality */}
              <div className="space-y-3">
                <Label>Quality (CRF)</Label>
                <div className="space-y-2">
                  <Slider
                    value={[getQualityValue()]}
                    onValueChange={([value]) => setQualityValue(value)}
                    min={0}
                    max={51}
                    step={1}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Best Quality (0)</span>
                    <span className="font-medium">CRF: {getQualityValue()}</span>
                    <span>Smallest Size (51)</span>
                  </div>
                </div>
              </div>

              {/* Codec */}
              <div className="space-y-3">
                <Label>Video Codec</Label>
                <Select
                  value={settings.codec}
                  onValueChange={(value: any) =>
                    setSettings(prev => ({ ...prev, codec: value }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {getCodecOptions().map((codec) => (
                      <SelectItem key={codec.value} value={codec.value}>
                        <div className="flex items-center gap-2">
                          <div>
                            <div className="flex items-center gap-2">
                              {codec.label}
                              {codec.recommended && (
                                <Badge variant="default" className="text-xs">
                                  Recommended
                                </Badge>
                              )}
                              {codec.value.includes('NVENC') && (
                                <Zap className="h-3 w-3 text-yellow-500" />
                              )}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {codec.description}
                            </div>
                          </div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Container Format */}
              <div className="space-y-3">
                <Label>Output Format</Label>
                <Select
                  value={settings.container_format}
                  onValueChange={(value: any) =>
                    setSettings(prev => ({ ...prev, container_format: value }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="MP4">MP4 (Recommended)</SelectItem>
                    <SelectItem value="MKV">MKV (High compatibility)</SelectItem>
                    <SelectItem value="WebM">WebM (Web optimized)</SelectItem>
                    <SelectItem value="AVI">AVI (Legacy)</SelectItem>
                    <SelectItem value="MOV">MOV (Apple/editing)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </TabsContent>

            {/* Video Settings */}
            <TabsContent value="video" className="space-y-6 mt-6">
              {/* Resolution */}
              <div className="space-y-3">
                <Label>Resolution</Label>
                <Select
                  value={settings.resolution ? `${settings.resolution[0]}x${settings.resolution[1]}` : 'original'}
                  onValueChange={(value) => {
                    if (value === 'original') {
                      setSettings(prev => ({ ...prev, resolution: undefined }));
                    } else {
                      const [width, height] = value.split('x').map(Number);
                      setSettings(prev => ({ ...prev, resolution: [width, height] }));
                    }
                  }}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="original">Original Resolution</SelectItem>
                    <SelectItem value="3840x2160">4K (3840×2160)</SelectItem>
                    <SelectItem value="2560x1440">1440p (2560×1440)</SelectItem>
                    <SelectItem value="1920x1080">1080p (1920×1080)</SelectItem>
                    <SelectItem value="1280x720">720p (1280×720)</SelectItem>
                    <SelectItem value="854x480">480p (854×480)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Framerate */}
              <div className="space-y-3">
                <Label>Frame Rate</Label>
                <div className="flex gap-2">
                  <Input
                    type="number"
                    placeholder={videoInfo?.fps.toString() || "Original"}
                    value={settings.framerate || ''}
                    onChange={(e) =>
                      setSettings(prev => ({
                        ...prev,
                        framerate: e.target.value ? parseFloat(e.target.value) : undefined,
                      }))
                    }
                    min="1"
                    max="120"
                    step="0.1"
                  />
                  <span className="flex items-center text-sm text-muted-foreground">fps</span>
                </div>
              </div>

              {/* Bitrate */}
              <div className="space-y-3">
                <Label>Target Bitrate (optional)</Label>
                <div className="flex gap-2">
                  <Input
                    type="number"
                    placeholder="Auto"
                    value={settings.bitrate || ''}
                    onChange={(e) =>
                      setSettings(prev => ({
                        ...prev,
                        bitrate: e.target.value ? parseInt(e.target.value) : undefined,
                      }))
                    }
                    min="100"
                    max="50000"
                  />
                  <span className="flex items-center text-sm text-muted-foreground">kbps</span>
                </div>
              </div>

              {/* Audio Settings */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {settings.mute_audio ? (
                      <VolumeX className="h-4 w-4" />
                    ) : (
                      <Volume2 className="h-4 w-4" />
                    )}
                    <Label>Audio</Label>
                  </div>
                  <Switch
                    checked={!settings.mute_audio}
                    onCheckedChange={(checked) =>
                      setSettings(prev => ({ ...prev, mute_audio: !checked }))
                    }
                  />
                </div>

                {!settings.mute_audio && (
                  <div className="space-y-2">
                    <Label>Audio Bitrate</Label>
                    <Select
                      value={settings.audio_bitrate?.toString() || '128'}
                      onValueChange={(value) =>
                        setSettings(prev => ({ ...prev, audio_bitrate: parseInt(value) }))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="64">64 kbps (Low)</SelectItem>
                        <SelectItem value="128">128 kbps (Standard)</SelectItem>
                        <SelectItem value="192">192 kbps (High)</SelectItem>
                        <SelectItem value="256">256 kbps (Very High)</SelectItem>
                        <SelectItem value="320">320 kbps (Maximum)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>
            </TabsContent>

            {/* Advanced Settings */}
            <TabsContent value="advanced" className="space-y-6 mt-6">
              {/* Encoding Preset */}
              <div className="space-y-3">
                <Label>Encoding Preset</Label>
                <Select
                  value={settings.preset}
                  onValueChange={(value: any) =>
                    setSettings(prev => ({ ...prev, preset: value }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {getPresetOptions().map((preset) => (
                      <SelectItem key={preset.value} value={preset.value}>
                        <div className="flex flex-col">
                          <div className="flex items-center gap-2">
                            {preset.label}
                            {preset.recommended && (
                              <Badge variant="default" className="text-xs">
                                Recommended
                              </Badge>
                            )}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {preset.description}
                          </div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Profile */}
              <div className="space-y-3">
                <Label>H.264 Profile</Label>
                <Select
                  value={settings.profile}
                  onValueChange={(value: any) =>
                    setSettings(prev => ({ ...prev, profile: value }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Baseline">Baseline (Most compatible)</SelectItem>
                    <SelectItem value="Main">Main (Good compatibility)</SelectItem>
                    <SelectItem value="High">High (Recommended)</SelectItem>
                    <SelectItem value="High10">High 10-bit</SelectItem>
                    <SelectItem value="High422">High 4:2:2</SelectItem>
                    <SelectItem value="High444">High 4:4:4</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Two-pass encoding */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Two-pass Encoding</Label>
                  <p className="text-sm text-muted-foreground">
                    Better quality but slower encoding
                  </p>
                </div>
                <Switch
                  checked={settings.two_pass}
                  onCheckedChange={(checked) =>
                    setSettings(prev => ({ ...prev, two_pass: checked }))
                  }
                />
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Output Estimation */}
      {estimatedOutput && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Info className="h-5 w-5" />
              Estimated Output
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold">{estimatedOutput.size}</div>
                <div className="text-sm text-muted-foreground">File Size</div>
              </div>
              <div>
                <div className="text-2xl font-bold">{estimatedOutput.time}</div>
                <div className="text-sm text-muted-foreground">
                  Compression Time
                  {cudaAvailable && settings.use_cuda && (
                    <Zap className="h-3 w-3 inline ml-1 text-yellow-500" />
                  )}
                </div>
              </div>
              <div>
                <div className="text-2xl font-bold">{estimatedOutput.quality}</div>
                <div className="text-sm text-muted-foreground">Quality Level</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* CUDA Status */}
      {cudaAvailable && (
        <Alert>
          <Zap className="h-4 w-4" />
          <AlertDescription>
            CUDA acceleration is available! GPU encoding will be significantly faster than CPU encoding.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
