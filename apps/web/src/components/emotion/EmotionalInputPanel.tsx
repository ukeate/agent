import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
import { logger } from '../../utils/logger'
  Mic, 
  MicOff, 
  Video, 
  VideoOff, 
  Camera,
  Upload,
  Send,
  Activity,
  Heart,
  Brain,
  Zap
} from 'lucide-react';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

// 类型定义
export interface EmotionalInputData {
  text?: string;
  audioBlob?: Blob;
  videoBlob?: Blob;
  imageFile?: File;
  physiologicalData?: PhysiologicalData;
  timestamp: Date;
  modalities: ModalityType[];
}

export interface PhysiologicalData {
  heartRate?: number;
  skinConductance?: number;
  temperature?: number;
  brainActivity?: number[];
}

export enum ModalityType {
  TEXT = 'text',
  AUDIO = 'audio', 
  VIDEO = 'video',
  IMAGE = 'image',
  PHYSIOLOGICAL = 'physiological'
}

interface EmotionalInputPanelProps {
  onSubmit: (data: EmotionalInputData) => void;
  onRealTimeData?: (data: Partial<EmotionalInputData>) => void;
  isProcessing?: boolean;
  enabledModalities?: ModalityType[];
  className?: string;
}

// 录音状态管理
interface RecordingState {
  isRecording: boolean;
  audioLevel: number;
  duration: number;
  blob?: Blob;
}

// 视频录制状态
interface VideoRecordingState {
  isRecording: boolean;
  stream?: MediaStream;
  duration: number;
  blob?: Blob;
}

export const EmotionalInputPanel: React.FC<EmotionalInputPanelProps> = ({
  onSubmit,
  onRealTimeData,
  isProcessing = false,
  enabledModalities = [ModalityType.TEXT, ModalityType.AUDIO, ModalityType.VIDEO],
  className
}) => {
  // 状态管理
  const [activeTab, setActiveTab] = useState<ModalityType>(ModalityType.TEXT);
  const [textInput, setTextInput] = useState('');
  const [recordingState, setRecordingState] = useState<RecordingState>({
    isRecording: false,
    audioLevel: 0,
    duration: 0
  });
  const [videoState, setVideoState] = useState<VideoRecordingState>({
    isRecording: false,
    duration: 0
  });
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [physiologicalData, setPhysiologicalData] = useState<PhysiologicalData>({});

  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const videoRecorderRef = useRef<MediaRecorder | null>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const videoElementRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recordingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const videoTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const audioAnalyzerRef = useRef<AnalyserNode | null>(null);

  // 清理函数
  useEffect(() => {
    return () => {
      stopAudioRecording();
      stopVideoRecording();
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      if (videoTimerRef.current) clearInterval(videoTimerRef.current);
    };
  }, []);

  // 音频录制功能
  const startAudioRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100
        } 
      });
      
      audioStreamRef.current = stream;
      
      // 设置音频分析
      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      analyser.fftSize = 256;
      audioAnalyzerRef.current = analyser;
      
      // 开始音频电平监控
      const updateAudioLevel = () => {
        if (audioAnalyzerRef.current) {
          const dataArray = new Uint8Array(audioAnalyzerRef.current.frequencyBinCount);
          audioAnalyzerRef.current.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
          const level = (average / 255) * 100;
          
          setRecordingState(prev => ({ ...prev, audioLevel: level }));
          
          // 实时数据传递
          if (onRealTimeData) {
            onRealTimeData({
              timestamp: new Date(),
              modalities: [ModalityType.AUDIO]
            });
          }
        }
        
        if (recordingState.isRecording) {
          requestAnimationFrame(updateAudioLevel);
        }
      };
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      const chunks: Blob[] = [];
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        setRecordingState(prev => ({ 
          ...prev, 
          blob, 
          isRecording: false,
          audioLevel: 0 
        }));
        
        // 停止音频流
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(100); // 每100ms收集数据
      
      setRecordingState(prev => ({ 
        ...prev, 
        isRecording: true, 
        duration: 0 
      }));
      
      // 开始计时
      recordingTimerRef.current = setInterval(() => {
        setRecordingState(prev => ({ 
          ...prev, 
          duration: prev.duration + 1 
        }));
      }, 1000);
      
      updateAudioLevel();
      
    } catch (error) {
      logger.error('音频录制失败:', error);
      toast.error('无法访问麦克风，请检查权限设置');
    }
  }, [recordingState.isRecording, onRealTimeData]);

  const stopAudioRecording = useCallback(() => {
    if (mediaRecorderRef.current && recordingState.isRecording) {
      mediaRecorderRef.current.stop();
      
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }
      
      if (audioStreamRef.current) {
        audioStreamRef.current.getTracks().forEach(track => track.stop());
        audioStreamRef.current = null;
      }
    }
  }, [recordingState.isRecording]);

  // 视频录制功能
  const startVideoRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        },
        audio: true
      });
      
      if (videoElementRef.current) {
        videoElementRef.current.srcObject = stream;
      }
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'video/webm;codecs=vp9,opus'
      });
      
      const chunks: Blob[] = [];
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        setVideoState(prev => ({ 
          ...prev, 
          blob, 
          isRecording: false 
        }));
        
        // 停止视频流
        stream.getTracks().forEach(track => track.stop());
        if (videoElementRef.current) {
          videoElementRef.current.srcObject = null;
        }
      };
      
      videoRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      
      setVideoState(prev => ({ 
        ...prev, 
        isRecording: true, 
        stream, 
        duration: 0 
      }));
      
      // 开始计时
      videoTimerRef.current = setInterval(() => {
        setVideoState(prev => ({ 
          ...prev, 
          duration: prev.duration + 1 
        }));
        
        // 实时数据传递
        if (onRealTimeData) {
          onRealTimeData({
            timestamp: new Date(),
            modalities: [ModalityType.VIDEO]
          });
        }
      }, 1000);
      
    } catch (error) {
      logger.error('视频录制失败:', error);
      toast.error('无法访问摄像头，请检查权限设置');
    }
  }, [onRealTimeData]);

  const stopVideoRecording = useCallback(() => {
    if (videoRecorderRef.current && videoState.isRecording) {
      videoRecorderRef.current.stop();
      
      if (videoTimerRef.current) {
        clearInterval(videoTimerRef.current);
        videoTimerRef.current = null;
      }
      
      if (videoState.stream) {
        videoState.stream.getTracks().forEach(track => track.stop());
      }
    }
  }, [videoState.isRecording, videoState.stream]);

  // 图像处理
  const handleImageUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      if (file.size > 5 * 1024 * 1024) { // 5MB限制
        toast.error('图片大小不能超过5MB');
        return;
      }
      
      setSelectedImage(file);
      
      if (onRealTimeData) {
        onRealTimeData({
          imageFile: file,
          timestamp: new Date(),
          modalities: [ModalityType.IMAGE]
        });
      }
    } else {
      toast.error('请选择有效的图片文件');
    }
  }, [onRealTimeData]);

  const updatePhysiologicalData = useCallback((next: PhysiologicalData) => {
    setPhysiologicalData(next);
    if (onRealTimeData && Object.keys(next).length > 0) {
      onRealTimeData({
        physiologicalData: next,
        timestamp: new Date(),
        modalities: [ModalityType.PHYSIOLOGICAL]
      });
    }
  }, [onRealTimeData]);

  // 提交数据
  const handleSubmit = useCallback(() => {
    const activeModalities: ModalityType[] = [];
    const submissionData: EmotionalInputData = {
      timestamp: new Date(),
      modalities: []
    };

    // 收集各种模态的数据
    if (textInput.trim()) {
      submissionData.text = textInput.trim();
      activeModalities.push(ModalityType.TEXT);
    }

    if (recordingState.blob) {
      submissionData.audioBlob = recordingState.blob;
      activeModalities.push(ModalityType.AUDIO);
    }

    if (videoState.blob) {
      submissionData.videoBlob = videoState.blob;
      activeModalities.push(ModalityType.VIDEO);
    }

    if (selectedImage) {
      submissionData.imageFile = selectedImage;
      activeModalities.push(ModalityType.IMAGE);
    }

    if (Object.keys(physiologicalData).length > 0) {
      submissionData.physiologicalData = physiologicalData;
      activeModalities.push(ModalityType.PHYSIOLOGICAL);
    }

    if (activeModalities.length === 0) {
      toast.error('请至少输入一种模态的数据');
      return;
    }

    submissionData.modalities = activeModalities;
    onSubmit(submissionData);

    // 清理已提交的数据
    setTextInput('');
    setRecordingState({ isRecording: false, audioLevel: 0, duration: 0 });
    setVideoState({ isRecording: false, duration: 0 });
    setSelectedImage(null);
    setPhysiologicalData({});
  }, [textInput, recordingState.blob, videoState.blob, selectedImage, physiologicalData, onSubmit]);

  // 格式化时间显示
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          情感输入面板
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as ModalityType)}>
          <TabsList className="grid w-full grid-cols-5">
            {enabledModalities.includes(ModalityType.TEXT) && (
              <TabsTrigger value={ModalityType.TEXT}>文本</TabsTrigger>
            )}
            {enabledModalities.includes(ModalityType.AUDIO) && (
              <TabsTrigger value={ModalityType.AUDIO}>音频</TabsTrigger>
            )}
            {enabledModalities.includes(ModalityType.VIDEO) && (
              <TabsTrigger value={ModalityType.VIDEO}>视频</TabsTrigger>
            )}
            {enabledModalities.includes(ModalityType.IMAGE) && (
              <TabsTrigger value={ModalityType.IMAGE}>图像</TabsTrigger>
            )}
            {enabledModalities.includes(ModalityType.PHYSIOLOGICAL) && (
              <TabsTrigger value={ModalityType.PHYSIOLOGICAL}>生理</TabsTrigger>
            )}
          </TabsList>

          {/* 文本输入 */}
          <TabsContent value={ModalityType.TEXT} className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">文本输入</label>
              <Textarea
                placeholder="请描述您当前的感受或想表达的内容..."
                value={textInput}
                onChange={(e) => {
                  setTextInput(e.target.value);
                  if (onRealTimeData && e.target.value.trim()) {
                    onRealTimeData({
                      text: e.target.value,
                      timestamp: new Date(),
                      modalities: [ModalityType.TEXT]
                    });
                  }
                }}
                rows={4}
                className="resize-none"
              />
              {textInput && (
                <div className="flex justify-between text-sm text-muted-foreground">
                  <span>{textInput.length} 字符</span>
                  <Badge variant="outline">TEXT</Badge>
                </div>
              )}
            </div>
          </TabsContent>

          {/* 音频录制 */}
          <TabsContent value={ModalityType.AUDIO} className="space-y-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">音频录制</label>
                {recordingState.blob && (
                  <Badge variant="outline">已录制 {formatDuration(recordingState.duration)}</Badge>
                )}
              </div>
              
              <div className="flex items-center gap-4">
                <Button
                  onClick={recordingState.isRecording ? stopAudioRecording : startAudioRecording}
                  variant={recordingState.isRecording ? "destructive" : "default"}
                  size="lg"
                  disabled={isProcessing}
                >
                  {recordingState.isRecording ? (
                    <>
                      <MicOff className="h-4 w-4 mr-2" />
                      停止录制
                    </>
                  ) : (
                    <>
                      <Mic className="h-4 w-4 mr-2" />
                      开始录制
                    </>
                  )}
                </Button>
                
                {recordingState.isRecording && (
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center gap-2">
                      <Activity className="h-4 w-4 text-red-500 animate-pulse" />
                      <span className="text-sm font-mono">
                        {formatDuration(recordingState.duration)}
                      </span>
                    </div>
                    <div className="space-y-1">
                      <div className="text-xs text-muted-foreground">音量等级</div>
                      <Progress value={recordingState.audioLevel} className="h-2" />
                    </div>
                  </div>
                )}
              </div>
              
              {recordingState.blob && (
                <div className="p-3 bg-muted rounded-lg">
                  <audio 
                    controls 
                    src={URL.createObjectURL(recordingState.blob)}
                    className="w-full"
                  />
                </div>
              )}
            </div>
          </TabsContent>

          {/* 视频录制 */}
          <TabsContent value={ModalityType.VIDEO} className="space-y-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">视频录制</label>
                {videoState.blob && (
                  <Badge variant="outline">已录制 {formatDuration(videoState.duration)}</Badge>
                )}
              </div>
              
              <div className="space-y-4">
                <video
                  ref={videoElementRef}
                  autoPlay
                  muted
                  playsInline
                  className={cn(
                    "w-full aspect-video bg-black rounded-lg",
                    !videoState.isRecording && !videoState.blob && "hidden"
                  )}
                />
                
                {videoState.blob && !videoState.isRecording && (
                  <video
                    controls
                    src={URL.createObjectURL(videoState.blob)}
                    className="w-full aspect-video rounded-lg"
                  />
                )}
                
                <div className="flex items-center gap-4">
                  <Button
                    onClick={videoState.isRecording ? stopVideoRecording : startVideoRecording}
                    variant={videoState.isRecording ? "destructive" : "default"}
                    size="lg"
                    disabled={isProcessing}
                  >
                    {videoState.isRecording ? (
                      <>
                        <VideoOff className="h-4 w-4 mr-2" />
                        停止录制
                      </>
                    ) : (
                      <>
                        <Video className="h-4 w-4 mr-2" />
                        开始录制
                      </>
                    )}
                  </Button>
                  
                  {videoState.isRecording && (
                    <div className="flex items-center gap-2">
                      <div className="h-3 w-3 bg-red-500 rounded-full animate-pulse" />
                      <span className="text-sm font-mono">
                        {formatDuration(videoState.duration)}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </TabsContent>

          {/* 图像上传 */}
          <TabsContent value={ModalityType.IMAGE} className="space-y-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">图像上传</label>
                {selectedImage && (
                  <Badge variant="outline">{selectedImage.name}</Badge>
                )}
              </div>
              
              <div className="space-y-4">
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  variant="outline"
                  size="lg"
                  disabled={isProcessing}
                  className="w-full"
                >
                  <Upload className="h-4 w-4 mr-2" />
                  选择图片
                </Button>
                
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
                
                {selectedImage && (
                  <div className="space-y-2">
                    <img
                      src={URL.createObjectURL(selectedImage)}
                      alt="Selected"
                      className="w-full max-h-64 object-contain rounded-lg border"
                    />
                    <div className="text-sm text-muted-foreground">
                      {selectedImage.name} ({(selectedImage.size / 1024 / 1024).toFixed(2)} MB)
                    </div>
                  </div>
                )}
              </div>
            </div>
          </TabsContent>

          {/* 生理数据 */}
          <TabsContent value={ModalityType.PHYSIOLOGICAL} className="space-y-4">
            <div className="space-y-4">
              <label className="text-sm font-medium">生理信号</label>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">心率 (bpm)</label>
                  <input
                    type="number"
                    inputMode="numeric"
                    value={physiologicalData.heartRate ?? ''}
                    onChange={(e) => {
                      const v = e.target.value === '' ? undefined : Number(e.target.value);
                      const next = { ...physiologicalData };
                      if (v === undefined || Number.isNaN(v)) delete next.heartRate;
                      else next.heartRate = v;
                      updatePhysiologicalData(next);
                    }}
                    className="w-full px-3 py-2 border rounded-md text-sm bg-background"
                    disabled={isProcessing}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">皮肤电导 (μS)</label>
                  <input
                    type="number"
                    inputMode="decimal"
                    value={physiologicalData.skinConductance ?? ''}
                    onChange={(e) => {
                      const v = e.target.value === '' ? undefined : Number(e.target.value);
                      const next = { ...physiologicalData };
                      if (v === undefined || Number.isNaN(v)) delete next.skinConductance;
                      else next.skinConductance = v;
                      updatePhysiologicalData(next);
                    }}
                    className="w-full px-3 py-2 border rounded-md text-sm bg-background"
                    disabled={isProcessing}
                  />
                </div>
                <div className="space-y-1 sm:col-span-2">
                  <label className="text-xs text-muted-foreground">体温 (°C)</label>
                  <input
                    type="number"
                    inputMode="decimal"
                    value={physiologicalData.temperature ?? ''}
                    onChange={(e) => {
                      const v = e.target.value === '' ? undefined : Number(e.target.value);
                      const next = { ...physiologicalData };
                      if (v === undefined || Number.isNaN(v)) delete next.temperature;
                      else next.temperature = v;
                      updatePhysiologicalData(next);
                    }}
                    className="w-full px-3 py-2 border rounded-md text-sm bg-background"
                    disabled={isProcessing}
                  />
                </div>
                <div className="space-y-1 sm:col-span-2">
                  <label className="text-xs text-muted-foreground">脑电活动 (逗号分隔)</label>
                  <input
                    value={physiologicalData.brainActivity?.join(',') ?? ''}
                    onChange={(e) => {
                      const raw = e.target.value.trim();
                      const arr = raw
                        ? raw
                            .split(',')
                            .map((x) => Number(x.trim()))
                            .filter((n) => Number.isFinite(n))
                        : [];
                      const next = { ...physiologicalData };
                      if (arr.length === 0) delete next.brainActivity;
                      else next.brainActivity = arr;
                      updatePhysiologicalData(next);
                    }}
                    className="w-full px-3 py-2 border rounded-md text-sm bg-background"
                    disabled={isProcessing}
                  />
                </div>
              </div>
              
              {Object.keys(physiologicalData).length > 0 && (
                <div className="grid grid-cols-2 gap-4">
                  {physiologicalData.heartRate != null && (
                    <div className="p-3 bg-muted rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <Heart className="h-4 w-4 text-red-500" />
                        <span className="text-sm font-medium">心率</span>
                      </div>
                      <div className="text-2xl font-bold">
                        {physiologicalData.heartRate} <span className="text-sm font-normal">bpm</span>
                      </div>
                    </div>
                  )}
                  
                  {physiologicalData.skinConductance != null && (
                    <div className="p-3 bg-muted rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <Zap className="h-4 w-4 text-blue-500" />
                        <span className="text-sm font-medium">皮肤电导</span>
                      </div>
                      <div className="text-2xl font-bold">
                        {physiologicalData.skinConductance.toFixed(2)} <span className="text-sm font-normal">μS</span>
                      </div>
                    </div>
                  )}
                  
                  {physiologicalData.temperature != null && (
                    <div className="p-3 bg-muted rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <Activity className="h-4 w-4 text-orange-500" />
                        <span className="text-sm font-medium">体温</span>
                      </div>
                      <div className="text-2xl font-bold">
                        {physiologicalData.temperature.toFixed(1)} <span className="text-sm font-normal">°C</span>
                      </div>
                    </div>
                  )}
                  
                  {physiologicalData.brainActivity && (
                    <div className="p-3 bg-muted rounded-lg col-span-2">
                      <div className="flex items-center gap-2 mb-2">
                        <Brain className="h-4 w-4 text-purple-500" />
                        <span className="text-sm font-medium">脑电活动</span>
                      </div>
                      <div className="grid grid-cols-8 gap-1">
                        {physiologicalData.brainActivity.map((value, index) => (
                          <div key={index} className="text-center">
                            <div className="text-xs text-muted-foreground">
                              C{index + 1}
                            </div>
                            <div className="text-sm font-mono">
                              {value.toFixed(0)}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>

        {/* 提交按钮 */}
        <div className="flex justify-end pt-4 mt-4 border-t">
          <Button
            onClick={handleSubmit}
            disabled={isProcessing || (
              !textInput.trim() && 
              !recordingState.blob && 
              !videoState.blob && 
              !selectedImage && 
              Object.keys(physiologicalData).length === 0
            )}
            size="lg"
          >
            <Send className="h-4 w-4 mr-2" />
            {isProcessing ? '处理中...' : '提交分析'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};
