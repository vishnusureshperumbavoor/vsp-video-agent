import React, { useRef, useState } from 'react';
import { Play, Pause, RotateCcw, Volume2, VolumeX, Download, Maximize, Film, Sparkles } from 'lucide-react';
import { VideoMetadata, VideoJob } from '../services/api';

interface VideoPlayerProps {
  currentVideo: VideoMetadata | null;
  activeJob: VideoJob | null;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({ currentVideo, activeJob }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isLoop, setIsLoop] = useState(true);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      videoRef.current.play();
      setIsPlaying(true);
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  const toggleMute = () => {
    if (!videoRef.current) return;
    videoRef.current.muted = !videoRef.current.muted;
    setIsMuted(videoRef.current.muted);
  };

  const handleSpeedChange = () => {
    if (!videoRef.current) return;
    const speeds = [0.5, 1, 1.5, 2];
    const nextIdx = (speeds.indexOf(playbackRate) + 1) % speeds.length;
    const nextSpeed = speeds[nextIdx];
    videoRef.current.playbackRate = nextSpeed;
    setPlaybackRate(nextSpeed);
  };

  const handleFullscreen = () => {
    if (!videoRef.current) return;
    if (videoRef.current.requestFullscreen) {
      videoRef.current.requestFullscreen();
    }
  };

  const isGenerating = activeJob?.status === 'running' || activeJob?.status === 'queued';

  return (
    <div className="bg-[#111827] border border-gray-800 rounded-2xl overflow-hidden flex flex-col shadow-2xl">
      {/* Video Display Area */}
      <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden group">
        {isGenerating ? (
          <div className="flex flex-col items-center justify-center p-8 text-center space-y-4">
            <div className="relative">
              <div className="w-16 h-16 rounded-full border-4 border-indigo-500/20 border-t-indigo-500 animate-spin" />
              <Film className="w-6 h-6 text-indigo-400 absolute inset-0 m-auto" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-white mb-1">
                {activeJob?.status === 'queued' ? 'Job Queued on A100...' : 'Generating Video Diffusion Frames...'}
              </h4>
              <p className="text-xs text-gray-400 max-w-md line-clamp-2">
                {activeJob?.prompt}
              </p>
            </div>
            {activeJob?.logs && activeJob.logs.length > 0 && (
              <div className="text-[11px] font-mono text-indigo-300 bg-gray-900/90 px-3 py-1.5 rounded-lg border border-gray-800 max-w-lg truncate">
                {activeJob.logs[activeJob.logs.length - 1]}
              </div>
            )}
          </div>
        ) : currentVideo ? (
          <>
            <video
              ref={videoRef}
              src={currentVideo.url}
              loop={isLoop}
              autoPlay
              playsInline
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              className="w-full h-full object-contain"
            />

            {/* Overlay Play Button on Hover */}
            <div
              onClick={togglePlay}
              className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 flex items-center justify-center transition cursor-pointer"
            >
              <div className="w-14 h-14 rounded-full bg-black/60 backdrop-blur border border-white/20 flex items-center justify-center text-white shadow-xl hover:scale-110 transition">
                {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6 ml-1" />}
              </div>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center text-gray-600 p-8 text-center space-y-2">
            <Film className="w-12 h-12 opacity-40 mb-1" />
            <p className="text-sm font-medium text-gray-400">No Video Selected</p>
            <p className="text-xs text-gray-600">Generate a prompt or select a video from the gallery below.</p>
          </div>
        )}
      </div>

      {/* Control Bar & Metadata */}
      {currentVideo && (
        <div className="p-4 bg-[#0E1424] border-t border-gray-800/80 flex flex-col space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex-1 mr-4">
              <h3 className="text-xs font-bold text-white flex items-center space-x-2 truncate">
                <span className="truncate">{currentVideo.filename}</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  {currentVideo.type || 'T2V'}
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-gray-800 text-gray-300">
                  {currentVideo.size_mb} MB
                </span>
              </h3>
              <p className="text-[11px] text-gray-400 mt-0.5 line-clamp-1 italic">
                "{currentVideo.prompt}"
              </p>
            </div>

            {/* Media Controls */}
            <div className="flex items-center space-x-2">
              <button
                onClick={togglePlay}
                className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-200 transition"
                title={isPlaying ? 'Pause' : 'Play'}
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </button>

              <button
                onClick={handleSpeedChange}
                className="px-2.5 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-xs font-mono text-gray-200 transition"
                title="Playback Speed"
              >
                {playbackRate}x
              </button>

              <button
                onClick={() => setIsLoop(!isLoop)}
                className={`p-2 rounded-lg transition ${
                  isLoop ? 'bg-indigo-600/30 text-indigo-400 border border-indigo-500/40' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
                title="Toggle Loop"
              >
                <RotateCcw className="w-4 h-4" />
              </button>

              <button
                onClick={toggleMute}
                className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-200 transition"
                title={isMuted ? 'Unmute' : 'Mute'}
              >
                {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
              </button>

              <a
                href={currentVideo.url}
                download={currentVideo.filename}
                className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-200 transition"
                title="Download MP4"
              >
                <Download className="w-4 h-4" />
              </a>

              <button
                onClick={handleFullscreen}
                className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-200 transition"
                title="Fullscreen"
              >
                <Maximize className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
