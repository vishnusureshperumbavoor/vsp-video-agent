import React from 'react';
import { Film, Play, Download, Clock, HardDrive, Sparkles } from 'lucide-react';
import { VideoMetadata } from '../services/api';

interface VideoGalleryProps {
  videos: VideoMetadata[];
  selectedVideo: VideoMetadata | null;
  onSelectVideo: (video: VideoMetadata) => void;
}

export const VideoGallery: React.FC<VideoGalleryProps> = ({
  videos,
  selectedVideo,
  onSelectVideo,
}) => {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold text-gray-200 uppercase tracking-wider flex items-center space-x-1.5">
          <Film className="w-3.5 h-3.5 text-indigo-400" />
          <span>Local Video Library ({videos.length})</span>
        </h3>
        <span className="text-[11px] text-gray-500">Stored on SSD in public/videos</span>
      </div>

      {videos.length === 0 ? (
        <div className="border border-gray-800 rounded-xl p-8 bg-[#111827]/40 text-center space-y-2">
          <Film className="w-8 h-8 text-gray-600 mx-auto" />
          <p className="text-xs font-medium text-gray-400">No Generated Videos Yet</p>
          <p className="text-[11px] text-gray-500 max-w-sm mx-auto">
            Connect your Colab A100 runtime and submit your first prompt to generate studio clips.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {videos.map((vid) => {
            const isSelected = selectedVideo?.filename === vid.filename;
            return (
              <div
                key={vid.filename}
                onClick={() => onSelectVideo(vid)}
                className={`group relative rounded-xl overflow-hidden border bg-[#111827] cursor-pointer transition shadow-md ${
                  isSelected
                    ? 'border-indigo-500 ring-1 ring-indigo-500'
                    : 'border-gray-800 hover:border-gray-700'
                }`}
              >
                {/* Thumbnail / Video Preview */}
                <div className="aspect-video bg-black/60 relative overflow-hidden flex items-center justify-center">
                  <video
                    src={vid.url}
                    preload="metadata"
                    muted
                    className="w-full h-full object-cover group-hover:scale-105 transition duration-300"
                    onMouseEnter={(e) => (e.currentTarget as HTMLVideoElement).play().catch(() => {})}
                    onMouseLeave={(e) => {
                      const v = e.currentTarget as HTMLVideoElement;
                      v.pause();
                      v.currentTime = 0;
                    }}
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-60 group-hover:opacity-40 transition" />

                  <div className="absolute top-2 left-2 flex items-center space-x-1">
                    <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-black/70 text-indigo-300 border border-indigo-500/20">
                      {vid.type || 'T2V'}
                    </span>
                  </div>

                  <div className="absolute bottom-2 right-2 text-[10px] font-mono bg-black/70 px-1.5 py-0.5 rounded text-gray-300">
                    {vid.duration || 5}s
                  </div>
                </div>

                {/* Card Info */}
                <div className="p-2.5 space-y-1">
                  <p className="text-xs text-gray-200 font-medium line-clamp-1 group-hover:text-indigo-300 transition">
                    {vid.prompt}
                  </p>
                  <div className="flex items-center justify-between text-[10px] text-gray-500">
                    <span className="flex items-center space-x-1">
                      <HardDrive className="w-3 h-3" />
                      <span>{vid.size_mb} MB</span>
                    </span>
                    <span className="truncate max-w-[90px]">{vid.filename}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
