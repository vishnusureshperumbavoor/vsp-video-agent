import React, { useState } from 'react';
import { Film, Plus, Trash2, Video, Sparkles, Scissors, CheckCircle2 } from 'lucide-react';
import { VideoMetadata } from '../services/api';

interface Scene {
  id: string;
  name: string;
  prompt: string;
  assignedVideo?: string;
}

interface StoryboardPanelProps {
  gallery: VideoMetadata[];
  onStitch: (filenames: string[]) => void;
  isStitching: boolean;
}

export const StoryboardPanel: React.FC<StoryboardPanelProps> = ({
  gallery,
  onStitch,
  isStitching,
}) => {
  const [scenes, setScenes] = useState<Scene[]>([
    {
      id: '1',
      name: 'Scene 1: The Opening Shot',
      prompt: 'Wide cinematic shot of the protagonist walking through the neon-lit city at night, heavy rain reflections',
    },
    {
      id: '2',
      name: 'Scene 2: Close-up Reveal',
      prompt: 'Dramatic close-up of protagonist stopping under streetlamp, looking directly into the camera',
    },
  ]);

  const addScene = () => {
    const num = scenes.length + 1;
    setScenes([
      ...scenes,
      {
        id: String(Date.now()),
        name: `Scene ${num}: Climax / Transition`,
        prompt: 'Dynamic camera motion with cinematic lighting...',
      },
    ]);
  };

  const removeScene = (id: string) => {
    setScenes(scenes.filter((s) => s.id !== id));
  };

  const assignVideoToScene = (sceneId: string, filename: string) => {
    setScenes(
      scenes.map((s) => (s.id === sceneId ? { ...s, assignedVideo: filename } : s))
    );
  };

  const readyToStitch = scenes.filter((s) => s.assignedVideo).length >= 2;

  const handleStitch = () => {
    const filenames = scenes
      .map((s) => s.assignedVideo)
      .filter((fn): fn is string => Boolean(fn));
    onStitch(filenames);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xs font-bold text-gray-200 uppercase tracking-wider flex items-center space-x-1.5">
            <Film className="w-3.5 h-3.5 text-indigo-400" />
            <span>Multi-Shot Movie Timeline</span>
          </h3>
          <p className="text-[11px] text-gray-400">
            Assign generated clips to scenes and stitch them into a continuous film using FFmpeg.
          </p>
        </div>
        <button
          onClick={addScene}
          className="px-2.5 py-1 rounded-lg bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-600/30 text-xs font-medium flex items-center space-x-1 transition"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>Add Scene</span>
        </button>
      </div>

      {/* Scenes List */}
      <div className="space-y-3">
        {scenes.map((scene, idx) => (
          <div
            key={scene.id}
            className="p-3.5 rounded-xl bg-[#131B2E] border border-gray-800 space-y-2 hover:border-gray-700 transition"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-indigo-400 flex items-center space-x-1.5">
                <span className="w-4 h-4 rounded-full bg-indigo-500/20 text-indigo-300 flex items-center justify-center text-[10px]">
                  {idx + 1}
                </span>
                <span>{scene.name}</span>
              </span>
              {scenes.length > 1 && (
                <button
                  onClick={() => removeScene(scene.id)}
                  className="text-gray-500 hover:text-red-400 p-1"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            <p className="text-xs text-gray-300 italic bg-[#0B0F19] p-2 rounded-lg border border-gray-800/80">
              "{scene.prompt}"
            </p>

            {/* Video Assignment */}
            <div className="flex items-center space-x-2 pt-1">
              <Video className="w-3.5 h-3.5 text-gray-500" />
              <select
                value={scene.assignedVideo || ''}
                onChange={(e) => assignVideoToScene(scene.id, e.target.value)}
                className="bg-[#0B0F19] border border-gray-800 text-xs text-gray-300 rounded-lg px-2 py-1.5 focus:outline-none focus:border-indigo-500 flex-1"
              >
                <option value="">-- Choose generated clip for this scene --</option>
                {gallery.map((vid) => (
                  <option key={vid.filename} value={vid.filename}>
                    {vid.filename} ({vid.duration || 5}s - {vid.prompt.substring(0, 35)}...)
                  </option>
                ))}
              </select>

              {scene.assignedVideo && (
                <span className="text-emerald-400 text-xs flex items-center space-x-1">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>Ready</span>
                </span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Stitch Button */}
      <button
        onClick={handleStitch}
        disabled={!readyToStitch || isStitching}
        className={`w-full py-3 rounded-xl font-semibold text-xs flex items-center justify-center space-x-2 transition shadow-lg ${
          !readyToStitch || isStitching
            ? 'bg-gray-800 text-gray-500 cursor-not-allowed border border-gray-700'
            : 'bg-gradient-to-r from-emerald-600 via-teal-600 to-indigo-600 text-white shadow-emerald-600/25 active:scale-[0.99]'
        }`}
      >
        <Scissors className="w-4 h-4" />
        <span>
          {isStitching
            ? 'Stitching Scenes with FFmpeg...'
            : readyToStitch
            ? 'Stitch Movie (Lossless Concat)'
            : 'Assign at least 2 scenes to Stitch'}
        </span>
      </button>
    </div>
  );
};
