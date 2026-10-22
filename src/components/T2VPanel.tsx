import React, { useState } from 'react';
import { Sparkles, Wand2, Sliders, Play, Film } from 'lucide-react';
import { T2VParams } from '../services/api';

interface T2VPanelProps {
  onGenerate: (params: T2VParams) => void;
  isGenerating: boolean;
  isConnected: boolean;
}

const PRESET_PROMPTS = [
  "A futuristic cyberpunk supercar driving down a rainy Tokyo neon street at night, cinematic reflections, ultra-realistic, 4k",
  "Close-up cinematic portrait of a young female explorer with hazel eyes, gentle warm lighting, slight breeze moving her hair, 8k film still",
  "FPV drone flying through an ancient mossy rainforest canyon with glowing mystical waterfalls, golden hour volumetric sunlight",
  "Astronaut floating outside a massive orbital ring station overlooking Earth at sunrise, epic sci-fi realism, interstellar"
];

export const T2VPanel: React.FC<T2VPanelProps> = ({ onGenerate, isGenerating, isConnected }) => {
  const [prompt, setPrompt] = useState(PRESET_PROMPTS[0]);
  const [negativePrompt, setNegativePrompt] = useState("low quality, blurry, distorted faces, flickering, artifacts, bad anatomy");
  const [aspectRatio, setAspectRatio] = useState("16:9");
  const [duration, setDuration] = useState(5);
  const [steps, setSteps] = useState(30);
  const [modelSize, setModelSize] = useState("1.3B");
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    onGenerate({
      prompt,
      negative_prompt: negativePrompt,
      aspect_ratio: aspectRatio,
      duration,
      steps,
      model_size: modelSize,
      seed: Math.floor(Math.random() * 100000)
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Prompt Field */}
      <div>
        <label className="block text-xs font-semibold text-gray-300 mb-1.5 flex items-center justify-between">
          <span className="flex items-center space-x-1.5">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            <span>Prompt (Describe Scene, Motion & Camera)</span>
          </span>
          <span className="text-[11px] text-gray-500">{prompt.length} chars</span>
        </label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={4}
          placeholder="Describe your scene in detail: lighting, camera motion, character actions..."
          className="w-full bg-[#131B2E] border border-gray-800 rounded-xl p-3 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition resize-none"
        />
      </div>

      {/* Preset Chips */}
      <div>
        <span className="text-[11px] font-medium text-gray-400 mb-1.5 block">Quick Cinematic Inspirations:</span>
        <div className="flex flex-wrap gap-1.5">
          {PRESET_PROMPTS.map((p, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => setPrompt(p)}
              className="text-[11px] px-2.5 py-1 rounded-lg bg-gray-900/80 border border-gray-800 text-gray-400 hover:text-indigo-300 hover:border-indigo-500/30 transition truncate max-w-[280px]"
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {/* Settings Grid */}
      <div className="grid grid-cols-3 gap-3 pt-2">
        {/* Aspect Ratio */}
        <div>
          <label className="block text-[11px] font-semibold text-gray-400 mb-1">Aspect Ratio</label>
          <div className="grid grid-cols-3 gap-1 bg-[#131B2E] p-1 rounded-lg border border-gray-800">
            {["16:9", "9:16", "1:1"].map((ratio) => (
              <button
                key={ratio}
                type="button"
                onClick={() => setAspectRatio(ratio)}
                className={`py-1 text-xs rounded font-medium transition ${
                  aspectRatio === ratio
                    ? "bg-indigo-600 text-white shadow"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                {ratio}
              </button>
            ))}
          </div>
        </div>

        {/* Duration */}
        <div>
          <label className="block text-[11px] font-semibold text-gray-400 mb-1">Duration</label>
          <div className="grid grid-cols-2 gap-1 bg-[#131B2E] p-1 rounded-lg border border-gray-800">
            {[5, 10].map((sec) => (
              <button
                key={sec}
                type="button"
                onClick={() => setDuration(sec)}
                className={`py-1 text-xs rounded font-medium transition ${
                  duration === sec
                    ? "bg-indigo-600 text-white shadow"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                {sec}s
              </button>
            ))}
          </div>
        </div>

        {/* Model Scale */}
        <div>
          <label className="block text-[11px] font-semibold text-gray-400 mb-1">Wan 2.1 Model</label>
          <div className="grid grid-cols-2 gap-1 bg-[#131B2E] p-1 rounded-lg border border-gray-800">
            {[
              { id: "1.3B", label: "1.3B Fast" },
              { id: "14B", label: "14B Pro" }
            ].map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setModelSize(m.id)}
                className={`py-1 text-xs rounded font-medium transition ${
                  modelSize === m.id
                    ? "bg-indigo-600 text-white shadow"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Advanced Toggle */}
      <div>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-xs text-gray-400 hover:text-gray-200 flex items-center space-x-1"
        >
          <Sliders className="w-3.5 h-3.5" />
          <span>{showAdvanced ? "Hide Advanced Settings" : "Show Advanced Settings (Negative Prompt, Steps)"}</span>
        </button>

        {showAdvanced && (
          <div className="mt-3 p-3 bg-[#131B2E] rounded-xl border border-gray-800 space-y-3">
            <div>
              <label className="block text-[11px] font-medium text-gray-400 mb-1">Negative Prompt</label>
              <input
                type="text"
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                className="w-full bg-[#0B0F19] border border-gray-800 rounded-lg px-2.5 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-indigo-500"
              />
            </div>
            <div>
              <label className="block text-[11px] font-medium text-gray-400 mb-1">
                Inference Steps: {steps}
              </label>
              <input
                type="range"
                min={20}
                max={50}
                step={5}
                value={steps}
                onChange={(e) => setSteps(Number(e.target.value))}
                className="w-full accent-indigo-500"
              />
            </div>
          </div>
        )}
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={isGenerating || !isConnected}
        className={`w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center space-x-2 transition shadow-lg ${
          isGenerating || !isConnected
            ? "bg-gray-800 text-gray-500 cursor-not-allowed border border-gray-700"
            : "bg-gradient-to-r from-indigo-600 via-indigo-500 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white shadow-indigo-500/25 active:scale-[0.99]"
        }`}
      >
        <Wand2 className={`w-4 h-4 ${isGenerating ? 'animate-spin' : ''}`} />
        <span>
          {!isConnected
            ? "Connect Colab First to Generate"
            : isGenerating
            ? "Generating on A100 GPU..."
            : "Generate Video on A100"}
        </span>
      </button>
    </form>
  );
};
