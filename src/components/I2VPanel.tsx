import React, { useState, useRef } from 'react';
import { Upload, Image as ImageIcon, Wand2, X, Sparkles } from 'lucide-react';

interface I2VPanelProps {
  onGenerate: (formData: FormData) => void;
  isGenerating: boolean;
  isConnected: boolean;
}

export const I2VPanel: React.FC<I2VPanelProps> = ({ onGenerate, isGenerating, isConnected }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("The character slowly smiles and turns toward the camera, cinematic lighting, ultra-realistic motion");
  const [aspectRatio, setAspectRatio] = useState("16:9");
  const [duration, setDuration] = useState(5);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const clearFile = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile || !prompt.trim()) return;

    const fd = new FormData();
    fd.append('image', selectedFile);
    fd.append('prompt', prompt);
    fd.append('aspect_ratio', aspectRatio);
    fd.append('duration', String(duration));
    fd.append('steps', '30');
    fd.append('seed', String(Math.floor(Math.random() * 100000)));

    onGenerate(fd);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Image Uploader */}
      <div>
        <label className="block text-xs font-semibold text-gray-300 mb-1.5 flex items-center space-x-1.5">
          <ImageIcon className="w-3.5 h-3.5 text-indigo-400" />
          <span>Reference Photo (Face, Character or Scenery)</span>
        </label>

        {!previewUrl ? (
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-gray-800 hover:border-indigo-500/50 rounded-xl p-6 bg-[#131B2E]/60 flex flex-col items-center justify-center cursor-pointer transition group"
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
              accept="image/*"
              className="hidden"
            />
            <div className="w-12 h-12 rounded-xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center mb-2 group-hover:scale-110 transition">
              <Upload className="w-6 h-6" />
            </div>
            <p className="text-xs font-medium text-gray-300">Click or drop your photo here</p>
            <p className="text-[11px] text-gray-500 mt-1">Supports JPG, PNG, WebP (under 10MB)</p>
          </div>
        ) : (
          <div className="relative rounded-xl overflow-hidden border border-gray-800 bg-gray-900 aspect-video flex items-center justify-center group">
            <img src={previewUrl} alt="Preview" className="w-full h-full object-contain" />
            <button
              type="button"
              onClick={clearFile}
              className="absolute top-2 right-2 p-1.5 rounded-lg bg-black/70 text-gray-300 hover:text-white transition"
              title="Remove image"
            >
              <X className="w-4 h-4" />
            </button>
            <span className="absolute bottom-2 left-2 text-[10px] bg-black/60 px-2 py-0.5 rounded text-gray-300">
              {selectedFile?.name}
            </span>
          </div>
        )}
      </div>

      {/* Motion Prompt */}
      <div>
        <label className="block text-xs font-semibold text-gray-300 mb-1.5 flex items-center justify-between">
          <span className="flex items-center space-x-1.5">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            <span>Motion Prompt (How Should It Animate?)</span>
          </span>
        </label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          placeholder="E.g. Camera slowly zooms in, character blinks and smiles warmly, hair blows in wind..."
          className="w-full bg-[#131B2E] border border-gray-800 rounded-xl p-3 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition resize-none"
        />
      </div>

      {/* Settings Grid */}
      <div className="grid grid-cols-2 gap-3">
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
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={isGenerating || !isConnected || !selectedFile}
        className={`w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center space-x-2 transition shadow-lg ${
          isGenerating || !isConnected || !selectedFile
            ? "bg-gray-800 text-gray-500 cursor-not-allowed border border-gray-700"
            : "bg-gradient-to-r from-purple-600 via-indigo-600 to-indigo-500 hover:from-purple-500 hover:to-indigo-400 text-white shadow-purple-500/25 active:scale-[0.99]"
        }`}
      >
        <Wand2 className={`w-4 h-4 ${isGenerating ? 'animate-spin' : ''}`} />
        <span>
          {!selectedFile
            ? "Upload Photo to Animate"
            : !isConnected
            ? "Connect Colab First"
            : isGenerating
            ? "Animating on A100..."
            : "Animate Photo on A100"}
        </span>
      </button>
    </form>
  );
};
