import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { T2VPanel } from './components/T2VPanel';
import { I2VPanel } from './components/I2VPanel';
import { StoryboardPanel } from './components/StoryboardPanel';
import { VideoPlayer } from './components/VideoPlayer';
import { VideoGallery } from './components/VideoGallery';
import { ColabConnectionModal } from './components/ColabConnectionModal';
import {
  ColabStatus,
  VideoMetadata,
  VideoJob,
  fetchColabStatus,
  fetchGallery,
  fetchJobs,
  generateT2V,
  generateI2V,
  stitchVideos,
  setupColabEnvironment,
  T2VParams,
} from './services/api';
import { Sparkles, Wand2, Film, Image as ImageIcon, Terminal, PlaySquare } from 'lucide-react';

export function App() {
  const [colabStatus, setColabStatus] = useState<ColabStatus | null>(null);
  const [gallery, setGallery] = useState<VideoMetadata[]>([]);
  const [jobs, setJobs] = useState<VideoJob[]>([]);
  const [activeTab, setActiveTab] = useState<'t2v' | 'i2v' | 'storyboard'>('t2v');
  const [selectedVideo, setSelectedVideo] = useState<VideoMetadata | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isSettingUp, setIsSettingUp] = useState(false);
  const [isStitching, setIsStitching] = useState(false);

  const loadData = async () => {
    try {
      const [cStatus, gal, jbs] = await Promise.all([
        fetchColabStatus().catch(() => null),
        fetchGallery().catch(() => []),
        fetchJobs().catch(() => []),
      ]);
      if (cStatus) setColabStatus(cStatus);
      if (gal) {
        setGallery(gal);
        if (!selectedVideo && gal.length > 0) {
          setSelectedVideo(gal[0]);
        }
      }
      if (jbs) setJobs(jbs);
    } catch (e) {
      console.error('Data load error:', e);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 4000);
    return () => clearInterval(interval);
  }, []);

  const activeJob = jobs.find((j) => j.status === 'running' || j.status === 'queued') || null;
  const isGenerating = Boolean(activeJob);
  const isConnected = colabStatus?.connected || colabStatus?.status === 'connected';

  const handleGenerateT2V = async (params: T2VParams) => {
    try {
      await generateT2V(params);
      loadData();
    } catch (err: any) {
      alert(`Generation failed to queue: ${err.message}`);
    }
  };

  const handleGenerateI2V = async (formData: FormData) => {
    try {
      await generateI2V(formData);
      loadData();
    } catch (err: any) {
      alert(`I2V failed to queue: ${err.message}`);
    }
  };

  const handleStitch = async (filenames: string[]) => {
    setIsStitching(true);
    try {
      const res = await stitchVideos(filenames);
      alert(`Movie stitched successfully: ${res.filename} (${res.size_mb} MB)`);
      loadData();
    } catch (err: any) {
      alert(`Stitching failed: ${err.message}`);
    } finally {
      setIsStitching(false);
    }
  };

  const handleSetupColab = async () => {
    setIsSettingUp(true);
    try {
      await setupColabEnvironment();
      alert('Colab environment setup job queued! Check logs below.');
      loadData();
    } catch (err: any) {
      alert(`Setup dispatch failed: ${err.message}`);
    } finally {
      setIsSettingUp(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0F19] text-gray-100 flex flex-col">
      {/* Top Navigation */}
      <Header
        status={colabStatus}
        onRefresh={loadData}
        onOpenConnect={() => setIsModalOpen(true)}
        onSetupColab={handleSetupColab}
        isSettingUp={isSettingUp}
      />

      {/* Main Studio Workspace */}
      <main className="flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 max-w-7xl mx-auto w-full">
        {/* Left Column: Creator Studio (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Mode Switcher Tabs */}
          <div className="bg-[#111827] p-1.5 rounded-2xl border border-gray-800 flex space-x-1 shadow-md">
            <button
              onClick={() => setActiveTab('t2v')}
              className={`flex-1 py-2.5 rounded-xl text-xs font-semibold flex items-center justify-center space-x-1.5 transition ${
                activeTab === 't2v'
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/40'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Text to Video</span>
            </button>

            <button
              onClick={() => setActiveTab('i2v')}
              className={`flex-1 py-2.5 rounded-xl text-xs font-semibold flex items-center justify-center space-x-1.5 transition ${
                activeTab === 'i2v'
                  ? 'bg-purple-600 text-white shadow-md shadow-purple-600/30'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/40'
              }`}
            >
              <ImageIcon className="w-3.5 h-3.5" />
              <span>Image to Video</span>
            </button>

            <button
              onClick={() => setActiveTab('storyboard')}
              className={`flex-1 py-2.5 rounded-xl text-xs font-semibold flex items-center justify-center space-x-1.5 transition ${
                activeTab === 'storyboard'
                  ? 'bg-emerald-600 text-white shadow-md shadow-emerald-600/30'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/40'
              }`}
            >
              <Film className="w-3.5 h-3.5" />
              <span>Storyboard</span>
            </button>
          </div>

          {/* Active Studio Box */}
          <div className="p-5 rounded-2xl bg-[#111827] border border-gray-800 shadow-xl">
            {activeTab === 't2v' && (
              <T2VPanel
                onGenerate={handleGenerateT2V}
                isGenerating={isGenerating}
                isConnected={Boolean(isConnected)}
              />
            )}
            {activeTab === 'i2v' && (
              <I2VPanel
                onGenerate={handleGenerateI2V}
                isGenerating={isGenerating}
                isConnected={Boolean(isConnected)}
              />
            )}
            {activeTab === 'storyboard' && (
              <StoryboardPanel
                gallery={gallery}
                onStitch={handleStitch}
                isStitching={isStitching}
              />
            )}
          </div>

          {/* Live Colab Log Terminal */}
          <div className="p-4 rounded-2xl bg-[#0E1424] border border-gray-800/90 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-gray-300 flex items-center space-x-1.5">
                <Terminal className="w-3.5 h-3.5 text-indigo-400" />
                <span>Colab A100 Live Logs</span>
              </span>
              <span className="text-[10px] text-gray-500 font-mono">
                {colabStatus?.recent_logs?.length || 0} events
              </span>
            </div>
            <div className="bg-[#080B14] p-3 rounded-xl border border-gray-800/60 font-mono text-[11px] text-gray-400 h-28 overflow-y-auto space-y-1">
              {colabStatus?.recent_logs && colabStatus.recent_logs.length > 0 ? (
                colabStatus.recent_logs.map((log, i) => (
                  <p key={i} className="text-gray-300 truncate">
                    {log}
                  </p>
                ))
              ) : (
                <p className="text-gray-600 italic">No recent Colab activity. Connect and dispatch a job.</p>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Player & Video Library (7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Active Cinema Player */}
          <VideoPlayer currentVideo={selectedVideo} activeJob={activeJob} />

          {/* Local Video Library Grid */}
          <VideoGallery
            videos={gallery}
            selectedVideo={selectedVideo}
            onSelectVideo={(v) => setSelectedVideo(v)}
          />
        </div>
      </main>

      {/* Connect Colab Modal */}
      <ColabConnectionModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        status={colabStatus}
        onRefresh={loadData}
      />
    </div>
  );
}

export default App;
