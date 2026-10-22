import React from 'react';
import { Film, Zap, RefreshCw, Cpu, CheckCircle2, AlertCircle, PlayCircle } from 'lucide-react';
import { ColabStatus } from '../services/api';

interface HeaderProps {
  status: ColabStatus | null;
  onRefresh: () => void;
  onOpenConnect: () => void;
  onSetupColab: () => void;
  isSettingUp: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  status,
  onRefresh,
  onOpenConnect,
  onSetupColab,
  isSettingUp
}) => {
  const isConnected = status?.connected || status?.status === 'connected';

  return (
    <header className="h-16 border-b border-gray-800 bg-[#0E1322] px-6 flex items-center justify-between sticky top-0 z-40">
      <div className="flex items-center space-x-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
          <Film className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="font-bold text-lg text-white tracking-tight">Vibe Video Studio</h1>
            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              Wan 2.1
            </span>
          </div>
          <p className="text-xs text-gray-400">A100 GPU Accelerated AI Filmmaking Studio</p>
        </div>
      </div>

      <div className="flex items-center space-x-3">
        {/* GPU Status Pill */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-gray-900/90 border border-gray-800 text-xs">
          <Cpu className="w-4 h-4 text-indigo-400" />
          <span className="text-gray-300 font-mono">
            {status?.gpu_info ? status.gpu_info : isConnected ? 'A100 Connected' : 'GPU Standby'}
          </span>
          <div className="flex items-center space-x-1 pl-1">
            <span
              className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
              }`}
            />
            <span className="text-gray-400 text-[11px] capitalize">
              {status?.status || 'Disconnected'}
            </span>
          </div>
        </div>

        {/* Action Buttons */}
        <button
          onClick={onRefresh}
          className="p-2 rounded-lg bg-gray-900 border border-gray-800 text-gray-400 hover:text-white hover:bg-gray-800 transition"
          title="Refresh Status & Gallery"
        >
          <RefreshCw className="w-4 h-4" />
        </button>

        <button
          onClick={onSetupColab}
          disabled={!isConnected || isSettingUp}
          className={`px-3 py-1.5 rounded-lg border text-xs font-medium flex items-center space-x-1.5 transition ${
            isConnected
              ? 'bg-purple-950/40 border-purple-800/60 text-purple-300 hover:bg-purple-900/40'
              : 'opacity-50 cursor-not-allowed bg-gray-900 border-gray-800 text-gray-500'
          }`}
          title="Install Diffusers & Wan 2.1 dependencies on Colab"
        >
          <Zap className="w-3.5 h-3.5 text-purple-400" />
          <span>{isSettingUp ? 'Setting up...' : 'Setup A100'}</span>
        </button>

        <button
          onClick={onOpenConnect}
          className="px-4 py-1.5 rounded-lg bg-gradient-to-r from-indigo-600 to-indigo-500 text-white text-xs font-semibold hover:from-indigo-500 hover:to-indigo-400 shadow-md shadow-indigo-600/20 flex items-center space-x-1.5 transition"
        >
          <CheckCircle2 className="w-3.5 h-3.5" />
          <span>{isConnected ? 'Colab Connected' : 'Connect Colab'}</span>
        </button>
      </div>
    </header>
  );
};
