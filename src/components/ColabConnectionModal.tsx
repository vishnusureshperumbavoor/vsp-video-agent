import React, { useState, useEffect } from 'react';
import { X, Cpu, CheckCircle2, AlertCircle, RefreshCw, Terminal, ExternalLink } from 'lucide-react';
import { ColabStatus, connectColab, probeGpu } from '../services/api';

interface ColabConnectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  status: ColabStatus | null;
  onRefresh: () => void;
}

export const ColabConnectionModal: React.FC<ColabConnectionModalProps> = ({
  isOpen,
  onClose,
  status,
  onRefresh,
}) => {
  const [profiles, setProfiles] = useState<string[]>(['Default', 'Profile 1', 'Profile 2']);
  const [selectedProfile, setSelectedProfile] = useState('Default');
  const [isConnecting, setIsConnecting] = useState(false);
  const [isProbing, setIsProbing] = useState(false);
  const [connectResult, setConnectResult] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      fetch('/colab/accounts')
        .then((res) => res.json())
        .then((data) => {
          if (Array.isArray(data) && data.length > 0) {
            setProfiles(data.map((acc: any) => acc.profile_directory || acc.name || acc));
          }
        })
        .catch(() => {});
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleConnect = async () => {
    setIsConnecting(true);
    setConnectResult(null);
    try {
      const res = await connectColab(selectedProfile);
      setConnectResult(res.status || 'Connected successfully!');
      onRefresh();
    } catch (err: any) {
      setConnectResult(`Connection failed: ${err.message}`);
    } finally {
      setIsConnecting(false);
    }
  };

  const handleProbeGpu = async () => {
    setIsProbing(true);
    setConnectResult('Probing Colab hardware (keeping cell in notebook)...');
    try {
      const res = await probeGpu(true);
      setConnectResult(`Hardware Verified: ${res.gpu_info || 'Connected'}`);
      onRefresh();
    } catch (err: any) {
      setConnectResult(`Hardware probe failed: ${err.message}`);
    } finally {
      setIsProbing(false);
    }
  };

  const isConnected = status?.connected || status?.status === 'connected';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="w-full max-w-lg bg-[#0F172A] border border-gray-800 rounded-2xl overflow-hidden shadow-2xl flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-gray-800 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white">Google Colab A100 Bridge</h3>
              <p className="text-[11px] text-gray-400">vspagent MCP Execution Harness</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-gray-400 hover:text-white transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="p-5 space-y-4 text-xs text-gray-300">
          {/* Status Box */}
          <div
            className={`p-3.5 rounded-xl border flex items-start space-x-3 ${
              isConnected
                ? 'bg-emerald-950/20 border-emerald-800/40 text-emerald-300'
                : 'bg-amber-950/20 border-amber-800/40 text-amber-300'
            }`}
          >
            {isConnected ? (
              <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0 text-emerald-400" />
            ) : (
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0 text-amber-400" />
            )}
            <div>
              <p className="font-semibold">{isConnected ? 'Colab A100 Connected' : 'Colab Runtime Disconnected'}</p>
              <p className="text-[11px] opacity-80 mt-0.5">
                {status?.message || 'Select your Chrome profile and click Connect to bridge to Colab.'}
              </p>
              {status?.gpu_info && (
                <p className="font-mono text-[10px] text-emerald-400 mt-1">GPU: {status.gpu_info}</p>
              )}
            </div>
          </div>

          {/* Profile Selector */}
          <div>
            <label className="block text-[11px] font-semibold text-gray-400 mb-1.5">
              Select Chrome Profile (where Colab is open)
            </label>
            <select
              value={selectedProfile}
              onChange={(e) => setSelectedProfile(e.target.value)}
              className="w-full bg-[#1E293B] border border-gray-700 rounded-xl p-2.5 text-xs text-white focus:outline-none focus:border-indigo-500"
            >
              {profiles.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>

          {/* Connect Action */}
          <button
            onClick={handleConnect}
            disabled={isConnecting}
            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-semibold flex items-center justify-center space-x-2 transition shadow-lg shadow-indigo-600/20"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isConnecting ? 'animate-spin' : ''}`} />
            <span>{isConnecting ? 'Pairing with Colab...' : 'Connect to Colab Runtime'}</span>
          </button>

          {isConnected && (
            <button
              onClick={handleProbeGpu}
              disabled={isProbing}
              className="w-full py-2 rounded-xl bg-gray-800/90 hover:bg-gray-750 text-indigo-300 hover:text-white font-semibold flex items-center justify-center space-x-2 transition border border-gray-700/80 shadow-md"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isProbing ? 'animate-spin text-indigo-400' : ''}`} />
              <span>{isProbing ? 'Probing Hardware...' : 'Verify GPU Hardware'}</span>
            </button>
          )}

          {connectResult && (
            <div className="p-2.5 bg-gray-900 rounded-lg border border-gray-800 font-mono text-[11px] text-gray-300">
              {connectResult}
            </div>
          )}

          {/* Colab Quick Guide */}
          <div className="border-t border-gray-800/80 pt-3 space-y-1.5 text-[11px] text-gray-400">
            <p className="font-semibold text-gray-300">Quick Reminder:</p>
            <p>1. Open Google Colab in your browser and switch runtime to <strong className="text-indigo-400">NVIDIA A100 GPU</strong>.</p>
            <p>2. Keep the Colab notebook tab active.</p>
            <p>3. Once connected, Vibe Video Studio will automatically dispatch Wan 2.1 inference.</p>
          </div>
        </div>
      </div>
    </div>
  );
};
