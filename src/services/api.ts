export interface ColabStatus {
  status: string;
  message: string;
  gpu_info?: string;
  tools?: string[];
  connected: boolean;
  colabAvailable?: boolean;
  recent_logs?: string[];
}

export interface VideoMetadata {
  filename: string;
  url: string;
  size_mb: number;
  created_at: number;
  prompt: string;
  aspect_ratio?: string;
  duration?: number;
  type?: string;
}

export interface VideoJob {
  id: string;
  type: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  created_at: number;
  prompt?: string;
  logs: string[];
  error?: string;
  video_url?: string;
}

export interface T2VParams {
  prompt: string;
  negative_prompt?: string;
  aspect_ratio?: string;
  duration?: number;
  steps?: number;
  seed?: number;
  model_size?: string;
}

export async function fetchHealth() {
  const res = await fetch('/api/health');
  return res.json();
}

export async function fetchColabStatus(): Promise<ColabStatus> {
  const res = await fetch('/colab/status');
  return res.json();
}

export async function connectColab(profileDirectory?: string) {
  const res = await fetch('/colab/connect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ profileDirectory, force: true }),
  });
  return res.json();
}

export async function setupColabEnvironment() {
  const res = await fetch('/api/video/setup-colab', {
    method: 'POST',
  });
  return res.json();
}

export async function generateT2V(params: T2VParams) {
  const res = await fetch('/api/video/generate/t2v', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function generateI2V(formData: FormData) {
  const res = await fetch('/api/video/generate/i2v', {
    method: 'POST',
    body: formData,
  });
  return res.json();
}

export async function fetchGallery(): Promise<VideoMetadata[]> {
  const res = await fetch('/api/video/gallery');
  return res.json();
}

export async function fetchJobs(): Promise<VideoJob[]> {
  const res = await fetch('/api/video/jobs');
  return res.json();
}

export async function stitchVideos(video_filenames: string[], output_filename?: string) {
  const res = await fetch('/api/video/stitch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_filenames, output_filename }),
  });
  return res.json();
}
