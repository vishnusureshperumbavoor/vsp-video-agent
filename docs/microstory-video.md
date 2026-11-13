# Micro-story videos (Level 1)

Agent-agnostic runbook. Any agent or terminal can follow it; nothing here is
tied to OpenCode.

Level 1 turns a short storyboard and one character reference into a **9:16 MP4
with narration, an ambient bed and burned-in captions**. It uses the GPU only
for the stills; TTS and editing run locally on CPU.

```
story.json + face.png
        │
        ├── FLUX.2 /edit ──> one still per scene
        ├── Piper TTS ─────> one WAV per scene
        ├── numpy pad ─────> ambient bed
        └── FFmpeg ────────> stills + audio + SRT -> MP4
```

## Prerequisites

```bash
pip install -r requirements-story.txt
ffmpeg -version          # needs libass, zoompan
```

- The image service must be reachable: `IMAGE_API_URL` and `IMAGE_API_TOKEN` in
  `.env`, and the A100 running.
- `story_video.py` reads the same `.env` as the rest of the repo.
- The first run downloads the Piper voice into `data/voices` (git-ignored).

## Story format

```json
{
  "title": "The Second Cup",
  "scenes": [
    {
      "prompt": "close-up portrait in a sunlit cafe, holding a coffee cup",
      "narration": "Gabriela always ordered one coffee."
    },
    {
      "prompt": "she looks toward a woman sketching by the window",
      "narration": "On Tuesday, a stranger sent a second cup across the room."
    }
  ]
}
```

- 1–8 scenes.
- `prompt` is the scene only; the renderer prepends the identity lock.
- `narration` is spoken verbatim and used for the caption.
- Keep each scene short: 8–20 words works well.

## Commands

```bash
# Validate and estimate; no GPU, TTS or FFmpeg
python scripts/story_video.py plan --script story.json

# Same, before spending anything
python scripts/story_video.py render \
  --script story.json --reference face.png --dry-run

# Render
python scripts/story_video.py render \
  --script story.json \
  --reference face.png \
  --out public/videos/story.mp4
```

Options: `--voice en_US-amy-medium`, `--width 768 --height 1344`, `--steps 4`,
`--guidance 1.0`, `--seed 42`, `--no-music`, `--no-subtitles`.

Outputs:

- `public/videos/story_*.mp4`
- `public/videos/story_*_poster.jpg`
- `public/stories/story_*/captions.srt`
- per-scene stills, WAVs and segments inside the work directory

## What each step does

1. **Stills** — `POST /edit` on the GPU image service with the reference image
   and an identity-locked scene prompt. Default 768×1344 (9:16).
2. **Narration** — Piper (`rhasspy/piper-voices`). One fixed voice across all
   scenes. No GPU.
3. **Ambient bed** — a quiet numpy-generated chord pad, ducked under the voice.
   This is a placeholder for a music model later.
4. **Captions** — SRT built from the actual WAV durations.
5. **Assembly** — FFmpeg: Ken Burns stills → concat → mix narration + pad →
   burn captions → H.264/AAC with `+faststart`.

## Level 2 and 3 (not implemented)

| Upgrade | Model | Trade-off |
| :--- | :--- | :--- |
| Cinematic B-roll | Wan 2.1 I2V 14B | minutes per scene, needs offload |
| Real music | MusicGen / Stable Audio Open / ACE-Step | check license |
| SFX | AudioGen / MOSS-SoundEffect | short clips |
| Talking character | LatentSync / MuseTalk / Wav2Lip | extra setup, lip-sync artifacts |

Level 2 would replace each still with a short I2V clip while keeping the same
narration/SRT/mix stages.

## Licensing

- Piper voices have per-voice licenses; check `rhasspy/piper-voices`.
- MusicGen weights are non-commercial; do not use them for a commercial
  Instagram account without checking the model card.
- Do not clone a real person's voice or likeness without consent.

## QA before publishing

```bash
ffprobe -v error -show_entries format=duration:stream=codec_name,width,height \
  -of default=nw=1 public/videos/story.mp4
```

- 9:16 (1080×1920), H.264 + AAC, duration matches the narration.
- Voice is audible over the pad; no clipping.
- Captions are in frame and timed to the voice.
- No text or watermark artifacts in the stills.
- The face is consistent across scenes; if not, improve the reference or curate
  the stills before assembling.

## Troubleshooting

- `piper is not installed` → `pip install -r requirements-story.txt`.
- Piper voice download fails → check network/HF access; the voice cache is
  `data/voices`.
- `subtitles` filter missing → install an FFmpeg build with libass, or pass
  `--no-subtitles`.
- Image service offline → resume the A100 and verify `IMAGE_API_URL/health`.
- The renderer is intentionally not run in CI; always start with `--dry-run`.
