# Making output look photographed, not generated

Both pipelines default to a polished look: smooth skin, even light, a face that
is symmetric to the point of being wrong. That is not a bug in the weights; it
is what the training data and the default prompt vocabulary reward.

Rules below are measured, not guessed.

## The four levers

| Lever | Still (FLUX) | Video (H3) |
| :--- | :--- | :--- |
| Steps | **8** (4 is plastic) | **20** for dialogue, 8 turbo for motion |
| Adapter scale | **0.7** (1.0 over-cooks) | turbo 1.0, or 0 with dialogue |
| Prompt vocabulary | raw-photo cues | 35mm film cues |
| Training data | sheet rendered at the *same* steps | — |

## Still images (FLUX.2 + identity LoRA)

Use `scripts/realism.py` for the suffix and the audit:

```python
from realism import enhance, audit

prompt = enhance("Gabriela at a rooftop bar at sunset, black dress", "still")
for warning in audit(prompt):
    print(warning)
```

The suffix that worked:

```text
raw candid photograph, 85mm lens, f/1.8, natural skin texture, visible pores,
subtle freckles, film grain, soft directional window light, no retouching
```

Words that **hurt** (the audit flags them): `instagram`, `airbrushed`,
`smooth skin`, `perfect skin`, `beauty filter`, `hyperrealistic`, `8k`,
`ultra hd`, `glossy`.

Why 8 steps: the LoRA was trained from a sheet generated at 4 steps, so the
softness is baked in. Inference at 8 recovers most of it; 12 adds texture but
drifts the face.

## Video with audio (MiniMax H3)

For speaking shots use dialogue tags — H3 will not invent words from prose:

```text
A medium close-up of a woman at a kitchen table. She looks into the camera and
speaks: <d>[English] Hey, it's me. I just got home. Call me back, okay?</d>
```

The CLI does this for you and switches the defaults:

```bash
.venv/bin/python scripts/h3_video.py "a woman at a kitchen table ..." \
  --dialogue "Hey, it's me. I just got home. Call me back, okay?" \
  --seconds 5 --size 448x768 --raw --host <ip> --machine-id <id>
```

| Setting | Value | Why |
| :--- | :--- | :--- |
| `--steps` | 20 | 8-step turbo blurs speech |
| `--lora-strength` | 0 | turbo adapter is motion-tuned |
| `--size` | 448x768 | portrait reads as a phone video |
| `--seconds` | 5 | the line must fit; H3 cuts audio at the end |
| `--raw` | on | adds film grain and practical-light cues |

The 35mm suffix:

```text
shot on 35mm film, shallow depth of field, natural skin texture, practical
lighting, subtle grain, handheld micro-movement, no digital smoothing
```

## What still breaks

| Artifact | Where | Mitigation |
| :--- | :--- | :--- |
| Invented text | books, signs, labels | avoid text in frame, or blur it |
| Extra fingers | hands near face | crop tighter, or hide hands |
| Teeth | wide smiles | neutral expression, closed mouth |
| Background faces | crowds | shallow depth of field, empty scene |
| Lip-sync drift | plosives (p/b/m) | shorter line, slower delivery |
| Clipped speech | last 0.3 s | request 1 s more than the line needs |

## Verify before you send

1. **Probe the container**

   ```bash
   ffprobe -v error -show_entries stream=codec_type,codec_name,channels,sample_rate \
     -of default=noprint_wrappers=1 out.mp4
   ```

2. **Check it is not silent** — `mean_volume` near −91 dB means an empty track.

3. **Transcribe speech.** For dialogue this is the only real proof:

   ```python
   from faster_whisper import WhisperModel
   model = WhisperModel("base.en", device="cpu", compute_type="int8")
   segments, _ = model.transcribe(audio_float32)
   print(" ".join(s.text for s in segments))
   ```

   A 5 s clip transcribes in a second; the words should match the script.

4. **Look at a frame**, not just the thumbnail. Extract at 1–2 s and check
   hands, teeth, ears, and any text.

## Model matrix

| Model | Realism strength | Audio | Notes |
| :--- | :--- | :--- | :--- |
| FLUX.2 klein 4B + identity LoRA | good at 8 steps / 0.7 | — | identity locked, polished |
| MiniMax H3 | very good at 20 steps | native stereo + dialogue | 5–15 s, open weights |
| MOSS-Image + NSFW adapter | good at 50 steps | — | different family, no identity |
| Wan 2.1 | motion-first | silent | needs a separate audio pass |

## The one-line rule

If the prompt describes a *vibe* ("beautiful", "perfect", "Instagram"), you get
a render. If it describes a *camera* ("85mm, f/1.8, window light, grain"), you
get a photograph.
