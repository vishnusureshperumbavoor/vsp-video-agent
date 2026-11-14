# Character LoRA: training, inference and limits

A single-subject identity adapter for the Telegram image app. The character is
fictional; use only images you have the right to use, 18+ subjects only.

- Trainer launcher: `scripts/character/train.py` via `scripts/character_cli.py`
- GPU service: `scripts/image_server.py`
- Backend switch: `IMAGE_USE_CHARACTER=1`
- Caption helper: `scripts/job_text.py`

## Current stack

| Layer | Value |
| :--- | :--- |
| Base | `black-forest-labs/FLUX.2-klein-4B` (Apache-2.0, distilled, 4B) |
| Adapter | `gbrla` LoRA, 32 MB, `pytorch_lora_weights.safetensors` |
| Inference | 8 steps, guidance 1.0, LoRA scale **0.7**, 1024×1024 |
| Trigger | `gbrla` (prepended on the GPU side) |
| Prompt | active character's appearance (backend) + trigger (GPU) + scene |

Training run that produced it: 28 curated images, **600 steps, 38m 26s**, final
loss **0.6**, checkpoints at 200/400/600 under
`/home/jl_fs/character/lora/checkpoint-*/`.

## Settings that matter

1. **Train on the images you want at inference resolution.** The sheet was
   rendered at 4 steps, so softness is baked into the weights; 8-step inference
   only partially corrects it.
2. **8 steps beats 4** by a wide margin. 4 steps looks airbrushed and plastic.
3. **Scale 0.7 beats 1.0.** Full strength over-cooks skin and edges.
4. **Raw-photo prompts beat "Instagram lifestyle".** Phrases that help:
   `raw candid photograph, 85mm lens, f/1.8, natural skin texture, visible
   pores, film grain, no retouching`. Phrases that hurt: `Instagram`, `beauty`,
   `smooth`, `airbrushed`.
5. **12 steps** adds texture but drifts the face. Not worth it for identity.

Measured comparison (cafe scene and close-up, raw and Instagram prompts):

| Config | Verdict |
| :--- | :--- |
| 4 steps, scale 1.0 | soft, plastic — the original problem |
| 8 steps, scale 0.7 | good balance |
| 12 steps, scale 0.6 | most texture, face drift |
| checkpoint-400, 8 / 0.7 | best overall |

The fix for more realism is **not** more inference steps: regenerate the
training sheet at 8 steps with raw-photo prompts, curate, retrain 400–600 steps.

## Serving path

`POST /character` on the GPU service loads the FLUX pipeline once, keeps the
adapter attached under the name `character`, enables it for this endpoint and
disables it for `/edit`. The backend selects it with `IMAGE_USE_CHARACTER=1`.

Why the split:

- `/imagine` → `/character`, LoRA **on**, identity preserved.
- `/edit` → FLUX reference edit, LoRA **off**, so an uploaded photo drives the
  result instead of the character.

Telegram captions use `scripts/job_text.py`: character jobs show only the
character name plus render time. The composed appearance prompt is never echoed
back to the user.

## Uncensored work

The FLUX LoRA is **not** an uncensored adapter, and reusing it on another base
is impossible: LoRAs bind to named layers, and MOSS/FLUX layer names differ.

| Goal | Route |
| :--- | :--- |
| Less prompt resistance, same face | abliterated FLUX.2 text encoder (`ponpoke/flux2-klein-4b-uncensored-text-encoder`) |
| Fully uncensored, no fixed identity | MOSS-Image + `axiomofmind/Moss-image-nsfw-uncensored-lora` |
| Fully uncensored **and** the same face | train a second identity adapter on the MOSS transformer, then stack both adapters |

The MOSS adapter ships a merged model under `merged-model/`
(~25.4 GiB, fixed scale 0.8). Prefer the 70 MB adapter unless you specifically
want the merged checkpoint for ComfyUI or a PEFT-free server.

## Known limits

- Every `/imagine` user currently gets the same LoRA; owner-only routing is not
  implemented.
- The daily image cap counts failed jobs. Raise `DAILY_IMAGE_LIMIT` or exclude
  failures at the service layer.
- The A100 instance is paused between sessions; resume changes the machine ID.
  Re-register it and restart the backend when it does.
