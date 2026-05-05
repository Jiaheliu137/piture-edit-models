# piture-edit-models

Pre-built super-resolution model weights served via jsdelivr CDN for the
browser-based image editor at `Jiaheliu137/piture_edit`. Each subdirectory
holds one model variant; URLs follow `cdn.jsdelivr.net/gh/Jiaheliu137/piture-edit-models@main/<path>`.

## Speed engine — Anime4K-CNN-2x (WebSR)

`anime4k-cnn/`

| File | Size | Network | Layers | Parallel branches |
|---|---|---|---|---|
| `cnn-8.bin`  | 34 KB  | `anime4k/cnn-2x-l`  | 17 | 2 |
| `cnn-16.bin` | 122 KB | `anime4k/cnn-2x-16` | 31 | 4 |
| `cnn-28.bin` | 355 KB | `anime4k/cnn-2x-28` | 52 | 7 |

Binary `A4K\0` weight format reverse-engineered from `free.upscaler.video`'s
production bundle. Architecture from [bloc97/Anime4K](https://github.com/bloc97/Anime4K)
(MIT); the three width variants share a common conv-block backbone, scaling
the number of parallel feature branches to trade VRAM for sharpness.

Loaded via the [`@websr/websr`](https://github.com/sb2702/websr) runtime
(MIT) inside `piture_edit/src/services/websr-extended/` (locally rebundled
to expose the `cnn-2x-{16,28}` networks the public npm release omits).

The auto-tier picker selects:
- `cnn-2x-28` (large) for images ≤ 1920×1080
- `cnn-2x-16` (medium) for images > 1920×1080

## Quality engine — TF.js GraphModels (WebGPU)

`realesrgan/general_fast-192/` — Real-ESRGAN-general-x4v3
- Source: [`xinntao/Real-ESRGAN`](https://github.com/xinntao/Real-ESRGAN) (Apache-2.0)
- Scale: 4× per inference
- Tile size: 192×192 input → 768×768 output (12 px overlap)
- Use case: real-world photo upscaling

`realcugan/2x-no-denoise-64/` — Real-CUGAN 2x no-denoise
- Source: [`bilibili/ailab/Real-CUGAN`](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) (MIT)
- Scale: 2× per inference
- Tile size: 64×64 input → 128×128 output (4 px overlap)
- Use case: anime / illustration upscaling (non-photo content)

Format: TensorFlow.js GraphModel + FP16 weights. Conversion pipeline
PyTorch → ONNX → TF SavedModel → TF.js (via tfjs-converter), originally
from [`xororz/web-realesrgan`](https://github.com/xororz/web-realesrgan)
(GPL-2.0). Only the converted weights are republished here.

## Browser usage

The editor wraps these URLs with a Cache API persistence layer so each
returning user does zero network requests after the first load. Direct
access:

```ts
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgpu'

// Quality engine
const model = await tf.loadGraphModel(
  'https://raw.githubusercontent.com/Jiaheliu137/piture-edit-models/main/realesrgan/general_fast-192/model.json'
)

// Speed engine
const buf = await fetch(
  'https://cdn.jsdelivr.net/gh/Jiaheliu137/piture-edit-models@main/anime4k-cnn/cnn-16.bin'
).then(r => r.arrayBuffer())
```

## Licenses

Each model carries its upstream license:
- Anime4K (network architecture) — MIT
- Real-ESRGAN — Apache-2.0
- Real-CUGAN — MIT
- WebSR runtime (used to load Anime4K weights) — MIT
