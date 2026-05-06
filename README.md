# piture-edit-models

Browser-side super-resolution model weights served via jsdelivr CDN for the
image editor at [`Jiaheliu137/piture_edit`](https://github.com/Jiaheliu137/piture_edit).
This repo holds the binary artifacts; the editor's runtime, tile-stitching,
and engine selection logic live in the editor repo itself.

The editor user only ever sees two engine choices (**快速** / **质量**) and
two content-type choices for the quality engine (**照片** / **动漫**). All
the technical detail below — model architectures, tile sizes, scale factors,
auto-tier heuristics — is hidden from end-users by design.

---

## Two engines, side by side

| | Speed engine （快速） | Quality engine （质量） |
|---|---|---|
| Architecture | Anime4K-CNN-2x (3 width tiers) | Real-ESRGAN-general / Real-CUGAN |
| Runtime | [WebSR](https://github.com/sb2702/websr) (hand-written WGSL compute shaders) | TensorFlow.js + WebGPU |
| Inference style | **Whole image, single GPU dispatch** | **Tiled** (192×192 with 12 px overlap) |
| 4K → 8K wall-clock | ~350 ms | ~31 s |
| Model size on disk | 34 KB / 122 KB / 355 KB | ~2.4 MB |
| First-load cost | One of three .bin files | Two ~2.4 MB GraphModel weights |
| Output | Strict 2× | Photo 4× (downscaled if requested) / Anime 2× |
| Content-aware | No (single retrained mixed dataset) | Yes (separate photo / anime weights) |
| Best for | Casual upscaling, fast iteration | Heavy-detail photos, anime/illustration with clean line art |

---

## Repo layout

```
anime4k-cnn/
  cnn-8.bin                                 34 KB    speed-small  / cnn-2x-l   (17 layers, 2-way parallel)
  cnn-16.bin                                122 KB   speed-medium / cnn-2x-16  (31 layers, 4-way parallel)
  cnn-28.bin                                355 KB   speed-large  / cnn-2x-28  (52 layers, 7-way parallel)

realesrgan/
  general_fast-192/
    model.json                              66 KB    GraphModel topology
    group1-shard1of1.bin                    2.4 MB   FP16 weights (SRVGGNetCompact, ~1 M params)

realcugan/
  2x-no-denoise-64/                                  ← legacy 64-tile variant, kept for reference
    model.json                              60 KB
    group1-shard1of1.bin                    2.5 MB
  2x-no-denoise-192/                                 ← actively used by the editor
    model.json                              60 KB
    group1-shard1of1.bin                    2.5 MB   same trained weights as 64 tile, 192×192 baked input
```

All weight files are referenced by the editor through jsdelivr's GitHub
CDN mirror — `https://cdn.jsdelivr.net/gh/Jiaheliu137/piture-edit-models@main/<path>`
— with a 7-day immutable cache and edge PoPs. The browser then layers a
Cache API bucket on top so returning users make zero network requests.

---

## Speed engine — Anime4K-CNN-2x family

### What it is

Three network width variants built on the same conv-block backbone from
[`bloc97/Anime4K`](https://github.com/bloc97/Anime4K). Each tier scales the
number of parallel feature branches: deeper isn't the answer here, **wider
is**. All three share the same 6+1 forward stages but differ in branch
count.

| File | Network | Layers | Parallel branches | Mid-channel width |
|---|---|---|---|---|
| `cnn-8.bin`  | `anime4k/cnn-2x-l`  | 17 | **2** | 16  |
| `cnn-16.bin` | `anime4k/cnn-2x-16` | 31 | **4** | 32  |
| `cnn-28.bin` | `anime4k/cnn-2x-28` | 52 | **7** | 56  |

The three weight files are reverse-engineered from
`free.upscaler.video`'s production bundle — sb2702's private `A4K\0`
binary packaging. The architecture itself is MIT (Anime4K + WebSR).
The container parser ships with the editor in `src/services/a4k-binary.ts`.

### Why no tiling

The whole network is implemented as raw WGSL compute shaders by sb2702's
[WebSR](https://github.com/sb2702/websr). Compared to a TF.js GraphModel:

- **No baked-in input shape** — input texture dimensions are runtime
  parameters in the shader, so a single weight file works at any
  resolution.
- **In-place feature-map reuse** — every layer overwrites the previous
  layer's output buffer, so peak VRAM stays around 500 MB even for 4K
  inputs.
- **Single GPU dispatch** — the entire forward pass commits one
  `device.queue.submit()`. Compare to ~300 dispatches for a 4K tiled
  TF.js run.

End result: 4K → 8K finishes in ~350 ms on a desktop WebGPU adapter.

### Auto-tier picker

The editor chooses one of the three tiers automatically based on input
pixel count. Logic is a 1:1 port of `free.upscaler.video`'s runtime
heuristic:

```
if backend == webgl AND device == mobile  →  small  (cnn-2x-l)
elif backend == webgl                      →  medium (cnn-2x-16)
elif total_pixels > 1920 × 1080 = 2.07 M  →  medium (cnn-2x-16)
else                                       →  large  (cnn-2x-28)
```

The 1080p threshold is a VRAM-budget tradeoff: cnn-2x-28 has 7-way
parallel feature maps that get expensive on 4K input but produce
visibly sharper output at ≤ 1080p. Users never see this picker; the
editor just runs it.

### Speed-engine UX inheritance from free.upscaler.video

- Image inputs are always **single-shot 2×** (matches free's
  `selectedResolution: "2x"` state machine; users wanting 4× run twice).
- The legacy "real-life / animation" picker is hidden in the speed
  engine because all three tiers share the same retrained weights —
  sb2702 doesn't bifurcate by content type at this layer.

---

## Quality engine — Real-ESRGAN-general + Real-CUGAN

Two genuinely distinct networks, picked by the user via the **照片 /
动漫** toggle. Both run on TensorFlow.js's WebGPU backend.

### `realesrgan/general_fast-192/` — for photos

- **Architecture**: SRVGGNetCompact, ~1 M parameters
- **Source**: [`xinntao/Real-ESRGAN`](https://github.com/xinntao/Real-ESRGAN)
  — the `realesr-general-x4v3` weight tier (Apache-2.0)
- **Native scale**: 4× per inference
- **Tile size**: 192×192 input → 768×768 output, 12 px overlap
- **Training data**: Real-ESRGAN's full degradation pipeline (synthetic
  noise + JPEG compression + blur kernels). Best for **real photos with
  realistic degradations** — old scans, low-quality social-media
  uploads, etc. Bicubic-trained academic SOTA models (HAT, SwinIR) often
  score higher PSNR on clean inputs but underperform on real-world dirt.
- **What "general fast" means in xinntao's zoo**: this is the **light**
  variant of Real-ESRGAN, ~1 M params instead of the 16 M `x4plus`. The
  full version doesn't fit in a browser memory budget at usable speed.

The `192` baked-in tile size is a 9× tile-count reduction over the
historical 64×64 graph. Same trained weights, different graph IO.

### `realcugan/2x-no-denoise-192/` — for anime / illustration

- **Architecture**: UNet variant (CUGAN), ~600 K parameters
- **Source**: [`bilibili/ailab/Real-CUGAN`](https://github.com/bilibili/ailab/tree/main/Real-CUGAN)
  — `up2x-latest-no-denoise` checkpoint (MIT)
- **Native scale**: 2× per inference
- **Tile size**: 192×192 input → 384×384 output, 12 px overlap
- **Training data**: bilibili's anime dataset — million-scale, modern
  digital animation. Specialised for **clean anime line art** and
  **modern flat-shaded illustrations**. Photographs run through this
  model will look "anime-fied" (lines hardened, skin tones drift toward
  a flatter palette).
- **Why no-denoise**: Real-CUGAN ships in 4 denoise levels (no-denoise
  / conservative / denoise1x / denoise3x). For clean digital sources,
  the denoise variants over-smooth genuine line detail; no-denoise
  preserves crisp strokes. If you regularly upscale low-bitrate anime
  video frames (rare in an image editor), one of the denoise variants
  would suit better.

### Why "tile + stitch" is needed here

TF.js GraphModel inputs have **fixed shapes baked in at conversion
time**. The editor uses 192-tile variants because:

- **64 × 64**: forces ~2 700 GPU dispatches for a 4K render. Dispatch
  overhead dominates wall-clock.
- **192 × 192**: ~300 dispatches. ~9× wall-clock speedup at the cost of
  ~9× peak VRAM per tile. Comfortably below WebGPU's 256 MB single-
  buffer ceiling.
- **384 × 384** (tested, abandoned): some intermediate kernels in the
  TF.js WebGPU backend hit shape-related fallback paths that cancel
  the tile-count savings. Stay at 192 until the backend matures.

### Tile stitching — overlap & blend

```
image (input) →  layoutTiles(W, H, T=192, P=12)
                       │
                       ▼
                 [tile 0, tile 1, ..., tile N]
                       │
                  for each tile:
                       crop → fromPixels → model.execute → clip[0,1]
                       → tf.browser.draw → output canvas crop
                       │
                       ▼
                 stitch using core region (un-padded centre)
                       │
                       ▼
                 final output canvas (W × scale, H × scale)
```

Each tile's *core* (the `(T - 2P) × (T - 2P)` centre region) is the
only part actually written to the stitched output. The 12-pixel padding
on each side is throwaway context that lets convolutions near the tile
edge see realistic neighbours instead of an artificial zero-padded
boundary — without this, every tile boundary would show as a faint
hairline seam.

Edge tiles slide back to keep the full T×T window inside the image, so
no mirror-padding logic is needed. The first row/col tile zeroes only
the *outer* padding (the image edge has no neighbour anyway), and the
last row/col tile zeroes only the *inner* padding (its core extends to
the image edge).

### `tf.browser.draw` over `tf.browser.toPixels`

The runTile inner loop uses `tf.browser.draw(tensor, canvas)` to blit
the model's output tensor straight into the per-tile canvas — a GPU
command, not a CPU readback. The older `tf.browser.toPixels` path
round-trips GPU → CPU → canvas and dominates wall-clock for the 299-
tile case (~15-20 s of pure readback overhead, almost half the total
inference time). `draw` cuts that to a single GPU blit per tile.

TF.js itself prints a console warning recommending `draw`; we follow.

---

## Browser usage from the editor

The editor's `src/services/upscaler-config.ts` declares URLs:

```ts
const WEIGHTS_BASE =
  'https://cdn.jsdelivr.net/gh/Jiaheliu137/piture-edit-models@main/anime4k-cnn'

export const UPSCALER_TIERS = {
  small:  { url: `${WEIGHTS_BASE}/cnn-8.bin`,  networkName: 'anime4k/cnn-2x-l',  ... },
  medium: { url: `${WEIGHTS_BASE}/cnn-16.bin`, networkName: 'anime4k/cnn-2x-16', ... },
  large:  { url: `${WEIGHTS_BASE}/cnn-28.bin`, networkName: 'anime4k/cnn-2x-28', ... },
}

export const QUALITY_UPSCALERS = {
  'real-life': {
    modelUrl: 'https://raw.githubusercontent.com/Jiaheliu137/piture-edit-models/main/realesrgan/general_fast-192/model.json',
    tileInput: 192, padSize: 12, modelScale: 4,
  },
  animation: {
    modelUrl: 'https://raw.githubusercontent.com/Jiaheliu137/piture-edit-models/main/realcugan/2x-no-denoise-192/model.json',
    tileInput: 192, padSize: 12, modelScale: 2,
  },
}
```

The speed engine fetches `.bin` files via Cache API; the quality engine
loads GraphModels via TF.js's IndexedDB IOHandler. Both persist across
page reloads — first-time users pay the download cost once, returning
users make zero network requests.

---

## Standalone usage in your own browser app

```ts
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgpu'
await tf.setBackend('webgpu')
await tf.ready()

// Quality engine — Real-ESRGAN photo
const photoModel = await tf.loadGraphModel(
  'https://raw.githubusercontent.com/Jiaheliu137/piture-edit-models/main/realesrgan/general_fast-192/model.json',
)
// Input: float32 NHWC tensor [1, 192, 192, 3] in [0, 1]
// Output: float32 NHWC tensor [1, 768, 768, 3] in [0, 1]
const result = photoModel.execute(input192) as tf.Tensor

// Speed engine — Anime4K-CNN
import WebSR from '@websr/websr'  // or piture_edit's local fork that adds cnn-2x-{16,28}
const buf = await fetch(
  'https://cdn.jsdelivr.net/gh/Jiaheliu137/piture-edit-models@main/anime4k-cnn/cnn-16.bin'
).then(r => r.arrayBuffer())
// Then: parse via piture_edit/src/services/a4k-binary.ts
```

For full tile-and-stitch logic and the auto-tier picker, mirror the
editor's `upscaler.ts` / `upscaler-quality.ts` — they're MIT-licensed
in the [piture_edit repo](https://github.com/Jiaheliu137/piture_edit).

---

## Auto-conversion path (for re-deriving the weights)

Real-CUGAN ONNX/PyTorch → TF.js GraphModel reproduction:

```bash
pip install torch onnx onnxruntime onnxscript
git clone --depth=1 https://github.com/bilibili/ailab.git
curl -L https://github.com/ZHCSOFT/Real-CUGAN/releases/download/Real-CUGAN/weights_v3.zip -o w.zip
unzip -j w.zip up2x-latest-no-denoise.pth
# Then export with dynamic spatial axes via torch.onnx.export
# (opset 17, dynamic_axes on H/W). Reference script in
# piture_edit/.playwright-mcp during the migration window.
```

Real-ESRGAN ONNX is publicly mirrored at
[`OwlMaster/AllFilesRope/realesr-general-x4v3.onnx`](https://huggingface.co/OwlMaster/AllFilesRope/blob/main/realesr-general-x4v3.onnx).
Both are converted ONNX → TF.js via `tensorflowjs_converter
--quantize_float16` to produce the `model.json` + `.bin` shards
shipped here.

---

## Why not ONNX Runtime Web

Tested in the migration window. ort-web 1.25's WebGPU JSEP backend has
two blockers for our two models:

- **Real-CUGAN**: ~30 % of nodes (PixelShuffle, SE-block reshapes) lack
  WebGPU implementations and fall back to wasm. Each op-boundary
  triggers GPU↔CPU sync. Wall-clock comes out **~6× slower** than the
  TF.js path on identical inputs.
- **Real-ESRGAN**: the `Clip` op crashes the JSEP kernel with `Failed
  to generate kernel's output[0]` at any tile size we tried (192, 384).
  Reproducible upstream bug.

Re-evaluate when ort-web 1.27+ ships op coverage parity with TF.js.

---

## Licenses

| Component | License | Upstream |
|---|---|---|
| Anime4K (network architecture) | MIT | [bloc97/Anime4K](https://github.com/bloc97/Anime4K) |
| WebSR runtime (loads Anime4K weights) | MIT | [sb2702/websr](https://github.com/sb2702/websr) |
| `cnn-8/16/28.bin` (sb2702 retrained weights) | reverse-engineered from a public web bundle; redistributed for browser interoperability |
| Real-ESRGAN | Apache-2.0 | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) |
| Real-CUGAN | MIT | [bilibili/ailab/Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) |
