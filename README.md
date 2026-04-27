# piture-edit-models

Pre-converted TensorFlow.js super-resolution models for browser-based local upscaling.

## Models

### `realesrgan/general_fast-64/`
- **Source**: Real-ESRGAN-general-x4v3 (`xinntao/Real-ESRGAN`, Apache-2.0)
- **Scale**: 4×
- **Tile size**: 64×64 input → 256×256 output
- **Use case**: General-purpose photo upscaling
- **Format**: TensorFlow.js GraphModel, FP16 weights

### `realcugan/2x-no-denoise-64/`
- **Source**: Real-CUGAN 2x no-denoise (`bilibili/ailab/Real-CUGAN`, MIT)
- **Scale**: 2×
- **Tile size**: 64×64 input → 128×128 output
- **Use case**: Anime / illustration upscaling
- **Format**: TensorFlow.js GraphModel, FP16 weights

## Conversion Pipeline

PyTorch → ONNX → TensorFlow SavedModel → TensorFlow.js (via tfjs-converter).

Conversion was performed by [`xororz/web-realesrgan`](https://github.com/xororz/web-realesrgan)
(GPL-2.0). Only the converted model weights are republished here; no GPL code
from that project is included.

## Usage in browser

```ts
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgpu'

const model = await tf.loadGraphModel(
  'https://github.com/<USER>/piture-edit-models/releases/download/v1/realesrgan_general_fast-64_model.json'
)
// or via raw.githubusercontent.com if hosted on a branch
```

## Licenses

Each model carries its upstream license. See `LICENSE` files in each subdirectory
or upstream projects:
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- Real-CUGAN: https://github.com/bilibili/ailab/tree/main/Real-CUGAN
