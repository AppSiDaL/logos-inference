import * as ort from 'onnxruntime-web'

const INPUT_SIZE = 640
const CONF_THRESHOLD = 0.25

let session: ort.InferenceSession | null = null

export const CLASS_NAMES: Record<number, string> = {
  0: 'Morena',
  1: 'Movimiento Ciudadano',
  2: 'PAN',
  3: 'PT',
  4: 'Verde',
  5: 'PRD',
  6: 'PRI',
}

export interface Detection {
  x1: number
  y1: number
  x2: number
  y2: number
  conf: number
  classId: number
}

export const loadModel = async () => {
  session = await ort.InferenceSession.create('/models/best_nano.onnx')
}

function preprocess(source: HTMLImageElement | HTMLVideoElement): {
  tensor: ort.Tensor
  scaleX: number
  scaleY: number
} {
  const srcW = source instanceof HTMLImageElement ? source.naturalWidth : source.videoWidth
  const srcH = source instanceof HTMLImageElement ? source.naturalHeight : source.videoHeight

  const canvas = document.createElement('canvas')
  canvas.width = INPUT_SIZE
  canvas.height = INPUT_SIZE
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(source, 0, 0, INPUT_SIZE, INPUT_SIZE)
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE)

  const stride = INPUT_SIZE * INPUT_SIZE
  const float32 = new Float32Array(3 * stride)
  for (let i = 0; i < stride; i++) {
    float32[i]           = data[i * 4]     / 255
    float32[i + stride]  = data[i * 4 + 1] / 255
    float32[i + stride * 2] = data[i * 4 + 2] / 255
  }

  return {
    tensor: new ort.Tensor('float32', float32, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    scaleX: srcW / INPUT_SIZE,
    scaleY: srcH / INPUT_SIZE,
  }
}

export const runInference = async (
  source: HTMLImageElement | HTMLVideoElement
): Promise<Detection[]> => {
  if (!session) throw new Error('Model not loaded')

  const { tensor, scaleX, scaleY } = preprocess(source)

  const inputName = session.inputNames[0]
  const out = await session.run({ [inputName]: tensor })
  const data = out[session.outputNames[0]].data as Float32Array
  const dims = out[session.outputNames[0]].dims

  // End-to-end (NMS-free) output: [1, N, 6] => [x1, y1, x2, y2, score, classId]
  // Boxes already filtered/deduplicated by the model, in 640x640 pixel space.
  const numDets = dims[1]
  const detections: Detection[] = []

  for (let i = 0; i < numDets; i++) {
    const o = i * 6
    const conf = data[o + 4]
    if (conf < CONF_THRESHOLD) continue

    detections.push({
      x1: data[o]     * scaleX,
      y1: data[o + 1] * scaleY,
      x2: data[o + 2] * scaleX,
      y2: data[o + 3] * scaleY,
      conf,
      classId: Math.round(data[o + 5]),
    })
  }

  return detections
}
