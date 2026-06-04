import * as ort from 'onnxruntime-web'

const INPUT_SIZE = 640
const CONF_THRESHOLD = 0.25
const IOU_THRESHOLD = 0.45

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

function iou(a: Detection, b: Detection): number {
  const ix1 = Math.max(a.x1, b.x1)
  const iy1 = Math.max(a.y1, b.y1)
  const ix2 = Math.min(a.x2, b.x2)
  const iy2 = Math.min(a.y2, b.y2)
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1)
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
  return union > 0 ? inter / union : 0
}

function nms(dets: Detection[]): Detection[] {
  dets.sort((a, b) => b.conf - a.conf)
  const kept: Detection[] = []
  for (const d of dets) {
    if (!kept.some(k => iou(d, k) > IOU_THRESHOLD)) kept.push(d)
  }
  return kept
}

export const runInference = async (
  source: HTMLImageElement | HTMLVideoElement
): Promise<Detection[]> => {
  if (!session) throw new Error('Model not loaded')

  const { tensor, scaleX, scaleY } = preprocess(source)

  const inputName = session.inputNames[0]
  const out = await session.run({ [inputName]: tensor })
  const outputName = session.outputNames[0]
  const data = out[outputName].data as Float32Array
  const dims = out[outputName].dims

  // Handle both [1, 4+nc, 8400] and transposed [1, 8400, 4+nc]
  const isTransposed = dims[1] > dims[2]
  const numAttrs   = isTransposed ? dims[2] : dims[1]
  const numAnchors = isTransposed ? dims[1] : dims[2]

  const getValue = (attr: number, anchor: number) =>
    isTransposed
      ? data[anchor * numAttrs + attr]
      : data[attr * numAnchors + anchor]

  const detections: Detection[] = []

  for (let i = 0; i < numAnchors; i++) {
    let maxConf = 0
    let maxClass = 0
    for (let c = 4; c < numAttrs; c++) {
      const conf = getValue(c, i)
      if (conf > maxConf) { maxConf = conf; maxClass = c - 4 }
    }
    if (maxConf < CONF_THRESHOLD) continue

    const cx = getValue(0, i)
    const cy = getValue(1, i)
    const w  = getValue(2, i)
    const h  = getValue(3, i)

    detections.push({
      x1: (cx - w / 2) * scaleX,
      y1: (cy - h / 2) * scaleY,
      x2: (cx + w / 2) * scaleX,
      y2: (cy + h / 2) * scaleY,
      conf: maxConf,
      classId: maxClass,
    })
  }

  return nms(detections)
}
