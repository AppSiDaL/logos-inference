export const INPUT_SIZE = 640

export const RAW_FLOOR = 0.05

export interface Detection {
  x1: number
  y1: number
  x2: number
  y2: number
  conf: number
  classId: number
}

export function decode(
  data: Float32Array,
  dims: readonly number[],
  scaleX: number,
  scaleY: number
): Detection[] {

  const detections: Detection[] = []

  const numDets = dims[1]

  for (let i = 0; i < numDets; i++) {

    const offset = i * 6

    const conf = data[offset + 4]

    if (conf < RAW_FLOOR) {
      continue
    }

    detections.push({
      x1: data[offset] * scaleX,
      y1: data[offset + 1] * scaleY,
      x2: data[offset + 2] * scaleX,
      y2: data[offset + 3] * scaleY,
      conf,
      classId: Math.round(
        data[offset + 5]
      )
    })
  }

  return detections
}