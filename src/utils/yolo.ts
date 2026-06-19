import type { Detection } from './decode'

export const CLASS_NAMES: Record<number, string> = {
  0: 'Morena',
  1: 'Movimiento Ciudadano',
  2: 'PAN',
  3: 'PT',
  4: 'Verde',
  5: 'PRD',
  6: 'PRI',
}

export const CLASS_COLORS: Record<number, string> = {
  0: '#8B0000', // Morena
  1: '#F58220', // MC
  2: '#005BAC', // PAN
  3: '#D91E18', // PT
  4: '#00A651', // Verde
  5: '#FFD400', // PRD
  6: '#E30613', // PRI
}

let worker: Worker | null = null
let requestId = 0

const pending = new Map<
  number,
  {
    resolve: (detections: Detection[]) => void
    reject: (error: Error) => void
  }
>()

export const loadModel = async () => {
  return new Promise<void>((resolve, reject) => {

    worker = new Worker(
      new URL('./yoloWorker.ts', import.meta.url),
      { type: 'module' }
    )

    const timeout = setTimeout(() => {
      reject(new Error('Worker timeout'))
    }, 30000)

    worker.onmessage = (event) => {

      const msg = event.data

      if (msg.type === 'ready') {
        clearTimeout(timeout)
        resolve()
        return
      }

      if (msg.type === 'result') {

        const request = pending.get(msg.id)

        if (request) {
          request.resolve(msg.detections)
          pending.delete(msg.id)
        }

        return
      }

      if (msg.type === 'error') {

        const request = pending.get(msg.id)

        if (request) {
          request.reject(
            new Error(msg.message)
          )
          pending.delete(msg.id)
        }
      }
    }

    worker.onerror = (error) => {
      clearTimeout(timeout)
      reject(error)
    }

    worker.postMessage({
      type: 'init'
    })
  })
}

export const runInference = async (
  source: HTMLImageElement | HTMLVideoElement
): Promise<Detection[]> => {

  if (!worker) {
    throw new Error('Worker not initialized')
  }

  const bitmap =
    await createImageBitmap(source)

  const id = ++requestId

  return new Promise<Detection[]>(
    (resolve, reject) => {

      pending.set(id, {
        resolve,
        reject
      })

      worker!.postMessage(
        {
          type: 'infer',
          id,
          bitmap
        },
        [bitmap]
      )
    }
  )
}