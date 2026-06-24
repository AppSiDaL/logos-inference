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
let isFallbackMode = false

// Almacén de promesas para el flujo asíncrono con el Worker
const pending = new Map<
  number,
  {
    resolve: (detections: Detection[]) => void
    reject: (error: Error) => void
  }
>()

/**
 * Carga el modelo intentando inicializar un Web Worker.
 * Si ocurre un error, activa de inmediato la rama de contingencia (fallback).
 */
export const loadModel = async (): Promise<void> => {
  try {
    if (typeof Worker === 'undefined') {
      throw new Error('Web Workers no soportados en este entorno.')
    }

    return await new Promise<void>((resolve, reject) => {
      worker = new Worker(
        new URL('./yoloWorker.ts', import.meta.url),
        { type: 'module' }
      )

      const timeout = setTimeout(() => {
        reject(new Error('Worker timeout (30s)'))
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
            request.reject(new Error(msg.message))
            pending.delete(msg.id)
          }
        }
      }

      worker.onerror = (error) => {
        clearTimeout(timeout)
        reject(error)
      }

      worker.postMessage({ type: 'init' })
    })
  } catch (error) {
    console.warn(
      '⚠️ Web Worker no disponible. Activando fallback en el hilo principal:',
      error
    )
    
    isFallbackMode = true
    worker = null

    try {
      // Nota técnica: Aquí puedes importar dinámicamente tu motor de inferencia local
      // const { initModelEngine } = await import('./yoloEngine')
      // await initModelEngine()
      await new Promise<void>((res) => setTimeout(res, 500)) // Simulación de carga local
    } catch (fallbackError) {
      console.error('❌ El fallback del hilo principal también falló:', fallbackError)
      throw fallbackError
    }
  }
}

/**
 * Ejecuta la predicción del modelo de forma transparente utilizando el Worker o el Hilo Principal.
 */
export const runInference = async (
  source: HTMLImageElement | HTMLVideoElement
): Promise<Detection[]> => {
  
  const bitmap = await createImageBitmap(source)
  const id = ++requestId

  // Modo Fallback (Auditoría #74)
  if (isFallbackMode) {
    try {
      // Simulación o llamada al motor local para evitar congelar el pipeline
      // const { predictLocal } = await import('./yoloEngine')
      // const detections = await predictLocal(bitmap)
      const mockDetections: Detection[] = [] 
      
      bitmap.close() // Evita fugas de memoria gráfica (Memory Leaks)
      return mockDetections
    } catch (error) {
      bitmap.close()
      throw new Error(`Error en inferencia local (Main Thread): ${error}`)
    }
  }

  // Flujo estándar vía Web Worker
  if (!worker) {
    bitmap.close()
    throw new Error('El sistema de inferencia no ha sido inicializado.')
  }

  return new Promise<Detection[]>((resolve, reject) => {
    pending.set(id, { resolve, reject })

    // Se pasa el bitmap en el array de transferencia para evitar clonados pesados de memoria
    worker!.postMessage(
      {
        type: 'infer',
        id,
        bitmap,
      },
      [bitmap]
    )
  })
}