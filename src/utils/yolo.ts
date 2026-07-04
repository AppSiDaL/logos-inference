import * as ort from 'onnxruntime-web'
import type { Detection } from './decode'
import { decode } from './decode'

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
  0: '#8B0000',
  1: '#F58220',
  2: '#005BAC',
  3: '#D91E18',
  4: '#00A651',
  5: '#FFD400',
  6: '#E30613',
}

let worker: Worker | null = null
let requestId = 0
let isFallbackMode = false
let localSession: ort.InferenceSession | null = null 

const pending = new Map<
  number,
  {
    resolve: (detections: Detection[]) => void
    reject: (error: Error) => void
  }
>()

/**
 * Carga el modelo intentando inicializar el Web Worker con validación criptográfica.
 * Si falla la integridad o la carga, inicia la sesión local de contingencia en el hilo principal.
 */
export const loadModel = async (): Promise<void> => {
  const REAL_MODEL_PATH = '/models/best_nano.onnx'

  try {
    if (typeof Worker === 'undefined') {
      throw new Error('Web Workers no soportados en este navegador.')
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
          // Captura controlada de errores de inicialización (cuando no hay ID de petición aún)
          if (msg.id === undefined) {
            clearTimeout(timeout)
            reject(new Error(msg.message))
            return
          }

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
  } catch (error: any) {
    console.warn(
      '⚠️ [YOLO CONTINGENCIA] Web Worker falló o no es compatible. Iniciando Fallback en Hilo Principal...',
      error?.message || error
    )
    
    isFallbackMode = true
    if (worker) {
      worker.terminate()
      worker = null
    }

    try {
      // CORRECCIÓN: Apuntamos con precisión al binario real dentro de public/models/
      localSession = await ort.InferenceSession.create(REAL_MODEL_PATH, {
        executionProviders: ['wasm'], 
      })
      console.log('✅ [YOLO CONTINGENCIA] Modelo ONNX cargado con éxito en el Hilo Principal.')
    } catch (fallbackError) {
      console.error('🚨 [YOLO CRÍTICO] El fallback del hilo principal también falló:', fallbackError)
      throw fallbackError
    }
  }
}

/**
 * Convierte el ImageBitmap en los canales numéricos (Layout CHW) requeridos.
 */
const preprocessBitmap = (canvas: HTMLCanvasElement, bitmap: ImageBitmap, targetWidth = 640, targetHeight = 640): Float32Array => {
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Contexto 2D no disponible para preprocesamiento')
  
  ctx.drawImage(bitmap, 0, 0, targetWidth, targetHeight)
  const imgData = ctx.getImageData(0, 0, targetWidth, targetHeight)
  const data = imgData.data

  const floatData = new Float32Array(targetWidth * targetHeight * 3)
  
  for (let i = 0; i < data.length / 4; i++) {
    floatData[i] = data[i * 4] / 255.0                             // R
    floatData[i + targetWidth * targetHeight] = data[i * 4 + 1] / 255.0 // G
    floatData[i + targetWidth * targetHeight * 2] = data[i * 4 + 2] / 255.0 // B
  }

  return floatData
}

/**
 * Ejecuta la inferencia real. Garantiza la liberación de recursos gráficos (ImageBitmap).
 */
export const runInference = async (
  source: HTMLImageElement | HTMLVideoElement
): Promise<Detection[]> => {
  
  const bitmap = await createImageBitmap(source)
  const id = ++requestId

  try {
    // --- ESCENARIO FALLBACK (HILO PRINCIPAL) ---
    if (isFallbackMode) {
      if (!localSession) {
        throw new Error('Modelo local no disponible en el hilo principal.')
      }

      const tempCanvas = document.createElement('canvas')
      tempCanvas.width = 640
      tempCanvas.height = 640

      const inputData = preprocessBitmap(tempCanvas, bitmap, 640, 640)
      const tensorInput = new ort.Tensor('float32', inputData, [1, 3, 640, 640])
      
      const outputMap = await localSession.run({ images: tensorInput })
      
      const outputTensor = outputMap[Object.keys(outputMap)[0]]
      const rawOutput = outputTensor.data as Float32Array
      const dims = outputTensor.dims

      const srcWidth = source.nodeName === 'IMG' ? (source as HTMLImageElement).naturalWidth : (source as HTMLVideoElement).videoWidth
      const srcHeight = source.nodeName === 'IMG' ? (source as HTMLImageElement).naturalHeight : (source as HTMLVideoElement).videoHeight

      const scaleX = srcWidth / 640
      const scaleY = srcHeight / 640

      return decode(rawOutput, dims, scaleX, scaleY)
    }

    // --- ESCENARIO ESTÁNDAR (WEB WORKER) ---
    if (!worker) {
      throw new Error('Web Worker no se encuentra inicializado.')
    }

    return await new Promise<Detection[]>((resolve, reject) => {
      pending.set(id, { resolve, reject })
      
      worker!.postMessage(
        {
          type: 'infer',
          id,
          bitmap,
        },
        [bitmap]
      )
    })

  } catch (error) {
    console.error(`❌ [YOLO RUNTIME ERROR] Error procesando la petición #${id}:`, error)
    throw error 
  } finally {
    bitmap.close()
  }
}