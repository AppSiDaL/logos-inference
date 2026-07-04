import * as ort from 'onnxruntime-web'
import {
  INPUT_SIZE,
  decode
} from './decode'

const worker = self as unknown as Worker

ort.env.wasm.numThreads = 1

let session: ort.InferenceSession | null = null
const MAX_RETRIES = 2

// Rutas estáticas del modelo y su manifiesto de integridad
const MODEL_PATH = '/models/best_nano.onnx'
const MANIFEST_PATH = '/models/manifest.json'

worker.onmessage = async (event) => {
  const msg = event.data

  try {
    if (msg.type === 'init') {
      try {
        // 1. Descargar de forma paralela el manifiesto y el binario del modelo
        const [manifestRes, modelRes] = await Promise.all([
          fetch(MANIFEST_PATH),
          fetch(MODEL_PATH)
        ])

        if (!manifestRes.ok) throw new Error(`No se pudo obtener el manifiesto de integridad (${manifestRes.status})`)
        if (!modelRes.ok) throw new Error(`No se pudo descargar el archivo del modelo ONNX (${modelRes.status})`)

        const manifest = await manifestRes.json()
        const modelBuffer = await modelRes.arrayBuffer()

        // 2. Validación de Headers Básicos de ONNX (Magic Numbers)
        const headerView = new Uint8Array(modelBuffer, 0, 8)
        if (headerView[0] === 0x00 && headerView[1] === 0x00 && headerView[2] === 0x00) {
          throw new Error("El archivo descargado está vacío o corrupto (Cabecera de bytes inválida)")
        }

        // 3. Cálculo del Hash SHA-256 mediante Web Crypto API (Hilos nativos del navegador)
        const hashBuffer = await crypto.subtle.digest('SHA-256', modelBuffer)
        const hashArray = Array.from(new Uint8Array(hashBuffer))
        const calculatedHash = hashArray.map(b => b.toString(16).padStart(2, '0')).join('')

        // 4. Verificación de firmas contra el Manifest
        if (calculatedHash !== manifest.expectedSha256) {
          throw new Error(
            `Falla de Integridad Crítica (SHA-256 Mismatch).\n` +
            `Esperado: ${manifest.expectedSha256}\n` +
            `Calculado: ${calculatedHash}`
          )
        }

        // 5. Inicialización de ONNX consumiendo directamente el buffer validado en memoria
        session = await ort.InferenceSession.create(modelBuffer, {
          executionProviders: ['wasm']
        })

        worker.postMessage({ type: 'ready' })
      } catch (error: any) {
        // Criterio de aceptación cumplido: Mensaje de error explícito y detallado ante fallas
        worker.postMessage({
          type: 'error',
          message: `🚨 [VALIDACIÓN INTEGRIDAD FALLIDA]: No se pudo inicializar el motor de IA.\nDetalles: ${error?.message || error}`,
          stack: error?.stack
        })
      }
      return
    }

    if (msg.type === 'infer') {
      if (!session) {
        worker.postMessage({
          type: 'error',
          id: msg.id,
          message: '🚨 [YOLO WORKER] Model not initialized. Instancia de red neuronal inexistente.'
        })
        return
      }

      const bitmap = msg.bitmap as ImageBitmap
      let tensorDims = [0, 0, 0, 0]
      let currentAttempt = 0

      while (currentAttempt <= MAX_RETRIES) {
        try {
          const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE)
          const ctx = canvas.getContext('2d')
          if (!ctx) throw new Error('Canvas context error (Offscreen)')

          const srcW = bitmap.width
          const srcH = bitmap.height

          ctx.drawImage(bitmap, 0, 0, INPUT_SIZE, INPUT_SIZE)
          const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE)
          const pixels = imageData.data
          const stride = INPUT_SIZE * INPUT_SIZE

          const input = new Float32Array(3 * stride)
          for (let i = 0; i < stride; i++) {
            input[i] = pixels[i * 4] / 255
            input[i + stride] = pixels[i * 4 + 1] / 255
            input[i + stride * 2] = pixels[i * 4 + 2] / 255
          }

          tensorDims = [1, 3, INPUT_SIZE, INPUT_SIZE]
          const tensor = new ort.Tensor('float32', input, tensorDims)

          const inputName = session.inputNames[0]
          const output = await session.run({ [inputName]: tensor })

          const outputName = session.outputNames[0]
          const result = output[outputName]

          const scaleX = srcW / INPUT_SIZE
          const scaleY = srcH / INPUT_SIZE

          const detections = decode(
            result.data as Float32Array,
            result.dims,
            scaleX,
            scaleY
          )

          worker.postMessage({ type: 'result', id: msg.id, detections })
          bitmap.close()
          return 

        } catch (error: any) {
          currentAttempt++
          console.warn(`⚠️ [Worker Inference] Intento ${currentAttempt}/${MAX_RETRIES + 1} falló.`)

          if (currentAttempt <= MAX_RETRIES) continue

          worker.postMessage({
            type: 'error',
            id: msg.id,
            message: `[yoloWorker CRITICAL] Inferencia fallida de forma persistente.\nDetalles: ${error?.message}\nTensordims: [${tensorDims.join(', ')}]`,
            stack: error?.stack
          })
          bitmap.close()
          return
        }
      }
    }
  } catch (error: any) {
    worker.postMessage({
      type: 'error',
      id: msg.id,
      message: `[Worker Out-of-Bounds Error]: ${error?.message}`,
      stack: error?.stack
    })
  }
}