import * as ort from 'onnxruntime-web'

import {
  INPUT_SIZE,
  decode
} from './decode'

const worker =
  self as unknown as Worker

ort.env.wasm.numThreads = 1

let session:
  ort.InferenceSession | null = null

worker.onmessage =
  async (event) => {

    const msg = event.data

    try {

      if (msg.type === 'init') {

        session =
          await ort.InferenceSession.create(
            '/models/best_nano.onnx'
          )

        worker.postMessage({
          type: 'ready'
        })

        return
      }

      if (msg.type === 'infer') {

        if (!session) {

          worker.postMessage({
            type: 'error',
            id: msg.id,
            message:
              'Model not initialized'
          })

          return
        }

        const bitmap =
          msg.bitmap as ImageBitmap

        const canvas =
          new OffscreenCanvas(
            INPUT_SIZE,
            INPUT_SIZE
          )

        const ctx =
          canvas.getContext('2d')

        if (!ctx) {

          worker.postMessage({
            type: 'error',
            id: msg.id,
            message:
              'Canvas context error'
          })

          return
        }

        const srcW =
          bitmap.width

        const srcH =
          bitmap.height

        ctx.drawImage(
          bitmap,
          0,
          0,
          INPUT_SIZE,
          INPUT_SIZE
        )

        const imageData =
          ctx.getImageData(
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
          )

        const pixels =
          imageData.data

        const stride =
          INPUT_SIZE *
          INPUT_SIZE

        const input =
          new Float32Array(
            3 * stride
          )

        for (
          let i = 0;
          i < stride;
          i++
        ) {

          input[i] =
            pixels[i * 4] / 255

          input[
            i + stride
          ] =
            pixels[
              i * 4 + 1
            ] / 255

          input[
            i + stride * 2
          ] =
            pixels[
              i * 4 + 2
            ] / 255
        }

        const tensor =
          new ort.Tensor(
            'float32',
            input,
            [
              1,
              3,
              INPUT_SIZE,
              INPUT_SIZE
            ]
          )

        const inputName =
          session.inputNames[0]

        const output =
          await session.run({
            [inputName]:
              tensor
          })

        const outputName =
          session.outputNames[0]

        const result =
          output[outputName]

        const scaleX =
          srcW / INPUT_SIZE

        const scaleY =
          srcH / INPUT_SIZE

        const detections =
          decode(
            result.data as Float32Array,
            result.dims,
            scaleX,
            scaleY
          )

        worker.postMessage({
          type: 'result',
          id: msg.id,
          detections
        })

        bitmap.close()

        return
      }

    } catch (error) {

      worker.postMessage({
        type: 'error',
        id: msg.id,
        message:
          error instanceof Error
            ? error.message
            : 'Unknown error'
      })
    }
  }