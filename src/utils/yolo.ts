import * as ort from 'onnxruntime-web'

let session: ort.InferenceSession

export const loadModel = async () => {
  session = await ort.InferenceSession.create(
    '/models/best.onnx'
  )

  console.log('Modelo cargado')
}

export const getSession = () => session