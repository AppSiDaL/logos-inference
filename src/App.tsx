import { useEffect, useRef, useState } from 'react'
import * as ort from 'onnxruntime-web'
import { loadModel, getSession } from './utils/yolo'

function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const imageRef = useRef<HTMLImageElement | null>(null)

  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  const [selectedFile, setSelectedFile] = useState('')
  const [previewURL, setPreviewURL] = useState('')
  const [fileType, setFileType] = useState('')

  const [loading, setLoading] = useState(false)

  const [modelReady, setModelReady] = useState(false)

  useEffect(() => {
    const initModel = async () => {
      await loadModel()

      setModelReady(true)

      console.log('YOLO listo')
    }

    initModel()
  }, [])

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0]

    if (file) {
      setSelectedFile(file.name)

      const url = URL.createObjectURL(file)

      setPreviewURL(url)

      setFileType(file.type)
    }
  }

  const handleModelTest = async () => {
    if (!imageRef.current) {
      alert('Primero sube una imagen')

      return
    }

    try {
      setLoading(true)

      const image = imageRef.current

      const session = getSession()

      const tempCanvas =
        document.createElement('canvas')

      tempCanvas.width = 640
      tempCanvas.height = 640

      const ctx = tempCanvas.getContext('2d')

      if (!ctx) return

      ctx.drawImage(image, 0, 0, 640, 640)

      const imageData = ctx.getImageData(
        0,
        0,
        640,
        640
      )

      const { data } = imageData

      const float32Data =
        new Float32Array(3 * 640 * 640)

      for (let i = 0; i < 640 * 640; i++) {
        float32Data[i] = data[i * 4] / 255

        float32Data[i + 640 * 640] =
          data[i * 4 + 1] / 255

        float32Data[i + 640 * 640 * 2] =
          data[i * 4 + 2] / 255
      }

      const tensor = new ort.Tensor(
        'float32',
        float32Data,
        [1, 3, 640, 640]
      )

      const results = await session.run({
        images: tensor
      })

      console.log(results)

      drawFakeDetection()

      setLoading(false)
    } catch (error) {
      console.error(error)

      alert('Error ejecutando el modelo')

      setLoading(false)
    }
  }

  const drawFakeDetection = () => {
    const canvas = canvasRef.current

    const image = imageRef.current

    if (!canvas || !image) return

    canvas.width = image.width

    canvas.height = image.height

    const ctx = canvas.getContext('2d')

    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    ctx.strokeStyle = '#00ff88'

    ctx.lineWidth = 5

    ctx.strokeRect(120, 120, 260, 260)

    ctx.fillStyle = '#00ff88'

    ctx.font = '24px Arial'

    ctx.fillText(
      'Logo Detectado',
      120,
      105
    )
  }

  return (
    <>
      <style>
        {`
          *{
            margin:0;
            padding:0;
            box-sizing:border-box;
            font-family:Arial, Helvetica, sans-serif;
          }

          body{
            background:
              radial-gradient(circle at top left, #153b72 0%, transparent 35%),
              radial-gradient(circle at bottom right, #0c2b58 0%, transparent 35%),
              linear-gradient(135deg, #050b16, #091425, #10233d);

            min-height:100vh;
            overflow-x:hidden;
          }

          .container{
            width:100%;
            min-height:100vh;

            display:flex;
            justify-content:center;
            align-items:center;

            padding:30px;
          }

          .card{
            width:100%;
            max-width:950px;

            background:rgba(10,20,40,0.82);

            border:1px solid rgba(255,255,255,0.08);

            border-radius:28px;

            padding:50px;

            backdrop-filter:blur(12px);

            box-shadow:
              0 0 30px rgba(0,102,255,0.15),
              0 0 70px rgba(0,102,255,0.08);
          }

          .title{
            text-align:center;

            font-size:3.2rem;

            color:white;

            margin-bottom:15px;
          }

          .subtitle{
            text-align:center;

            color:#b9c8e0;

            font-size:1.1rem;

            line-height:1.7;

            max-width:700px;

            margin:auto auto 45px auto;
          }

          .buttonContainer{
            display:flex;
            justify-content:center;
            gap:25px;

            flex-wrap:wrap;

            margin-bottom:40px;
          }

          .uploadButton,
          .testButton{
            border:none;

            padding:18px 32px;

            border-radius:18px;

            font-size:1rem;
            font-weight:bold;

            cursor:pointer;

            min-width:260px;

            transition:0.35s ease;

            position:relative;

            overflow:hidden;

            color:white;
          }

          .uploadButton{
            background:
              linear-gradient(
                135deg,
                #0b63f6,
                #3a8dff
              );
          }

          .testButton{
            background:
              linear-gradient(
                135deg,
                #13325c,
                #1d4d8f
              );
          }

          .uploadButton:hover,
          .testButton:hover{
            transform:
              translateY(-6px)
              scale(1.03);
          }

          .uploadButton:active,
          .testButton:active{
            transform:scale(0.96);
          }

          .fileInfo{
            background:rgba(255,255,255,0.05);

            border:1px solid rgba(255,255,255,0.08);

            border-radius:18px;

            padding:22px;

            color:white;

            margin-bottom:35px;
          }

          .fileInfo h3{
            color:#7fb4ff;

            margin-bottom:10px;
          }

          .previewContainer{
            margin-top:10px;

            text-align:center;
          }

          .previewContainer h2{
            color:white;

            margin-bottom:25px;

            font-size:2rem;
          }

          .previewWrapper{
            position:relative;

            display:inline-block;
          }

          .previewImage{
            width:100%;

            max-height:500px;

            object-fit:contain;

            border-radius:20px;

            border:1px solid rgba(255,255,255,0.08);
          }

          .previewVideo{
            width:100%;

            max-height:500px;

            border-radius:20px;
          }

          .canvas{
            position:absolute;

            top:0;
            left:0;
          }

          .status{
            text-align:center;

            margin-bottom:25px;

            color:#9dd2ff;

            font-weight:bold;
          }

          @media(max-width:768px){

            .card{
              padding:30px;
            }

            .title{
              font-size:2.3rem;
            }

            .buttonContainer{
              flex-direction:column;
              align-items:center;
            }

            .uploadButton,
            .testButton{
              width:100%;
            }
          }
        `}
      </style>

      <div className="container">
        <div className="card">

          <h1 className="title">
            Sistema de vision por computadora
          </h1>

          <p className="subtitle">
            Plataforma para subir imágenes o videos y probar
            modelos de inteligencia artificial de vision
            por computadora.
          </p>

          <div className="status">
            {modelReady
              ? '✅ Modelo YOLO cargado'
              : '⏳ Cargando modelo YOLO...'}
          </div>

          <div className="buttonContainer">

            <button
              className="uploadButton"
              onClick={handleUploadClick}
            >
              📁 Subir Imagen o Video
            </button>

            <button
              className="testButton"
              onClick={handleModelTest}
            >
              {loading
                ? '⏳ Detectando...'
                : '🚀 Probar Modelo'}
            </button>

          </div>

          <input
            type="file"
            accept="image/*,video/*"
            ref={fileInputRef}
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />

          {selectedFile && (
            <div className="fileInfo">

              <h3>Archivo Seleccionado</h3>

              <p>{selectedFile}</p>

            </div>
          )}

          {previewURL && (
            <div className="previewContainer">

              <h2>Vista Previa</h2>

              {fileType.startsWith('image') ? (
                <div className="previewWrapper">

                  <img
                    ref={imageRef}
                    src={previewURL}
                    alt="preview"
                    className="previewImage"
                  />

                  <canvas
                    ref={canvasRef}
                    className="canvas"
                  />

                </div>
              ) : (
                <video
                  controls
                  className="previewVideo"
                >
                  <source
                    src={previewURL}
                    type={fileType}
                  />
                </video>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  )
}

export default App