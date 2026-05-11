import { useRef, useState } from 'react'

function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const [selectedFile, setSelectedFile] = useState('')
  const [previewURL, setPreviewURL] = useState('')
  const [fileType, setFileType] = useState('')

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

  const handleModelTest = () => {
    alert(
      'El modelo YOLO aún no está conectado.\n\nPróximamente podrás probar tu entrenamiento aquí.'
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

            box-shadow:
              0 10px 25px rgba(0,102,255,0.3);
          }

          .testButton{
            background:
              linear-gradient(
                135deg,
                #13325c,
                #1d4d8f
              );

            box-shadow:
              0 10px 25px rgba(0,70,180,0.25);
          }

          .uploadButton:hover,
          .testButton:hover{
            transform:
              translateY(-6px)
              scale(1.03);

            box-shadow:
              0 16px 35px rgba(0,102,255,0.45),
              0 0 18px rgba(0,102,255,0.2);
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

            animation:fadeIn 0.5s ease;
          }

          .fileInfo h3{
            color:#7fb4ff;

            margin-bottom:10px;
          }

          .previewContainer{
            margin-top:10px;

            text-align:center;

            animation:fadeIn 0.5s ease;
          }

          .previewContainer h2{
            color:white;

            margin-bottom:25px;

            font-size:2rem;
          }

          .previewImage{
            width:100%;

            max-height:500px;

            object-fit:contain;

            border-radius:20px;

            border:1px solid rgba(255,255,255,0.08);

            box-shadow:
              0 10px 30px rgba(0,0,0,0.4);
          }

          .previewVideo{
            width:100%;

            max-height:500px;

            border-radius:20px;

            border:1px solid rgba(255,255,255,0.08);

            box-shadow:
              0 10px 30px rgba(0,0,0,0.4);
          }

          @keyframes fadeIn{
            from{
              opacity:0;
              transform:translateY(15px);
            }

            to{
              opacity:1;
              transform:translateY(0);
            }
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
              🚀 Probar Modelo
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
                <img
                  src={previewURL}
                  alt="preview"
                  className="previewImage"
                />
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