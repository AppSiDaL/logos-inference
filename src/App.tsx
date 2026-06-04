import { useEffect, useRef, useState } from 'react'
import { loadModel, runInference } from './utils/yolo'
import type { Detection } from './utils/yolo'
import './App.css'

function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const imageRef     = useRef<HTMLImageElement | null>(null)
  const canvasRef    = useRef<HTMLCanvasElement | null>(null)
  const videoRef     = useRef<HTMLVideoElement | null>(null)
  const animationRef = useRef<number | null>(null)

  const [selectedFile,    setSelectedFile]    = useState('')
  const [previewURL,      setPreviewURL]      = useState('')
  const [fileType,        setFileType]        = useState('')
  const [loading,         setLoading]         = useState(false)
  const [modelReady,      setModelReady]      = useState(false)
  const [detectionCount,  setDetectionCount]  = useState<number | null>(null)
  const [isDragging,      setIsDragging]      = useState(false)

  useEffect(() => {
    loadModel().then(() => setModelReady(true)).catch(console.error)
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [])

  const handleFile = (file: File) => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }
    setSelectedFile(file.name)
    setPreviewURL(prev => { if (prev) URL.revokeObjectURL(prev); return URL.createObjectURL(file) })
    setFileType(file.type)
    setDetectionCount(null)
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file && (file.type.startsWith('image') || file.type.startsWith('video'))) handleFile(file)
  }

  const drawDetections = (
    canvas: HTMLCanvasElement,
    detections: Detection[],
    displayW: number,
    displayH: number,
    srcW: number,
    srcH: number,
  ) => {
    canvas.width  = displayW
    canvas.height = displayH
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, displayW, displayH)
    if (detections.length === 0) return

    const scaleX   = displayW / srcW
    const scaleY   = displayH / srcH
    const fontSize = Math.max(12, Math.min(16, displayW / 45))
    ctx.font = `bold ${fontSize}px -apple-system, sans-serif`

    for (const { x1, y1, x2, y2, conf, classId } of detections) {
      const dx1 = x1 * scaleX
      const dy1 = y1 * scaleY
      const dx2 = x2 * scaleX
      const dy2 = y2 * scaleY
      const label = `Logo${classId > 0 ? ` ${classId}` : ''} ${(conf * 100).toFixed(0)}%`

      ctx.strokeStyle = '#F24C3D'
      ctx.lineWidth   = Math.max(2, displayW / 300)
      ctx.strokeRect(dx1, dy1, dx2 - dx1, dy2 - dy1)

      const pad    = fontSize * 0.4
      const textW  = ctx.measureText(label).width
      const labelH = fontSize + pad * 2
      const labelY = dy1 > labelH + 4 ? dy1 - labelH : dy2 + 4

      ctx.fillStyle = '#483E8C'
      ctx.fillRect(dx1 - 1, labelY, textW + pad * 2 + 2, labelH)
      ctx.fillStyle = '#ffffff'
      ctx.fillText(label, dx1 + pad, labelY + fontSize + pad * 0.4)
    }
  }

  const handleDetect = async () => {
    if (fileType.startsWith('image'))      await runImageDetection()
    else if (fileType.startsWith('video')) runVideoDetection()
  }

  const runImageDetection = async () => {
    const img    = imageRef.current
    const canvas = canvasRef.current
    if (!img || !canvas) return
    setLoading(true)
    try {
      const detections = await runInference(img)
      drawDetections(canvas, detections, img.offsetWidth, img.offsetHeight, img.naturalWidth, img.naturalHeight)
      setDetectionCount(detections.length)
    } catch (e) {
      console.error(e)
      alert('Error ejecutando el modelo')
    } finally {
      setLoading(false)
    }
  }

  const runVideoDetection = () => {
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    if (animationRef.current) cancelAnimationFrame(animationRef.current)
    video.play()

    let frameIdx = 0
    let lastDets: Detection[] = []
    let inferring = false

    const tick = () => {
      if (video.paused || video.ended) return

      if (!inferring && frameIdx % 3 === 0 && video.videoWidth > 0) {
        inferring = true
        runInference(video)
          .then(dets => { lastDets = dets; setDetectionCount(dets.length) })
          .catch(() => {})
          .finally(() => { inferring = false })
      }

      if (video.videoWidth > 0) {
        drawDetections(canvas, lastDets, video.offsetWidth, video.offsetHeight, video.videoWidth, video.videoHeight)
      }

      frameIdx++
      animationRef.current = requestAnimationFrame(tick)
    }

    tick()
  }

  return (
    <div className="page">
      <div className="card">

        <header className="header">
          <img src="/image.png" alt="Intelite" className="logo" />
          <h1 className="title">Detección de Logotipos</h1>
          <p className="subtitle">
            Analiza imágenes y video para detectar logotipos políticos con IA
          </p>
        </header>

        <div className={`status-badge ${modelReady ? 'ready' : 'loading-status'}`}>
          <span className="dot" />
          {modelReady ? 'Modelo listo' : 'Cargando modelo…'}
        </div>

        <div
          className={`dropzone${isDragging ? ' dragging' : ''}${previewURL ? ' has-file' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          {previewURL ? (
            <span className="dropzone-filename">{selectedFile}</span>
          ) : (
            <>
              <span className="dropzone-icon">📂</span>
              <span className="dropzone-text">Arrastra una imagen o video aquí</span>
              <span className="dropzone-hint">o haz clic para seleccionar</span>
            </>
          )}
        </div>

        <input
          type="file"
          accept="image/*,video/*"
          ref={fileInputRef}
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />

        <div className="actions">
          <button className="btn btn-upload" onClick={() => fileInputRef.current?.click()}>
            Subir archivo
          </button>
          <button
            className="btn btn-detect"
            onClick={handleDetect}
            disabled={!previewURL || loading || !modelReady}
          >
            {loading ? 'Detectando…' : 'Detectar logotipos'}
          </button>
        </div>

        {detectionCount !== null && (
          <div className={`detection-result ${detectionCount > 0 ? 'found' : 'empty'}`}>
            {detectionCount === 0
              ? 'No se encontraron logotipos en este archivo'
              : `${detectionCount} logotipo${detectionCount !== 1 ? 's' : ''} detectado${detectionCount !== 1 ? 's' : ''}`}
          </div>
        )}

        {previewURL && (
          <div className="preview-section">
            <div className="preview-wrapper">
              {fileType.startsWith('image') ? (
                <>
                  <img
                    ref={imageRef}
                    src={previewURL}
                    alt="Vista previa"
                    className="preview-media"
                  />
                  <canvas ref={canvasRef} className="overlay-canvas" />
                </>
              ) : (
                <>
                  <video ref={videoRef} controls className="preview-media">
                    <source src={previewURL} type={fileType} />
                  </video>
                  <canvas ref={canvasRef} className="overlay-canvas" style={{ pointerEvents: 'none' }} />
                </>
              )}
            </div>
          </div>
        )}

      </div>
    </div>
  )
}

export default App
