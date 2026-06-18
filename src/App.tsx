import { useCallback, useEffect, useRef, useState } from 'react'
import { loadModel, runInference, CLASS_NAMES } from './utils/yolo'
import type { Detection } from './utils/yolo'
import './App.css'

function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const imageRef     = useRef<HTMLImageElement | null>(null)
  const canvasRef    = useRef<HTMLCanvasElement | null>(null)
  const videoRef     = useRef<HTMLVideoElement | null>(null)
  const animationRef = useRef<number | null>(null)
  const selectedRef  = useRef<number | null>(null)
  const streamRef    = useRef<MediaStream | null>(null)

  const [selectedFile, setSelectedFile] = useState('')
  const [previewURL,   setPreviewURL]   = useState('')
  const [fileType,     setFileType]     = useState('')
  const [loading,      setLoading]      = useState(false)
  const [modelReady,   setModelReady]   = useState(false)
  const [isDragging,   setIsDragging]   = useState(false)
  const [detections,   setDetections]   = useState<Detection[]>([])
  const [hasRun,       setHasRun]       = useState(false)
  const [selectedIdx,  setSelectedIdx]  = useState<number | null>(null)
  const [isWebcam,     setIsWebcam]     = useState(false)

  useEffect(() => {
    loadModel().then(() => setModelReady(true)).catch(console.error)
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop())
    }
  }, [])

  const stopStream = () => {
    if (animationRef.current) { cancelAnimationFrame(animationRef.current); animationRef.current = null }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null }
    const video = videoRef.current
    if (video) { video.srcObject = null }
  }

  const handleFile = (file: File) => {
    stopStream()
    setIsWebcam(false)
    setSelectedFile(file.name)
    setPreviewURL(prev => { if (prev) URL.revokeObjectURL(prev); return URL.createObjectURL(file) })
    setFileType(file.type)
    setDetections([])
    setHasRun(false)
    setSelectedIdx(null)
    selectedRef.current = null
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

  const drawDetections = useCallback((
    canvas: HTMLCanvasElement,
    dets: Detection[],
    displayW: number,
    displayH: number,
    srcW: number,
    srcH: number,
    highlightIdx: number,
  ) => {
    canvas.width  = displayW
    canvas.height = displayH
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, displayW, displayH)
    if (dets.length === 0) return

    const scaleX   = displayW / srcW
    const scaleY   = displayH / srcH
    const fontSize = Math.max(12, Math.min(16, displayW / 45))
    ctx.font = `bold ${fontSize}px -apple-system, sans-serif`

    dets.forEach((det, idx) => {
      const dx1 = det.x1 * scaleX
      const dy1 = det.y1 * scaleY
      const dx2 = det.x2 * scaleX
      const dy2 = det.y2 * scaleY
      const isHi = idx === highlightIdx
      const label = `${CLASS_NAMES[det.classId] ?? `Clase ${det.classId}`} ${(det.conf * 100).toFixed(0)}%`

      if (isHi) {
        ctx.fillStyle = 'rgba(242, 76, 61, 0.28)'
        ctx.fillRect(dx1, dy1, dx2 - dx1, dy2 - dy1)
      }

      ctx.strokeStyle = isHi ? '#F24C3D' : 'rgba(242, 76, 61, 0.8)'
      ctx.lineWidth   = Math.max(2, displayW / 300) * (isHi ? 1.8 : 1)
      ctx.strokeRect(dx1, dy1, dx2 - dx1, dy2 - dy1)

      const pad    = fontSize * 0.4
      const textW  = ctx.measureText(label).width
      const labelH = fontSize + pad * 2
      const labelY = dy1 > labelH + 4 ? dy1 - labelH : dy2 + 4

      ctx.fillStyle = isHi ? '#F24C3D' : '#483E8C'
      ctx.fillRect(dx1 - 1, labelY, textW + pad * 2 + 2, labelH)
      ctx.fillStyle = '#ffffff'
      ctx.fillText(label, dx1 + pad, labelY + fontSize + pad * 0.4)
    })
  }, [])

  useEffect(() => {
    if (!fileType.startsWith('image')) return
    const img = imageRef.current
    const canvas = canvasRef.current
    if (!img || !canvas) return
    drawDetections(canvas, detections, img.offsetWidth, img.offsetHeight, img.naturalWidth, img.naturalHeight, selectedIdx ?? -1)
  }, [detections, selectedIdx, fileType, drawDetections])

  const handleDetect = async () => {
    if (fileType.startsWith('image'))      await runImageDetection()
    else if (fileType.startsWith('video')) runVideoDetection()
  }

  const runImageDetection = async () => {
    const img = imageRef.current
    if (!img) return
    setLoading(true)
    try {
      const dets = await runInference(img)
      setSelectedIdx(null)
      selectedRef.current = null
      setDetections(dets)
      setHasRun(true)
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
    setHasRun(true)

    let frameIdx = 0
    let lastDets: Detection[] = []
    let inferring = false

    const tick = () => {
      if (video.ended) { animationRef.current = null; return }

      if (!video.paused) {
        if (!inferring && frameIdx % 3 === 0 && video.videoWidth > 0) {
          inferring = true
          runInference(video)
            .then(dets => { lastDets = dets; setDetections(dets) })
            .catch(() => {})
            .finally(() => { inferring = false })
        }
        if (video.videoWidth > 0) {
          drawDetections(canvas, lastDets, video.offsetWidth, video.offsetHeight, video.videoWidth, video.videoHeight, selectedRef.current ?? -1)
        }
        frameIdx++
      }

      animationRef.current = requestAnimationFrame(tick)
    }

    tick()
  }

  const startWebcam = async () => {
    stopStream()
    setPreviewURL(prev => { if (prev) URL.revokeObjectURL(prev); return '' })
    setSelectedFile('')
    setFileType('')
    setDetections([])
    setHasRun(false)
    setSelectedIdx(null)
    selectedRef.current = null
    setIsWebcam(true)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      streamRef.current = stream
      const video = videoRef.current!
      video.srcObject = stream
      await video.play()
      setHasRun(true)
      runWebcamLoop()
    } catch {
      alert('No se pudo acceder a la cámara')
      setIsWebcam(false)
    }
  }

  const stopWebcam = () => {
    stopStream()
    setIsWebcam(false)
    setDetections([])
    setHasRun(false)
  }

  const runWebcamLoop = () => {
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    let frameIdx = 0
    let lastDets: Detection[] = []
    let inferring = false

    const tick = () => {
      if (!streamRef.current) { animationRef.current = null; return }

      if (!inferring && frameIdx % 3 === 0 && video.videoWidth > 0) {
        inferring = true
        runInference(video)
          .then(dets => { lastDets = dets; setDetections(dets) })
          .catch(() => {})
          .finally(() => { inferring = false })
      }
      if (video.videoWidth > 0) {
        drawDetections(canvas, lastDets, video.offsetWidth, video.offsetHeight, video.videoWidth, video.videoHeight, selectedRef.current ?? -1)
      }

      frameIdx++
      animationRef.current = requestAnimationFrame(tick)
    }

    tick()
  }

  const selectDetection = (i: number) => {
    const next = selectedIdx === i ? null : i
    setSelectedIdx(next)
    selectedRef.current = next
  }

  const showMedia = previewURL || isWebcam

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

        {!isWebcam && (
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
        )}

        <input
          type="file"
          accept="image/*,video/*"
          ref={fileInputRef}
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />

        <div className="actions">
          {!isWebcam ? (
            <>
              <button className="btn btn-upload" onClick={() => fileInputRef.current?.click()}>
                Subir archivo
              </button>
              <button
                className="btn btn-webcam"
                onClick={startWebcam}
                disabled={!modelReady}
              >
                Usar webcam
              </button>
              <button
                className="btn btn-detect"
                onClick={handleDetect}
                disabled={!previewURL || loading || !modelReady}
              >
                {loading ? 'Detectando…' : 'Detectar logotipos'}
              </button>
            </>
          ) : (
            <>
              <button className="btn btn-upload" onClick={() => fileInputRef.current?.click()}>
                Subir archivo
              </button>
              <button className="btn btn-webcam btn-webcam-stop" onClick={stopWebcam}>
                Detener webcam
              </button>
            </>
          )}
        </div>

        {showMedia && (
          <div className="results-layout">
            <div className="preview-section">
              <div className="preview-wrapper">
                {isWebcam ? (
                  <>
                    <video
                      ref={videoRef}
                      className="preview-media"
                      autoPlay
                      muted
                      playsInline
                    />
                    <canvas ref={canvasRef} className="overlay-canvas" style={{ pointerEvents: 'none' }} />
                  </>
                ) : fileType.startsWith('image') ? (
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

            {hasRun && (
              <aside className="detections-panel">
                <div className="panel-header">
                  <span className="panel-title">Detecciones</span>
                  <span className="panel-count">{detections.length}</span>
                </div>

                {detections.length === 0 ? (
                  <p className="panel-empty">Sin logotipos detectados</p>
                ) : (
                  <ul className="detection-list">
                    {detections.map((d, i) => (
                      <li
                        key={i}
                        className={`detection-item${selectedIdx === i ? ' active' : ''}`}
                        onClick={() => selectDetection(i)}
                      >
                        <span className="detection-class">
                          {CLASS_NAMES[d.classId] ?? `Clase ${d.classId}`}
                        </span>
                        <span className="detection-conf">{(d.conf * 100).toFixed(0)}%</span>
                      </li>
                    ))}
                  </ul>
                )}
              </aside>
            )}
          </div>
        )}

      </div>
    </div>
  )
}

export default App
