import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useMemo
} from 'react'
import { useFpsTracker } from './hooks/useFpsTracker'
import {
  loadModel,
  runInference,
  CLASS_NAMES
} from './utils/yolo'
import type { Detection } from './utils/yolo'
import './App.css'

const HISTORY_MAX = 8

interface HistoryItem {
  id: string
  name: string
  time: string
  total: number
  counts: Record<string, number>
  thumb: string
}

function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const imageRef     = useRef<HTMLImageElement | null>(null)
  const canvasRef    = useRef<HTMLCanvasElement | null>(null)
  const videoRef     = useRef<HTMLVideoElement | null>(null)
  const animationRef = useRef<number | null>(null)
  const selectedRef  = useRef<number | null>(null)
  const streamRef    = useRef<MediaStream | null>(null)

  const isLoopActiveRef = useRef<boolean>(false)

  const [selectedFile, setSelectedFile] = useState('')
  const [previewURL,   setPreviewURL]   = useState('')
  const [fileType,     setFileType]     = useState('')
  const [loading,      setLoading]      = useState(false) // <-- ESTADO DE CONTROL DE INFERENCIA
  const [modelReady,   setModelReady]   = useState(false)
  const [isDragging,   setIsDragging]   = useState(false)
  const [rawDetections, setRawDetections] = useState<Detection[]>([])
  const [confThreshold, setConfThreshold] = useState(0.25)
  const [visibleClasses, setVisibleClasses] = useState<Record<number, boolean>>(
    () =>
      Object.fromEntries(
        Object.keys(CLASS_NAMES).map(key => [
          Number(key),
          true
        ])
      )
  )

  const visibleClassesRef = useRef(visibleClasses)
  const confRef = useRef(0.25)
  const [hasRun,       setHasRun]       = useState(false)
  const [selectedIdx,  setSelectedIdx]  = useState<number | null>(null)
  const [isWebcam,     setIsWebcam]     = useState(false)

  // --- INTEGRACIÓN DEL CUSTOM HOOK DE PERFORMANCE ---
  const { fps, trackInference: trackInferenceFps, resetFps: resetFpsStats } = useFpsTracker()

  // --- HISTORIAL ---
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      const stored = localStorage.getItem('logo-history')
      if (stored) {
        const parsed = JSON.parse(stored)
        if (Array.isArray(parsed)) return parsed
      }
    } catch (e) {
      console.error('Historial inaccesible:', e)
    }
    return []
  })

  // --- FILTRADOS ---
  const detections = useMemo(() => {
    return rawDetections.filter(
      d => d.conf >= confThreshold && visibleClasses[d.classId]
    )
  }, [rawDetections, confThreshold, visibleClasses])

  const classStats = useMemo(() => {
    const counts: Record<number, number> = {}
    detections.forEach(det => {
      counts[det.classId] = (counts[det.classId] || 0) + 1
    })
    return Object.entries(counts)
      .map(([classId, count]) => ({
        classId: Number(classId),
        name: CLASS_NAMES[Number(classId)],
        count
      }))
      .sort((a, b) => b.count - a.count)
  }, [detections])

  useEffect(() => { confRef.current = confThreshold }, [confThreshold])
  useEffect(() => { visibleClassesRef.current = visibleClasses }, [visibleClasses])
  useEffect(() => { setSelectedIdx(null); selectedRef.current = null }, [confThreshold, visibleClasses])

  useEffect(() => {
    loadModel().then(() => setModelReady(true)).catch(console.error)
    return () => {
      isLoopActiveRef.current = false
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          track.stop()
          track.enabled = false
        })
      }
    }
  }, [])

  const stopStream = () => {
    isLoopActiveRef.current = false
    if (animationRef.current) { 
      cancelAnimationFrame(animationRef.current)
      animationRef.current = null 
    }
    if (streamRef.current) { 
      streamRef.current.getTracks().forEach(track => {
        track.stop()
        track.enabled = false
      })
      streamRef.current = null 
    }
    const video = videoRef.current
    if (video) { 
      video.pause()
      video.srcObject = null 
      video.load()
    }
    resetFpsStats()
  }

  const handleFile = (file: File) => {
    if (loading) return // Bloqueo funcional defensivo
    stopStream()
    setIsWebcam(false)
    setSelectedFile(file.name)
    setPreviewURL(prev => { if (prev) URL.revokeObjectURL(prev); return '' })
    setPreviewURL(URL.createObjectURL(file))
    setFileType(file.type)
    setRawDetections([])
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
    if (loading) {
      setIsDragging(false)
      return
    }
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
    if (canvas.width !== displayW || canvas.height !== displayH) {
      canvas.width = displayW
      canvas.height = displayH
    }

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
      const confidenceColor = `hsl(${det.conf * 120}, 100%, 45%)`

      if (isHi) {
        ctx.fillStyle = 'rgba(242, 76, 61, 0.25)'
        ctx.fillRect(dx1, dy1, dx2 - dx1, dy2 - dy1)
      }

      ctx.strokeStyle = isHi ? '#F24C3D' : confidenceColor
      ctx.lineWidth   = Math.max(2, displayW / 300) * (isHi ? 1.8 : 1)
      ctx.strokeRect(dx1, dy1, dx2 - dx1, dy2 - dy1)

      const pad    = fontSize * 0.4
      const textW  = ctx.measureText(label).width
      const labelH = fontSize + pad * 2
      const labelY = dy1 > labelH + 4 ? dy1 - labelH : dy2 + 4

      ctx.fillStyle = isHi ? '#F24C3D' : confidenceColor
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
    setLoading(true) // <-- Activación del estado de carga
    try {
      const dets = await runInference(img)
      setSelectedIdx(null)
      selectedRef.current = null
      setRawDetections(dets)
      setHasRun(true)

      const currentFiltered = dets.filter(d => d.conf >= confThreshold && visibleClasses[d.classId])
      const counts: Record<string, number> = {}
      currentFiltered.forEach(d => {
        const name = CLASS_NAMES[d.classId] || `Clase ${d.classId}`
        counts[name] = (counts[name] || 0) + 1
      })

      const thumbCanvas = document.createElement('canvas')
      const thumbCtx = thumbCanvas.getContext('2d')
      if (thumbCtx) {
        const maxDim = 96
        let tw = img.naturalWidth; let th = img.naturalHeight
        if (tw > th) { th = Math.round((th * maxDim) / tw); tw = maxDim } 
        else { tw = Math.round((tw * maxDim) / th); th = maxDim }
        thumbCanvas.width = tw; thumbCanvas.height = th
        thumbCtx.drawImage(img, 0, 0, tw, th)
        const thumbDataUrl = thumbCanvas.toDataURL('image/jpeg', 0.6)

        const newItem: HistoryItem = {
          id: crypto.randomUUID?.() || String(Date.now()),
          name: selectedFile,
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          total: currentFiltered.length, counts, thumb: thumbDataUrl
        }
        setHistory(prev => {
          const updated = [newItem, ...prev].slice(0, HISTORY_MAX)
          try { localStorage.setItem('logo-history', JSON.stringify(updated)) } catch (e) {}
          return updated
        })
      }
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false) // <-- Desactivación del estado de carga (Garantiza el desbloqueo)
    }
  }

  const runVideoDetection = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return
    if (animationRef.current) cancelAnimationFrame(animationRef.current)

    video.play()
    setHasRun(true)
    resetFpsStats()

    isLoopActiveRef.current = true
    let lastDets: Detection[] = []
    let inferring = false

    const tick = () => {
      if (!isLoopActiveRef.current || video.ended) { 
        animationRef.current = null
        return 
      }

      if (!video.paused && video.videoWidth > 0) {
        if (!inferring) {
          inferring = true
          runInference(video)
            .then(dets => {
              if (isLoopActiveRef.current) {
                setRawDetections(dets)
                lastDets = dets
                trackInferenceFps()
              }
            })
            .catch(() => {})
            .finally(() => {
              inferring = false
            })
        }
        const filtered = lastDets.filter(d => d.conf >= confRef.current && visibleClassesRef.current[d.classId])
        drawDetections(canvas, filtered, video.offsetWidth, video.offsetHeight, video.videoWidth, video.videoHeight, selectedRef.current ?? -1)
      }
      animationRef.current = requestAnimationFrame(tick)
    }
    tick()
  }

  const startWebcam = async () => {
    stopStream()
    setPreviewURL('')
    setSelectedFile(''); setFileType(''); setRawDetections([]); setHasRun(false); setSelectedIdx(null); selectedRef.current = null
    setIsWebcam(true); resetFpsStats()

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
    setRawDetections([]) 
    setHasRun(false) 
    setSelectedIdx(null) 
    selectedRef.current = null 
  }

  const runWebcamLoop = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    isLoopActiveRef.current = true
    let lastDets: Detection[] = []
    let inferring = false

    const tick = () => {
      if (!isLoopActiveRef.current || !streamRef.current) { 
        animationRef.current = null
        return 
      }

      if (video.videoWidth > 0) {
        if (!inferring) {
          inferring = true
          runInference(video)
            .then(dets => {
              if (isLoopActiveRef.current) { 
                setRawDetections(dets)
                lastDets = dets
                trackInferenceFps()
              }
            })
            .catch(() => {})
            .finally(() => {
              inferring = false
            })
        }
        const filtered = lastDets.filter(d => d.conf >= confRef.current && visibleClassesRef.current[d.classId])
        drawDetections(canvas, filtered, video.offsetWidth, video.offsetHeight, video.videoWidth, video.videoHeight, selectedRef.current ?? -1)
      }
      animationRef.current = requestAnimationFrame(tick)
    }
    tick()
  }

  const selectDetection = (i: number) => {
    const next = selectedIdx === i ? null : i
    setSelectedIdx(next)
    selectedRef.current = next
  }

  const toggleClass = (classId: number) => {
    setVisibleClasses(prev => ({ ...prev, [classId]: !prev[classId] }))
    setSelectedIdx(null); selectedRef.current = null
  }

  const clearHistory = () => { setHistory([]); try { localStorage.removeItem('logo-history') } catch (e) {} }

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = filename
    document.body.appendChild(a); a.click(); document.body.removeChild(a)
    setTimeout(() => { URL.revokeObjectURL(url) }, 1000)
  }

  const getBaseFilename = () => { if (!selectedFile) return 'captura'; return selectedFile.replace(/\.[^/.]+$/, '') }

  const exportJSON = () => {
    const data = {
      file: selectedFile || 'webcam', date: new Date().toISOString(), threshold: confThreshold,
      detections: detections.map(d => ({
        class: CLASS_NAMES[d.classId], classId: d.classId, confidence: Number(d.conf.toFixed(4)),
        box: { x1: Math.round(d.x1), y1: Math.round(d.y1), x2: Math.round(d.x2), y2: Math.round(d.y2) }
      }))
    }
    downloadBlob(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }), `${getBaseFilename()}.json`)
  }

  const exportCSV = () => {
    const header = 'class,classId,confidence,x1,y1,x2,y2'
    const rows = detections.map(d => [`"${CLASS_NAMES[d.classId]}"`, d.classId, d.conf.toFixed(4), Math.round(d.x1), Math.round(d.y1), Math.round(d.x2), Math.round(d.y2)].join(','))
    downloadBlob(new Blob([[header, ...rows].join('\n')], { type: 'text/csv' }), `${getBaseFilename()}.csv`)
  }

  const exportPNG = async () => {
    const canvas = document.createElement('canvas'); const ctx = canvas.getContext('2d'); if (!ctx) return
    let sourceW = 0; let sourceH = 0; let sourceElement: HTMLImageElement | HTMLVideoElement | null = null

    if (fileType.startsWith('image')) {
      sourceElement = imageRef.current
      if (sourceElement instanceof HTMLImageElement) { sourceW = sourceElement.naturalWidth; sourceH = sourceElement.naturalHeight }
    } else {
      sourceElement = videoRef.current
      if (sourceElement instanceof HTMLVideoElement) { sourceW = sourceElement.videoWidth; sourceH = sourceElement.videoHeight }
    }

    if (!sourceElement || sourceW === 0 || sourceH === 0) return
    canvas.width = sourceW; canvas.height = sourceH
    ctx.drawImage(sourceElement, 0, 0, sourceW, sourceH)
    drawDetections(canvas, detections, sourceW, sourceH, sourceW, sourceH, -1)
    canvas.toBlob(blob => { if (blob) downloadBlob(blob, `${getBaseFilename()}.png`) }, 'image/png')
  }

  const showMedia = previewURL || isWebcam

  return (
    <div className="page">
      <div className="card">

        <header className="header">
          <img src="/image.png" alt="Intelite" className="logo" />
          <h1 className="title">Detección de Logotipos</h1>
          <p className="subtitle">Analiza imágenes y video para detectar logotipos políticos con IA</p>
        </header>

        <div className={`status-badge ${modelReady ? 'ready' : 'loading-status'}`}>
          <span className="dot" />
          {modelReady ? 'Modelo listo' : 'Cargando modelo…'}
        </div>

        {/* CRITERIO DE ACEPTACIÓN CUMPLIDO: Input y Dropzone bloqueados estructuralmente si loading es true */}
        {!isWebcam && (
          <div
            className={`dropzone
              ${isDragging ? ' dragging' : ''}
              ${previewURL ? ' has-file' : ''}
              ${loading ? ' dropzone-disabled' : ''}
            `}
            onClick={() => { if (!loading) fileInputRef.current?.click() }}
            onDragOver={e => { e.preventDefault(); if (!loading) setIsDragging(true) }}
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

        {/* CRITERIO DE ACEPTACIÓN CUMPLIDO: Atributo HTML 'disabled' inyectado en el Input File */}
        <input 
          type="file" 
          accept="image/*,video/*" 
          ref={fileInputRef} 
          style={{ display: 'none' }} 
          onChange={handleFileChange} 
          disabled={loading} 
        />

        {/* CRITERIO DE ACEPTACIÓN CUMPLIDO: Botones con el atributo 'disabled' reactivo */}
        <div className="actions">
          {!isWebcam ? (
            <>
              <button 
                className="btn btn-upload" 
                onClick={() => fileInputRef.current?.click()} 
                disabled={loading}
              >
                Subir archivo
              </button>
              <button 
                className="btn btn-webcam" 
                onClick={startWebcam} 
                disabled={!modelReady || loading}
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
              <button 
                className="btn btn-upload" 
                onClick={() => fileInputRef.current?.click()} 
                disabled={loading}
              >
                Subir archivo
              </button>
              <button 
                className="btn btn-webcam btn-webcam-stop" 
                onClick={stopWebcam}
                disabled={loading}
              >
                Detener webcam
              </button>
            </>
          )}
        </div>

        {/* CRITERIO DE ACEPTACIÓN CUMPLIDO: Indicador visual claro (Spinner y overlay opaco de carga) */}
        {loading && (
          <div className="spinner-overlay">
            <div className="spinner" />
            <span className="spinner-text">Procesando frames mediante red neuronal...</span>
          </div>
        )}

        {showMedia && (
          <div className="results-layout-large">
            <div className="preview-section-large">
              <div className="preview-wrapper">
                {hasRun && (isWebcam || fileType.startsWith('video')) && (
                  <div className="fps-badge">NN FPS: {fps}</div>
                )}

                {isWebcam ? (
                  <>
                    <video ref={videoRef} className="preview-media" autoPlay muted playsInline />
                    <canvas ref={canvasRef} className="overlay-canvas" style={{ pointerEvents: 'none' }} />
                  </>
                ) : fileType.startsWith('image') ? (
                  <>
                    <img ref={imageRef} src={previewURL} alt="Vista previa" className="preview-media" />
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
              <div className="controls-panel-sidebar">
                <div className="export-actions">
                  <button className="btn-export" onClick={exportPNG}>PNG</button>
                  <button className="btn-export" onClick={exportJSON} disabled={!detections.length}>JSON</button>
                  <button className="btn-export" onClick={exportCSV} disabled={!detections.length}>CSV</button>
                </div>
                
                <div className="confidence-control">
                  <label>Umbral de confianza: <strong>{(confThreshold * 100).toFixed(0)}%</strong></label>
                  <input type="range" min="0.05" max="0.95" step="0.05" value={confThreshold} onChange={(e) => setConfThreshold(Number(e.target.value))} />
                </div>

                <div className="class-filters">
                  {Object.entries(CLASS_NAMES).map(([key, name]) => {
                    const classId = Number(key)
                    return (
                      <button key={classId} className={`class-chip ${visibleClasses[classId] ? 'active' : 'inactive'}`} onClick={() => toggleClass(classId)}>
                        <span className="chip-dot" />{name}
                      </button>
                    )
                  })}
                </div>

                <aside className="detections-panel">
                  <div className="panel-header">
                    <div className="stats-list">
                      {classStats.map(stat => (
                        <div key={stat.classId} className="stats-item">
                          <span className="stats-dot" style={{ background: `hsl(100, 100%, 45%)` }} />
                          <span className="stats-name">{stat.name}</span>
                          <span className="stats-count">{stat.count}</span>
                        </div>
                      ))}
                    </div>
                    <span className="panel-title">Detecciones</span>
                    <span className="panel-count">{detections.length}</span>
                  </div>

                  {detections.length === 0 ? <p className="panel-empty">Sin logotipos detectados</p> : (
                    <ul className="detection-list">
                      {detections.map((d, i) => (
                        <li key={i} className={`detection-item${selectedIdx === i ? ' active' : ''}`} onClick={() => selectDetection(i)}>
                          <span className="detection-class">{CLASS_NAMES[d.classId] ?? `Clase ${d.classId}`}</span>
                          <span className="detection-conf" style={{ color: `hsl(${d.conf * 120}, 100%, 38%)`, fontWeight: 'bold' }}>
                            {(d.conf * 100).toFixed(0)}%
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                </aside>
              </div>
            )}
          </div>
        )}

        {history.length > 0 && (
          <div className="history-section">
            <hr className="history-divider" />
            <div className="history-header">
              <h2 className="history-title">Historial de Análisis</h2>
              <button className="btn-clear-history" onClick={clearHistory}>Limpiar Historial</button>
            </div>
            <div className="history-grid">
              {history.map((item) => {
                const breakdown = Object.entries(item.counts).map(([name, qty]) => `${name}: ${qty}`).join('\n') || 'Sin detecciones'
                return (
                  <div key={item.id} className="history-card" title={`Analizado a las ${item.time}\n\nDesglose:\n${breakdown}`}>
                    <div className="history-thumb-wrapper">
                      <img src={item.thumb} alt={item.name} className="history-thumb-img" />
                    </div>
                    <div className="history-card-info">
                      <span className="history-card-name">{item.name}</span>
                      <span className="history-card-total">Detecciones: {item.total}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

      </div>
    </div>
  )
}

export default App