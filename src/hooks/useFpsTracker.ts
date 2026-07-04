import { useState, useCallback, useRef } from 'react'

interface UseFpsTrackerResult {
  fps: number
  trackInference: () => void
  resetFps: () => void
}

/**
 * Custom Hook para rastrear el rendimiento (FPS) de la inferencia
 * utilizando un Promedio Móvil Exponencial (EMA) para suavizar lecturas.
 */
export function useFpsTracker(updateIntervalMs = 300, alpha = 0.2): UseFpsTrackerResult {
  const [fps, setFps] = useState<number>(0)
  
  const lastInferenceTimeRef = useRef<number>(0)
  const lastFpsUpdateRef = useRef<number>(0)
  const fpsEmaRef = useRef<number>(0)

  /**
   * Registra una nueva marca de inferencia y recalcula el FPS actual.
   */
  const trackInference = useCallback(() => {
    const now = performance.now()

    // Primer frame o reinicio reciente
    if (lastInferenceTimeRef.current === 0) {
      lastInferenceTimeRef.current = now
      lastFpsUpdateRef.current = now
      return
    }

    const delta = now - lastInferenceTimeRef.current
    lastInferenceTimeRef.current = now

    // Evitar divisiones por cero si los frames llegan demasiado rápido
    if (delta <= 0) return

    const currentFps = 1000 / delta

    // Aplicar fórmula EMA: S_t = α * Y_t + (1 - α) * S_{t-1}
    if (fpsEmaRef.current === 0) {
      fpsEmaRef.current = currentFps
    } else {
      fpsEmaRef.current = (alpha * currentFps) + ((1 - alpha) * fpsEmaRef.current)
    }

    // Actualizar el estado de React de forma periódica para mitigar repaints excesivos
    if (now - lastFpsUpdateRef.current > updateIntervalMs) {
      setFps(Math.round(fpsEmaRef.current))
      lastFpsUpdateRef.current = now
    }
  }, [alpha, updateIntervalMs])

  /**
   * Restablece los contadores internos a su estado inicial en seco.
   */
  const resetFps = useCallback(() => {
    setFps(0)
    fpsEmaRef.current = 0
    lastInferenceTimeRef.current = 0
    lastFpsUpdateRef.current = 0
  }, [])

  return { fps, trackInference, resetFps }
}