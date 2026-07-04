import { renderHook, act } from '@testing-library/react'
import { useFpsTracker } from './useFpsTracker'
import { describe, it, expect, beforeEach, vi } from 'vitest'

describe('useFpsTracker() Custom Hook', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    // Mockear performance.now() comenzando en un punto base fijo
    let currentTime = 1000
    vi.spyOn(performance, 'now').mockImplementation(() => currentTime)
    
    // Función auxiliar para simular el paso del tiempo en ms
    global.advanceTime = (ms: number) => {
      currentTime += ms
    }
  })

  it('debe inicializar en 0 FPS', () => {
    const { result } = renderHook(() => useFpsTracker())
    expect(result.current.fps).toBe(0)
  })

  it('debe acumular frames y actualizar el FPS real tras superar el intervalo', () => {
    const { result } = renderHook(() => useFpsTracker(300, 0.2))

    // Primer frame (inicializa marcas de tiempo)
    act(() => { result.current.trackInference() })
    
    // Simular 10 frames consecutivos con un intervalo constante de 33.33ms (~30 FPS)
    for (let i = 0; i < 10; i++) {
      act(() => {
        global.advanceTime(33.33) // delta constante
        result.current.trackInference()
      })
    }

    // El tiempo acumulado (333.3ms) ya superó los 300ms de refresco
    expect(result.current.fps).toBeCloseTo(30, 0)
  })

  it('debe limpiar los registros a 0 al invocar resetFps()', () => {
    const { result } = renderHook(() => useFpsTracker())

    act(() => { result.current.trackInference() })
    act(() => {
      global.advanceTime(33.33)
      result.current.trackInference()
    })

    act(() => { result.current.resetFps() })
    expect(result.current.fps).toBe(0)
  })
})