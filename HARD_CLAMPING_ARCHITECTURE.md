# Arquitectura de Hard Clamping y Coexistencia con ComfyUI Core

Este documento detalla la fundamentación matemática, el diseño de ingeniería y el mecanismo de integración no invasivo para el **Hard Clamping** en MiniMax H3 dentro de ComfyUI.

---

## 1. Fundamentos Matemáticos: Flow Matching y Clamping

MiniMax H3 opera sobre un espacio latente multimodal continuo en Flow Matching (Rectified Flow):
* **Flujo de Video:** $x^{(v)} \in \mathbb{R}^{B \times 24 \times T_v \times H/16 \times W/16}$
* **Flujo de Audio:** $x^{(a)} \in \mathbb{R}^{B \times 32 \times 2 \times T_a}$

### 1.1 Ecuación de Difusión (Rectified Flow)
La trayectoria lineal entre ruido $\epsilon \sim \mathcal{N}(0, I)$ en $\sigma=1$ y dato limpio $x_1$ en $\sigma=0$ es:
$$x_t = (1 - \sigma_t) x_1 + \sigma_t \epsilon$$

La red DiT predice el campo de velocidades $v_t = \frac{dx_t}{dt}$. El estimador del dato limpio predicho $\hat{x}_0$ (*denoised*) en el paso actual con ruido $\sigma_t$ es:
$$\hat{x}_0 = x_t - \sigma_t \cdot v_t$$

Y la velocidad correspondiente que el sampler ODE integra es:
$$v_t = \frac{x_t - \hat{x}_0}{\sigma_t}$$

### 1.2 Mecanismo de Hard Clamping
Si en un intervalo temporal específico $[t_{start}, t_{end}]$ deseamos fijar un clip o fotograma exacto al latente objetivo $x_{target}$:

1. **En el espacio de velocidad ($v$):**
   $$v_{clamp}[t_{start}:t_{end}] = \frac{x_t[t_{start}:t_{end}] - x_{target}}{\sigma_t}$$

2. **En el espacio de predicción limpia ($\hat{x}_0$ / *denoised*):**
   $$\hat{x}_0[t_{start}:t_{end}] = x_{target}$$

3. **Evolución del paso ODE (Integrador de Euler):**
   $$x_{next} = x_t + (\sigma_{next} - \sigma_t) \cdot \frac{x_t - x_{target}}{\sigma_t} = \frac{\sigma_{next}}{\sigma_t} x_t + \left(1 - \frac{\sigma_{next}}{\sigma_t}\right) x_{target}$$

### 1.3 Propiedades Teóricas
* **Cero Deriva:** Cuando $\sigma \to 0$ al final del muestreo, $x_{final} = x_{target}$ exactamente (error flotante 0).
* **Consistencia Contextual:** En los pasos intermedios $\sigma_t \in (0, 1)$, el latente en los fotogramas fijados toma el valor exacto $x_t = (1-\sigma_t)x_{target} + \sigma_t \epsilon$. La atención espacio-temporal bidireccional del DiT observa esta trayectoria exacta en su ventana de contexto, permitiendo que todos los fotogramas adyacentes (huecos y transiciones) sinteticen movimiento, iluminación y audio perfectamente coherentes.

---

## 2. Diferencias: Soft Guides vs. Hard Clamping

| Característica | Core Nativo (`minimax_keyframes`) | Hard Clamping (`_clamp_hard`) | Arquitectura Híbrida Coexistente |
| :--- | :--- | :--- | :--- |
| **Mecanismo** | Tokens de referencia en `PackedLayout` del DiT | Sobreescritura de velocidad / $\hat{x}_0$ en el Sampler | **Ambos simultáneos** |
| **Atención del DiT** | El modelo atiende a los tokens de guía | El modelo atiende a los latentes en el canvas | **Doble guía:** tokens de atención + canvas fijado |
| **Preservación de Pixeles** | Aproximada (sujeta a re-imaginación del DiT) | Exacta (cero pérdida ni desviación de color/rasgos) | **Exacta 100% en zonas fijadas** |
| **Transiciones / Gaps** | Suaves pero el clip original puede derivar | Guiadas por la trayectoria del canvas | **Transiciones fluidas hacia clips 100% idénticos** |
| **Invasividad en Core** | Nativo (no requiere parches) | Requería monkey-patch destructivo | **No invasivo vía `ModelPatcher` hooks** |

---

## 3. Arquitectura de Coexistencia con ComfyUI Core

Para evitar monkey-patching destructivo en `comfy.ldm.minimax.model` o `comfy.model_base`, aprovechamos el sistema oficial de hooks de ComfyUI:

### 3.1 Intercepción vía `ModelPatcher`
ComfyUI proporciona callbacks limpios por instancia de modelo:
1. `model.clone()`: Clona el patcher sin alterar el modelo base ni otras instancias en memoria.
2. `set_model_sampler_post_cfg_function(hook_fn)`: Registra un callback que se ejecuta tras el cálculo de CFG en cada paso de muestreo en `comfy/samplers.py`:
   ```python
   def clamp_hook(args):
       # args = {"denoised": x0, "input": xt, "sigma": sigma, "model_options": ...}
       denoised = args["denoised"]
       # Aplicar reemplazo temporal a denoised para video y audio
       return clamped_denoised
   ```

### 3.2 Manejo Multimodal (Desempaquetado / Reempaquetado)
En MiniMax H3, `denoised` puede ser:
* Un `NestedTensor` de `[video_tensor, audio_tensor]`.
* Un tensor empaquetado plano 1D/3D generado por `comfy.utils.pack_latents`.

El hook procesa ambos casos de forma transparente:
```python
if hasattr(denoised, "is_nested") and denoised.is_nested:
    streams = denoised.unbind()
    v, a = streams[0], (streams[1] if len(streams) > 1 else None)
    # Aplicar clamp sobre v y a
    return comfy.nested_tensor.NestedTensor([v, a] if a is not None else [v])
elif latent_shapes is not None and len(latent_shapes) > 1:
    unpacked = comfy.utils.unpack_latents(denoised, latent_shapes)
    v, a = unpacked[0], unpacked[1]
    # Aplicar clamp sobre v y a
    packed, _ = comfy.utils.pack_latents([v, a])
    return packed
else:
    # Solo video [B, C, T, H, W]
    # Aplicar clamp sobre video
    return denoised
```

### 3.3 Mapeo Temporal (Pixel Frames ↔ Latent Tokens)
* **Video:** Cadencia de tokens `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`.
  * Latente temporal index $k=0 \implies$ frame 0.
  * Latente temporal index $k=1 \implies$ frames 1..4, etc.
* **Audio:** Factor de escala `FRAME_RESCALE = 5.0 / 3.0` respecto al número de frames de video.

---

## 4. Diseño del Nodo en ComfyUI

### Entradas y Salidas
* **Entrada Opcional:** `model: ("MODEL",)` (opcional).
* **Entrada Principal:** `conditioning`, `latent`, `timeline_state`, `video vae`, `audio vae`.
* **Salida Principal:** `CONDITIONING` (con `minimax_keyframes` nativo para atención DiT).
* **Salida Opcional:** `MODEL` (con el hook `sampler_post_cfg_function` acoplado al `ModelPatcher` clonado).

### Flujo de Ejecución
1. El usuario arma su línea de tiempo en el widget UI (imágenes, videos, audios en posiciones específicas).
2. `MiniMaxH3Timeline.apply(...)`:
   * Codifica los clips a latentes con `video vae` y `audio vae`.
   * Construye los `minimax_keyframes` nativos para el `CONDITIONING`.
   * Si `model` está conectado: crea una lista de especificaciones de clamp `[{stream: "video", start_idx, end_idx, latent, strength}, {stream: "audio", ...}]`, clona `model` y le adosa el `sampler_post_cfg_function`.
3. Al ejecutar `KSampler`:
   * El DiT recibe `minimax_keyframes` a través de ComfyUI Core.
   * El sampler ejecuta el `sampler_post_cfg_function` en cada paso, forzando la convergencia exacta en las pistas seleccionadas.

---

## 5. Ventajas de Esta Solución
1. **100% Compatible con el Core:** No modifica ningún archivo ni clase global de ComfyUI.
2. **Coexistencia Total:** El usuario puede usar solo conditioning, solo clamping, o ambos en paralelo.
3. **Mantenibilidad:** Si ComfyUI actualiza su código de MiniMax H3, este custom node sigue funcionando sin modificaciones.
4. **Rendimiento:** Cero sobrecarga de re-empaquetado innecesario; slicing directo sobre tensores PyTorch en GPU.
