# Resultados del ejercicio 2 (corrección)

Generado con:

```bash
python3 -m ej2.vae_exp --dataset nba --fase barrido --trabajadores 5
python3 -m ej2.vae_exp --dataset nba --fase final --trabajadores 5
# idem f1, chino y banderas
python3 -m ej2.graficos
```

Los resultados anteriores a la corrección están archivados en `resultados/ej2_old/`.

## Por dataset

- `barrido/`: `paso_<id>.csv` con una fila por semilla y `paso_<id>.meta.json` con la huella
  de la configuración.
- `elecciones.json`: ganador de cada paso, `nivel_anticolapso`, `aceptado` y la validación.
- `final/`, `latente_4/`, `latente_16/`, `ae_latente_4/`, `ae_latente_16/`: `config.json`,
  `pesos_semilla_<s>.npz`, `historia_semilla_<s>.csv`, `muestras_semilla_<s>.npz`,
  `test.csv` y `metricas.json`.
- `figuras/`: las figuras de abajo.

## Figuras

| archivo | contenido |
|---|---|
| `00_resumen.png` | original, reconstrucción y muestra de `N(0, I)`, con nombres |
| `01_dataset.png` | muestra del dataset |
| `02_reconstruccion.png` | original vs reconstrucción en train y test |
| `03_latente_4.png` | μ del test y grilla decodificada del latente 4 |
| `04_latente_16_pca.png` | PCA de μ a 2D y grilla decodificada del latente 16 |
| `05_generacion.png` | muestras de `N(0, I)` y distancia al vecino más cercano de train |
| `06_curvas.png` | pérdidas, KL y unidades activas por época |
| `07_latente_4_vs_16.png` | reconstrucciones y muestras de latente 4 vs 16 |
| `08_banderas_decodificadas.png` | solo banderas: test, reconstrucción y muestras |
| `09_ae_vs_vae_muestras.png` | el mismo z ~ N(0, I) por el decoder del AE y del VAE |
| `10_distribucion_latente.png` | μ de train por dimensión contra N(0, 1), AE y VAE |
| `11_interpolacion.png` | interpolación entre dos códigos de test, AE y VAE |
| `metricas_generacion.csv` | calidad de las muestras de N(0, I) por modelo (5 semillas) |

En banderas, `figuras/auxiliares/` guarda los PNG que `utils/generar_banderas.py` renderiza
para `00_resumen.png` (originales, reconstrucciones y muestras con la misma geometría).
