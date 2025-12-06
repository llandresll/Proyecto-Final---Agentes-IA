# Sistema de Recomendación con Multi-Armed Bandits  
### Proyecto Final — IA Engineer

Este proyecto implementa un **Sistema de Recomendación basado en Multi-Armed Bandits** utilizando una muestra del dataset de reseñas de **Amazon Books (2023)**.  
El objetivo es construir un recomendador que aprenda de manera secuencial y compare diferentes estrategias de exploración/explotación.

---

## Dataset

Se empleó el dataset público:

**Amazon Reviews 2023 – Books**  
Fuente: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz

Formato original: `.jsonl.gz` (~6.22 GB)

Para este proyecto se utilizó una **muestra de 100 000 registros**, seleccionada mediante lectura por chunks para evitar uso excesivo de memoria.

---

## Estructura del Proyecto

```
ProyectoFinal/
│
├── data/
│   ├── Books.jsonl.gz         # Dataset original
│   ├── df_raw_sample.pkl      # Subset crudo (100k)
│   ├── train.pkl              # Datos para entrenamiento
│   └── test.pkl               # Datos para test
│
├── results/
│   ├── rewards_eg.pkl         # Rewards Epsilon-Greedy
│   ├── rewards_ucb.pkl        # Rewards UCB1
│   └── rewards_ts.pkl         # Rewards Thompson Sampling
│
├── src/
│ └── backend/
│ ├── recommender.py # Clase BookRecommender + ALS
│ └── api.py # API REST con FastAPI
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Análisis exploratorio
│   ├── 02_Preprocess.ipynb    # Procesamiento y división Train/Test
│   ├── 03_Modelling.ipynb     # Modelos clásicos (baseline supervisado)
│   ├── 04_Bandits.ipynb       # Implementación y entrenamiento de agentes
│   └── 05_Comparacion_Final.ipynb   # Gráficos y conclusiones
│
├── requirements.txt # Dependencias Python
└── README.md
```

---

## Fase 1 – Carga del Dataset

Para evitar cargar todo el archivo completo (que supera los 6GB), se implementó:

```python
def load_amazon_books_sample(path="../data/Books.jsonl.gz", sample_size=100000, chunk_size=50000):
    """Carga solo una muestra del dataset sin leerlo completo."""
    sampled_rows = []
    total_read = 0

    for chunk in pd.read_json(
        path,
        lines=True,
        compression="gzip",
        chunksize=chunk_size
    ):
        total_read += len(chunk)
        frac = sample_size / total_read

        if frac <= 0:
            break

        sampled_chunk = chunk.sample(
            frac=min(1, frac),
            replace=False,
            random_state=42
        )
        sampled_rows.append(sampled_chunk)

        if sum(len(c) for c in sampled_rows) >= sample_size:
            break

    df_sample = pd.concat(sampled_rows, ignore_index=True)

    if len(df_sample) > sample_size:
        df_sample = df_sample.sample(sample_size, random_state=42)

    return df_sample
```

Salida:

- 100 000 registros
- Columnas: rating, title, text, asin, user_id, timestamp, etc.

El dataframe resultante se guarda como `df_100k.pkl`.

---

## Fase 2 – Preprocesamiento

Incluye:

- Conversión de timestamps  
- Eliminación de columnas no necesarias  
- Codificación numérica de:
  - `user_id → user`
  - `asin → item`
- Conversión de ratings de 1–5 a recompensas numéricas  
- División en Train/Test (80/20)

El resultado se almacena como:

- `train.pkl`
- `test.pkl`

---

## Fase 3 – Implementación de Agentes Bandits

Se implementaron **tres estrategias desde cero**:

### ✔ Epsilon-Greedy (ε = 0.1)  
### ✔ UCB1  
### ✔ Thompson Sampling (Beta-Bernoulli)

Cada agente interactúa con un entorno que simula el comportamiento de un usuario real mediante muestras desde `train`.

Incluye:

- Clase del entorno `UserBanditEnv`
- Implementación de `select_item()` y `update()`
- Manejo interno de valores estimados, conteos, probabilidades Beta, etc.

---

## Fase 4 – Entrenamiento de Bandits

Cada agente ejecuta **5000 episodios**, donde en cada episodio:

1. Se selecciona un usuario aleatorio  
2. Se toma una interacción real de ese usuario  
3. El agente elige un ítem  
4. Se obtiene reward  
5. Se actualiza el modelo  

Se guardan los rewards:

```
/results/rewards_eg.pkl
/results/rewards_ucb.pkl
/results/rewards_ts.pkl
```

---

## Fase 5 – Comparación de Resultados

Se cargan los rewards guardados y se generan:

- Reward por episodio  
- Reward acumulado  
- Tabla comparativa  
- Agente ganador  
- Gráficos finales  

Código:

```python
rewards_eg = pd.read_pickle("../results/rewards_eg.pkl")
rewards_ucb = pd.read_pickle("../results/rewards_ucb.pkl")
rewards_ts = pd.read_pickle("../results/rewards_ts.pkl")
```

---

## Fase 6 – Conclusiones

### **Thompson Sampling fue el mejor agente**
- Mayor reward acumulado  
- Mayor estabilidad  
- Mejor balance exploración/explotación  

### Comparación general

| Agente | Exploración | Estabilidad | Resultado |
|--------|-------------|-------------|-----------|
| Epsilon-Greedy | Alta aleatoria | Baja | Peor desempeño |
| UCB1 | Balanceado | Alto | Segundo lugar |
| Thompson Sampling | Probabilística óptima | Muy alto | **Ganador** |

---

## Conclusión Global

Este sistema demuestra que **Multi-Armed Bandits son un enfoque eficiente, simple y escalable para sistemas de recomendación**, especialmente en escenarios con alta incertidumbre o interacción en línea.  
**Thompson Sampling** es la mejor estrategia para este proyecto y se recomienda como modelo principal de despliegue.

---

## Autor

**Andrés Rivadeneyra**  
Proyecto Final — IA Engineer  
2025
