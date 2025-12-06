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
│   ├── Books.jsonl.gz         
│   ├── df_raw_sample.pkl      
│   ├── train.pkl              
│   └── test.pkl               
│
├── results/
│   ├── rewards_eg.pkl         
│   ├── rewards_ucb.pkl        
│   └── rewards_ts.pkl         
│
├── src/
│   └── backend/
│       ├── recommender.py     
│       └── api.py             
│
├── notebooks/
│   ├── 01_EDA.ipynb           
│   ├── 02_Preprocess.ipynb    
│   ├── 03_Modelling.ipynb     
│   ├── 04_Bandits.ipynb       
│   └── 05_Comparacion_Final.ipynb
│
├── requirements.txt           
└── README.md
```

---

## Fase 1 – Carga del Dataset

✔ *Los notebooks realizan esta fase.*  

---

## Fase 2 – Preprocesamiento del Dataset  
Aquí comienza a intervenir **`recommender.py`**.

Incluye funciones como:  
- load_amazon_books_sample()  
- clean_data()  
- encode_ids()  
- split_data()  
- save_serialized()

Ejecutadas mediante:  
```
recommender.preprocess_pipeline()
```

---

## Fase 3 – Construcción del Modelo ALS

`recommender.py` implementa:  
- create_user_item_matrix()  
- train_model()  
- save_model()  
- load_model()  

Se usa para entrenar y guardar el modelo ALS.

---

## Fase 4 – Entrenamiento de Agentes Bandits

✔ *Realizado solo en notebooks.*  
No usa recommender.py ni la API.

---

## Fase 5 – Comparación de Resultados

✔ *Solo notebooks.*  

---

## Fase 6 – API REST con FastAPI

Aquí entra **`api.py`**, que carga el modelo ALS y expone endpoints REST.

Endpoints:  
- GET /recommendations/{user_id}  
- GET /history/{user_id}  
- GET /users  

Comandos:  
```
uvicorn api:app --reload
```

---

## Resumen de Roles

| Archivo | Rol |
|--------|-----|
| **recommender.py** | Preprocesamiento, ALS, generación de recomendaciones |
| **api.py** | API REST sobre el modelo ALS |
| Notebooks | EDA, procesamiento, modelado, bandits, comparación |

---

## Conclusión

El proyecto combina un modelo ALS para recomendaciones con agentes Multi-Armed Bandits como método experimental y una API funcional para despliegue del sistema.
