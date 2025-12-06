from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
sys.path.append("../src/backend")

from recommender import BookRecommender, init_model

app = FastAPI(title="Book Recommender API")

# Variable global para el modelo
recommender = None
MODEL_PATH = "../results/als_model.pkl"
DATA_PATH = "../data/"

@app.on_event("startup")
async def startup_event():
    """Cargar modelo al iniciar la API"""
    global recommender
    recommender = init_model(data_path=DATA_PATH, model_path=MODEL_PATH)

@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: int, top_k: int = 10):
    """Obtener recomendaciones para un usuario"""
    recommendations, status = recommender.get_recommendations(user_id, top_k)
    
    if recommendations is None:
        raise HTTPException(status_code=404, detail=status)
    
    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "status": status
    }

@app.get("/history/{user_id}")
async def get_history(user_id: int, top_k: int = 10):
    """Obtener historial de un usuario"""
    history, status = recommender.get_user_history(user_id, top_k)
    
    if history is None:
        raise HTTPException(status_code=404, detail=status)
    
    return {
        "user_id": user_id,
        "history": history,
        "status": status
    }

@app.get("/users")
async def get_users():
    """Obtener lista de usuarios disponibles"""
    users = recommender.get_all_users()
    return {"total_users": len(users), "users": users[:100]}