import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from implicit.als import AlternatingLeastSquares
import pickle
import os

class BookRecommender:
    
    def __init__(self, data_path="../data/"):
        self.data_path = data_path
        self.model = None
        self.train_matrix = None
        self.user_to_idx = None
        self.item_to_idx = None
        self.train_df = None
        self.test_df = None
        self.df_sample = None
        self.df_cleaned = None
        self.df_result = None
        self.user_encoder = None
        self.item_encoder = None
        
    def load_amazon_books_sample(self, path="../data/Books.jsonl.gz", sample_size=100000, chunk_size=50000):
        """Carga solo una muestra del dataset sin leerlo completo."""
        sampled_rows = []
        total_read = 0

        for chunk in pd.read_json(path, lines=True, compression="gzip", chunksize=chunk_size):
            total_read += len(chunk)
            frac = sample_size / total_read
            if frac <= 0:
                break
            sampled_chunk = chunk.sample(frac=min(1, frac), replace=False, random_state=42)
            sampled_rows.append(sampled_chunk)
            if sum(len(c) for c in sampled_rows) >= sample_size:
                break

        self.df_sample = pd.concat(sampled_rows, ignore_index=True)
        if len(self.df_sample) > sample_size:
            self.df_sample = self.df_sample.sample(sample_size, random_state=42)
        print(f"Muestra cargada con {len(self.df_sample)} filas.")

    def clean_data(self):
        """Limpieza básica: eliminar duplicados, convertir timestamp."""
        self.df_cleaned = self.df_sample.drop_duplicates(subset=["user_id", "asin", "timestamp"])
        self.df_cleaned.loc[:, "timestamp"] = pd.to_datetime(self.df_cleaned["timestamp"], unit="ms", errors="coerce")
        print(f"Datos limpiados: {len(self.df_cleaned)} filas restantes.")

    def encode_ids(self):
        """Convierte user_id y asin a IDs numéricos."""
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.df_cleaned.loc[:, "user"] = self.user_encoder.fit_transform(self.df_cleaned["user_id"])
        self.df_cleaned.loc[:, "item"] = self.item_encoder.fit_transform(self.df_cleaned["asin"])
        print(f"Usuarios únicos: {len(self.user_encoder.classes_)}")
        print(f"Ítems únicos: {len(self.item_encoder.classes_)}")

    def build_model_dataset(self):
        """Extrae columnas necesarias para el modelo."""
        self.df_result = self.df_cleaned[["user", "item", "rating"]].copy()

    def split_data(self):
        """Separa train/test."""
        self.train_df, self.test_df = train_test_split(
            self.df_result,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )
        print(f"Datos divididos: {len(self.train_df)} en train, {len(self.test_df)} en test.")

    def save_serialized(self, path="../../data/"):
        """Guarda train/test/df_model como pickles."""
        os.makedirs(path, exist_ok=True)
        self.train_df.to_pickle(os.path.join(path, "train.pkl"))
        self.test_df.to_pickle(os.path.join(path, "test.pkl"))
        self.df_result.to_pickle(os.path.join(path, "df_model.pkl"))
        print(f"Datos guardados en {path}")

    def create_user_item_matrix(self):
        """Crea matriz sparse usuario-item para ALS."""
        users = sorted(self.train_df['user'].unique())
        items = sorted(self.train_df['item'].unique())
        
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {it: i for i, it in enumerate(items)}
        
        rows = [self.user_to_idx[u] for u in self.train_df['user']]
        cols = [self.item_to_idx[it] for it in self.train_df['item']]
        data = self.train_df['rating'].values
        
        self.train_matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        print(f"Matriz creada: {self.train_matrix.shape}")

    def train_model(self, factors=50, regularization=0.1, iterations=20):
        """Entrena modelo ALS."""
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42
        )
        self.model.fit(self.train_matrix)
        print(f"Modelo entrenado: {factors} factores, {iterations} iteraciones")

    def preprocess_pipeline(self, path="../../data/"):
        """Pipeline completo: carga, limpia, encoda, split."""
        self.load_amazon_books_sample()
        self.clean_data()
        self.encode_ids()
        self.build_model_dataset()
        self.split_data()
        self.save_serialized(path=path)
        print("Pipeline completado!")

    def save_model(self, filepath):
        """Guarda modelo entrenado con encoders y matriz."""
        model_data = {
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'user_factors': self.model.user_factors,
            'item_factors': self.model.item_factors,
            'train_matrix': self.train_matrix,
            'model_params': {
                'factors': self.model.factors,
                'regularization': self.model.regularization,
                'iterations': self.model.iterations
            }
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Modelo guardado en {filepath}")

    def load_model(self, filepath):
        """Carga modelo entrenado."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_encoder = model_data['user_encoder']
        self.item_encoder = model_data['item_encoder']
        self.user_to_idx = model_data['user_to_idx']
        self.item_to_idx = model_data['item_to_idx']
        self.train_matrix = model_data['train_matrix']
        
        params = model_data['model_params']
        self.model = AlternatingLeastSquares(
            factors=params['factors'],
            regularization=params['regularization'],
            iterations=params['iterations'],
            random_state=42
        )
        self.model.user_factors = model_data['user_factors']
        self.model.item_factors = model_data['item_factors']
        print(f"Modelo cargado desde {filepath}")
        
    def get_recommendations(self, user_id, top_k=10):
        """Generar recomendaciones para un usuario"""
        if user_id not in self.user_to_idx:
            return None, "Usuario no encontrado"
        
        user_idx = self.user_to_idx[user_id]
        
        # Obtener items ya vistos
        user_items = set(self.train_df[self.train_df['user'] == user_id]['item'].values)
        
        # Calcular scores
        all_items = list(self.item_to_idx.keys())
        recommendations = []
        
        for item in all_items:
            if item not in user_items:
                item_idx = self.item_to_idx[item]
                score = self.model.user_factors[user_idx] @ self.model.item_factors[item_idx]
                recommendations.append({'item': int(item), 'score': float(score)})
        
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:top_k]
        return recommendations, "OK"

    def get_user_history(self, user_id, top_k=10):
        """Obtener historial de un usuario"""
        if user_id not in self.user_to_idx:
            return None, "Usuario no encontrado"
        
        user_data = self.train_df[self.train_df['user'] == user_id]
        user_data = user_data.sort_values('rating', ascending=False).head(top_k)
        
        history = []
        for _, row in user_data.iterrows():
            history.append({'item': int(row['item']), 'rating': float(row['rating'])})
        
        return history, "OK"

    def get_all_users(self):
        """Obtener lista de usuarios disponibles"""
        return sorted([int(u) for u in self.user_to_idx.keys()])


def init_model(data_path="../data/", model_path="../results/als_model.pkl"):
    """Inicializa o carga modelo."""
    recommender = BookRecommender(data_path=data_path)
    
    if os.path.exists(model_path):
        print("Cargando modelo existente...")
        recommender.load_model(model_path)
    else:
        print("Entrenando nuevo modelo...")
        recommender.preprocess_pipeline(path=data_path)
        recommender.create_user_item_matrix()
        recommender.train_model(factors=50, regularization=0.1, iterations=20)
        recommender.save_model(model_path)
    
    print("Modelo listo!")
    return recommender