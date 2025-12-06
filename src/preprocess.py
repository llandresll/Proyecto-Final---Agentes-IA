import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. Limpieza de datos
# ---------------------------------------------------------

def clean_data(df):
    """
    Limpieza básica:
    - Eliminar duplicados
    - Convertir timestamp
    """
    df = df.drop_duplicates(subset=["user_id", "asin", "timestamp"])
    df.loc[:, "timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    return df


# ---------------------------------------------------------
# 2. Encoding de IDs
# ---------------------------------------------------------

def encode_ids(df):
    """
    Convierte user_id y asin a IDs numéricos para el modelo.
    """
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    df.loc[:, "user"] = user_enc.fit_transform(df["user_id"])
    df.loc[:, "item"] = item_enc.fit_transform(df["asin"])

    return df, user_enc, item_enc


# ---------------------------------------------------------
# 3. Crear dataset para el modelo
# ---------------------------------------------------------

def build_model_dataset(df):
    """
    Devuelve solo columnas necesarias para modelos de recomendación.
    """
    return df[["user", "item", "rating"]].copy()


# ---------------------------------------------------------
# 4. Split de entrenamiento y prueba
# ---------------------------------------------------------

def split_data(df_model):
    """
    Separa train/test.
    """
    train, test = train_test_split(
        df_model,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    return train, test


# ---------------------------------------------------------
# 5. Guardar archivos procesados
# ---------------------------------------------------------

def save_serialized(train, test, df_model, path="../data/"):
    """
    Guarda los datasets finales.
    """
    train.to_pickle(path + "train.pkl")
    test.to_pickle(path + "test.pkl")
    df_model.to_pickle(path + "df_model.pkl")


# ---------------------------------------------------------
# 6. Pipeline completo
# ---------------------------------------------------------

def preprocess_pipeline(df):
    """
    Pipeline completo del procesamiento:
    - Copia del df para evitar SettingWithCopyWarning
    - Limpieza
    - Encoding
    - Construcción dataset del modelo
    - Split train/test
    """
    df = df.copy()  # ← Solución real a tu warning

    df = clean_data(df)
    df, user_enc, item_enc = encode_ids(df)

    df_model = build_model_dataset(df)
    train, test = split_data(df_model)

    return train, test, df_model, user_enc, item_enc
