import pandas as pd

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
