# Examen de Recuperacion de Informacion

El corpus asignado y configurado es **SciFact** (5.183 documentos)

## Instalacion y construccion

Desde esta carpeta, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\prepare_data.py --corpus scifact
python scripts\build_index.py
python -m unittest discover -s tests -v
```

## Servicio local (no publica nada en internet)

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Pruebas manuales en otra terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health

$body = @{ query = 'vitamin D cancer risk'; model = 'bm25'; top_k = 10 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/search -Method Post -ContentType 'application/json' -Body $body
```

Cambiar `bm25` por `tfidf` prueba el segundo ranking. La documentacion interactiva queda disponible solo en el equipo en `http://127.0.0.1:8000/docs`.

## Archivos

- `api.py`: endpoints `GET /health` y `POST /search`, validacion y formato de respuesta.
- `src/preprocess.py`: normalizacion, tokenizacion, stopwords y stemming.
- `src/inverted_index.py`: vocabulario, postings, longitudes y normas TF-IDF.
- `src/models.py`: TF-IDF con coseno y BM25.
- `src/evaluate.py`: Precision@10, Recall@10 y NDCG@10.
- `scripts/prepare_data.py`: descarga corpus, consultas y qrels test sin alterar IDs.
- `scripts/build_index.py`: crea el indice y `artifacts/metrics.json`.
- `examen1bim.ipynb`: explicacion y resultados reproducibles.

