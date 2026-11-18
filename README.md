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

## Web service

La API REST se encuentra desplegada en Render:

**URL base:** https://examen-1er-bimestre-ri.onrender.com

### Comprobar disponibilidad

```http
GET /health
```

Prueba con PowerShell:

```powershell
curl.exe "https://examen-1er-bimestre-ri.onrender.com/health"
```

Respuesta esperada:

```json
{
  "status": "ok",
  "documents_indexed": 5183
}
```

### Realizar una búsqueda con BM25

```powershell
curl.exe --% -X POST https://examen-1er-bimestre-ri.onrender.com/search -H "Content-Type: application/json" --data-raw "{\"query\":\"vitamin D cancer risk\",\"model\":\"bm25\",\"top_k\":10}"
```

### Realizar una búsqueda con TF-IDF

```powershell
curl.exe --% -X POST https://examen-1er-bimestre-ri.onrender.com/search -H "Content-Type: application/json" --data-raw "{\"query\":\"vitamin D cancer risk\",\"model\":\"tfidf\",\"top_k\":10}"
```

### Formato de la solicitud

```http
POST /search
Content-Type: application/json
```

```json
{
  "query": "texto de la consulta",
  "model": "bm25",
  "top_k": 10
}
```

El campo `model` acepta `bm25` y `tfidf`. El valor de `top_k` debe estar entre 1 y 100.

Cada resultado incluye:

- Posición en el ranking.
- Identificador original del documento.
- Puntuación de relevancia.
- Título.
- Fragmento del documento.

> **Nota:** el servicio utiliza una instancia gratuita de Render. Después de un periodo de inactividad puede suspenderse temporalmente, por lo que la primera solicitud podría tardar aproximadamente un minuto mientras vuelve a iniciarse.

## Archivos

- `api.py`: endpoints `GET /health` y `POST /search`, validacion y formato de respuesta.
- `src/preprocess.py`: normalizacion, tokenizacion, stopwords y stemming.
- `src/inverted_index.py`: vocabulario, postings, longitudes y normas TF-IDF.
- `src/models.py`: TF-IDF con coseno y BM25.
- `src/evaluate.py`: Precision@10, Recall@10 y NDCG@10.
- `scripts/prepare_data.py`: descarga corpus, consultas y qrels test sin alterar IDs.
- `scripts/build_index.py`: crea el indice y `artifacts/metrics.json`.
- `examen1bim.ipynb`: explicacion y resultados reproducibles.

