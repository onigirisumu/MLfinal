from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer, util

# --------------------------------------------------------------------
# Paths (relative to this file)
# --------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent         
MODELS_DIR = BASE_DIR / "models"                     
STATIC_DIR = BASE_DIR / "static"                     
FRONT_DIR  = BASE_DIR.parent / "frontend"            

# --------------------------------------------------------------------
# FastAPI instance
# --------------------------------------------------------------------
app = FastAPI(title="CineMatch – Movie Recommender")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --------------------------------------------------------------------
# Load artefacts once on startup
# --------------------------------------------------------------------
print("Loading artefacts …")
MOVIES_DF: pd.DataFrame = pd.read_pickle(MODELS_DIR / "movies_data_light.pkl")
EMB_MATRIX: np.ndarray = np.load(MODELS_DIR / "movie_embeddings.npy")
MLB                           = joblib.load(MODELS_DIR / "multi_label_binarizer.pkl")
CLASSIFIER                    = joblib.load(MODELS_DIR / "multi_label_model.pkl")
EMBEDDER                      = SentenceTransformer("all-MiniLM-L6-v2")
print("Artefacts loaded")

# Convert stored embeddings to a single 2-D float32 array for speed
EMB_MATRIX = np.vstack(MOVIES_DF["embedding"].values).astype(np.float32)

class UserInput(BaseModel):
    overview: str

def get_recommendations(text: str, top_k: int = 5, prob_thresh: float = 0.25):
    # Embed user text
    user_vec = EMBEDDER.encode([text]).astype(np.float32)

    # Predict genres
    probs   = CLASSIFIER.predict_proba(user_vec)[0]
    binary  = (probs >= prob_thresh).astype(int)
    if binary.sum() == 0:
        binary[np.argmax(probs)] = 1  # ensure at least one genre
    pred_genres = list(MLB.inverse_transform(binary.reshape(1, -1))[0])

    # Filter movies that share ≥1 predicted genre
    mask = MOVIES_DF["genres"].apply(lambda g: any(x in g for x in pred_genres))
    idxs = np.where(mask.values)[0]

    # Fallback: if nothing matched, use whole dataset
    if len(idxs) == 0:
        idxs = np.arange(len(MOVIES_DF))

    # Cosine similarity between user_vec and candidate movies
    embedding_idxs = MOVIES_DF.iloc[idxs]["embedding_index"].values
    sims = util.cos_sim(user_vec, EMB_MATRIX[embedding_idxs])[0].cpu().numpy()
    top_local = sims.argsort()[-top_k:][::-1]          # indices inside idxs array
    top_rows  = MOVIES_DF.iloc[idxs[top_local]].copy()
    top_rows["similarity"] = sims[top_local]

    # Return genres + top movies as plain Python structures
    return pred_genres, top_rows[["title", "overview", "genres", "similarity"]]

# --------------------------------------------------------------------
# API endpoints
# --------------------------------------------------------------------
@app.post("/predict")
async def predict(user: UserInput):
    if not user.overview.strip():
        raise HTTPException(400, "Overview cannot be empty.")

    genres, recs_df = get_recommendations(user.overview, top_k=5)
    return {
        "genres": genres,
        "recommendations": recs_df.to_dict("records")
    }

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    html_path = FRONT_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))
