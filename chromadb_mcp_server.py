# ================================
# Import environment
# (Dependencies were installed)
# ================================

# Standard library
import os
import shutil
from pathlib import Path

# Third-party
import numpy as np
import torch
# Library to import to create our MCP server
from fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_core.documents import Document
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

# Initializing our MCP server instance
mcp = FastMCP("Food-Server")

# Data paths
DATA_DIR = Path.home() / "food-chatbot-data"
CULINARY_MAP_PATH = DATA_DIR / "California-Culinary-Map.txt"

# ================================
# Verify vector database
# ================================

DB_DIR = str((Path.home() / "chroma_multimodal").resolve())

if not os.path.isdir(DB_DIR):
    raise RuntimeError(
        f"Vector database directory not found: '{DB_DIR}'. "
        "Please run build_chromadb.py (Multimodal Vector Index Construction) first."
    )

article_db = Chroma(collection_name="restaurant_articles", persist_directory=DB_DIR)
image_db   = Chroma(collection_name="food_images",          persist_directory=DB_DIR)

n_articles = article_db._collection.count()
n_images   = image_db._collection.count()

if n_articles <= 0 or n_images <= 0:
    raise RuntimeError(
        "One or more collections are empty. Please rerun build_chromadb.py to rebuild the index."
    )

# Helper functions
# ================================
# Initialize embedding models
# ================================

# ---- Text embedding model (384-d) ----
text_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts, batch_size=64):
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine-ready
    ).astype(np.float32)

# ---- CLIP embedding model (512-d) for image + query text ----
device = "cpu"
clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
clip_model.eval()

@torch.no_grad()
def embed_images(paths, batch_size=16):
    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)          # (B,512)
        feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)         # cosine-ready
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)

@torch.no_grad()
def embed_query_clip_text(query: str):
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
    feats = clip_model.get_text_features(**inputs)              # (1,512)
    feats = feats.pooler_output
    feats = feats / feats.norm(dim=-1, keepdim=True)            # cosine-ready
    return feats[0].cpu().numpy().astype(np.float32)

# MCP Resource - Exposing the Raw Culinary Map data
@mcp.resource("culinary-map://california")
def get_culinary_map() -> str:
    """The full raw California Culinary Map text.
    Contains detailed descriptions of 100+ restaurants across California
    including their vibes, cuisines, ratings, and price ranges."""
    return CULINARY_MAP_PATH.read_text()

# TOOL 1 — Get Restaurants
@mcp.tool()
def get_restaurants(query: str, k: int = 5, where: dict | None = None) -> dict:
    """Search for restaurants."""
    q_vec = embed_texts([query])[0]  # 384-d
    res = article_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return res

# TOOL 2 — Get Recipes
@mcp.tool()
def get_recipes(query: str, k: int = 5, where: dict | None = None) -> dict:
    """Search for recipes."""
    q_vec = embed_query_clip_text(query)  # 512-d
    res = image_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return res

# TOOL 3 — Add a Restaurant
@mcp.tool()
def add_restaurant(name: str, cuisine: str, price: str, rating: float, location: str, description: str) -> str:
    """Add a new restaurant to the database."""
    print(f"Adding restaurant: {name}")
    # ================================
    # Build article document
    # ================================

    if not name:
        return

    name = name.strip()

    text = (
        f"Restaurant: {name}\n"
        f"Cuisine: {cuisine}\n"
        f"Location: {location}"
    )

    i = article_db._collection.count()

    # GUARANTEED UNIQUE
    doc_id = f"rest_{i}"

    article_doc = Document(
        page_content=text.strip(),
        metadata={
            "doc_id": doc_id,
            "name": name,
            "cuisine": cuisine,
            "location": location,
            "price_range": price,
            "rating": rating,
            "environment": description,
            "source": "restaurant",
        },
    )

    # ----- article DB -----
    A = embed_texts([article_doc.page_content])

    article_db._collection.upsert(
        ids=[article_doc.metadata["doc_id"]],
        embeddings=A.tolist(),
        documents=[article_doc.page_content],
        metadatas=[article_doc.metadata],
    )
    return f"✅ Successfully added '{name}' to the database!"

# TOOL 4 — Add a Recipe
@mcp.tool()
def add_recipe(name: str, cuisine: str, difficulty: str, servings: int, prep_time: str, cook_time: str, ingredients: str, src: str, instructions: str) -> str:
    """Add a new recipe to the database."""
    print(f"Adding recipe: {name}")
    i = image_db._collection.count() + 1
    image_path = str((Path.home() / "food-chatbot-data" / "recipe_images" / "added_recipe_images" / f"recipe{i}.png").resolve())
    shutil.copy(src, image_path)

    # -------- image --------
    doc_id = f"img_{i}"
    
    image_doc = Document(
        # keeps retrieval results readable
        page_content=name,
        metadata={
            "doc_id": doc_id,
            "image_path": image_path,
            "source": "recipe_image",
            "recipe_id": i,
            "name": name,
            "cuisine": cuisine,
            "servings": servings,
            "prep_time": prep_time,
            "cook_time": cook_time,
            "image_description": difficulty + " -> " + ingredients + " -> " + instructions,
        },
    )

    # ----- image DB -----
    V = embed_images([image_doc.metadata["image_path"]])

    image_db._collection.upsert(
        ids=[image_doc.metadata["doc_id"]],
        embeddings=V.tolist(),
        documents=[image_doc.page_content],
        metadatas=[image_doc.metadata],
    )

    return f"✅ Successfully added '{name}' recipe to the database!"

# Run the Server
if __name__ == "__main__":
    mcp.run()