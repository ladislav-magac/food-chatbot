# ================================
# Import libraries
# ================================

# Standard library
import glob
import json
import os
import shutil
from pathlib import Path

# Third-party library
import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

print("✅ Environment ready")

# ================================
# Prepare image data
# ================================
IMG_DIR  = Path.home() / "food-chatbot-data" / "recipe_images" / "synthetic_recipe_images"

image_paths = sorted(glob.glob(f"{IMG_DIR}/*.png"), key=len)
print(f"✅ Images found: {len(image_paths)}")

# ================================
# Load structured data
# ================================

# NOTE: Ensure the JSON files are in your current working directory.

with open(Path.home() / "food-chatbot-data" / "structured_restaurant_data.json", "r") as f:
    restaurants = json.load(f)

with open(Path.home() / "food-chatbot-data" / "augmented_food_recipe.json", "r") as f:
    recipes = json.load(f)

print(f"✅ Loaded restaurants: {len(restaurants)}")
print(f"✅ Loaded recipes:     {len(recipes)}")

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

print("✅ Text embedder ready")


# ---- Image embedding model (512-d) ----
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

print("✅ Image embedder ready")

# ================================
# Build article and image documents
# ================================

# -------- articles --------
article_docs = []

for i, r in enumerate(restaurants):
    name = str(r.get("name", "")).strip()
    if not name:
        continue

    text = (
    f"Restaurant: {name}\n"
    f"Cuisine: {r.get('food_style','')}\n"
    f"Location: {r.get('location','')}"
    )

    # GUARANTEED UNIQUE
    doc_id = f"rest_{i}"

    article_docs.append(
        Document(
            page_content=text.strip(),
            metadata={
                "doc_id": doc_id,
                "name": name,
                "cuisine": r.get("food_style"),
                "price_range": "$" * r.get("price_range"),
                "rating": r.get("rating", r.get("average_rating")),
                "environment": r.get("environment"),
                "location": r.get("location"),
                "source": "restaurant",
            },
        )
    )

print("✅ article docs:", len(article_docs))


# -------- images --------
image_docs = []

for i, (p, rec) in enumerate(zip(image_paths, recipes), start=1):
    doc_id = f"img_{i}"

    image_docs.append(
        Document(
            # keeps retrieval results readable
            page_content=rec.get("name", f"recipe image {i}"),
            metadata={
                "doc_id": doc_id,
                "image_path": p,
                "source": "recipe_image",
                "recipe_id": rec.get("id"),
                "name": rec.get("name"),
                "cuisine": rec.get("cuisine"),
                "servings": rec.get("servings"),
                "prep_time": rec.get("prep_time"),
                "cook_time": rec.get("cook_time"),
                "image_description": rec.get("image_description"),
            },
        )
    )

print("✅ image docs:", len(image_docs))

# ================================
# Construct and Persist Multimodal Vector Indexes
# ================================

DB_DIR = str((Path.home() / "chroma_multimodal").resolve())

if os.path.isdir(DB_DIR):
    shutil.rmtree(DB_DIR)  # Reset vector DB (important for reruns)

# ----- article DB -----
A = embed_texts([d.page_content for d in article_docs])

article_db = Chroma(
    collection_name="restaurant_articles",
    persist_directory=DB_DIR,
)

article_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in article_docs],
    embeddings=A.tolist(),
    documents=[d.page_content for d in article_docs],
    metadatas=[d.metadata for d in article_docs],
)

print("✅ Article DB ready")


# ----- image DB -----
V = embed_images([d.metadata["image_path"] for d in image_docs])

image_db = Chroma(
    collection_name="food_images",
    persist_directory=DB_DIR,
)

image_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in image_docs],
    embeddings=V.tolist(),
    documents=[d.page_content for d in image_docs],
    metadatas=[d.metadata for d in image_docs],
)

print("✅ Image DB ready")
print("🎉 Multimodal Vector Index Construction COMPLETE")