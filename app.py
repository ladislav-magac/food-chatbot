# ================================
# Import environment
# (Dependencies were installed)
# ================================

# Standard library
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Third-party
import gradio as gr
import numpy as np
import torch
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

print("✅ Environment ready")

# Configure OpenAI API
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

llm = ChatOpenAI(model="qwen2.5:3b", temperature=0.7, base_url="http://localhost:11434/v1", api_key="ollama")
json_llm = llm.bind(response_format={"type": "json_object"})

def classify_intent(user_message: str, llm: ChatOpenAI) -> str:
    """Classify user intent as restaurant, recipe, both, or clarification."""
    
    system_prompt = """You are an intent classifier for a food recommendation system.
    
Analyze the user's message and classify it as ONE of:
- "restaurant" - User wants restaurant recommendations
- "recipe" - User wants recipe recommendations
- "both" - User wants both restaurant and recipe recommendations
- "clarification" - User needs help or is asking a question
- "database" - User wants to add/edit/delete database entries

Examples:
"Where should I eat tonight?" → restaurant
"How do I make lasagna?" → recipe
"I want dinner ideas" → both
"What can you help me with?" → clarification
"I want to add a new restaurant" → database

Respond with ONLY the classification label."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    response = llm.invoke(messages)
    intent = response.content.strip().lower()
    
    # Validate intent
    valid_intents = ["restaurant", "recipe", "both", "clarification", "database"]
    if intent not in valid_intents:
        intent = "clarification"
    
    return intent

print("Intent classification function created!")

def extract_preferences(user_message: str, llm: ChatOpenAI) -> Dict[str, Any]:
    """Extract user preferences from natural language input."""
    
    system_prompt = """You are a preference extractor for a food recommendation system.
    
Extract user preferences from their message and return JSON with these keys:
- favorite_cuisines: List of mentioned cuisines (e.g., ["Italian", "Thai"])
- dietary_restrictions: List of dietary needs (e.g., ["vegetarian", "gluten-free"])
- dining_occasion: Type of dining (e.g., "casual", "fine dining", "quick bite")
- price_range: Price preference (e.g., "$", "$$", "$$$", "$$$$")
- flavor_preferences: List of flavor preferences (e.g., ["spicy", "sweet"])
- other_preferences: Any other relevant details

If a field is not mentioned, use an empty list or "not specified".

Example:
Input: "I love spicy Thai food and I'm vegetarian"
Output: {
  "favorite_cuisines": ["Thai"],
  "dietary_restrictions": ["vegetarian"],
  "dining_occasion": "not specified",
  "price_range": "not specified",
  "flavor_preferences": ["spicy"],
  "other_preferences": ""
}

Respond with ONLY valid JSON."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    response = llm.invoke(messages)
    
    try:
        preferences = json.loads(response.content)
    except:
        # Fallback if parsing fails
        preferences = {
            "favorite_cuisines": [],
            "dietary_restrictions": [],
            "dining_occasion": "not specified",
            "price_range": "not specified",
            "flavor_preferences": [],
            "other_preferences": ""
        }
    
    return preferences

print("Preference extraction function created!")

# ================================
# Verify vector database
# ================================

DB_DIR = str((Path.home() / "chroma_multimodal").resolve())

if not os.path.isdir(DB_DIR):
    raise RuntimeError(
        f"Vector database directory not found: '{DB_DIR}'. "
        "Please run build_index.py (Multimodal Vector Index Construction) first."
    )

article_db = Chroma(collection_name="restaurant_articles", persist_directory=DB_DIR)
image_db   = Chroma(collection_name="food_images",          persist_directory=DB_DIR)

n_articles = article_db._collection.count()
n_images   = image_db._collection.count()

if n_articles <= 0 or n_images <= 0:
    raise RuntimeError(
        "One or more collections are empty. Please rerun build_index.py to rebuild the index."
    )

print(f"✅ Article vectors: {n_articles}")
print(f"✅ Image vectors:   {n_images}")

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


# ---- CLIP embedding model (512-d) for image + query text ----
device = "cpu"
clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
clip_model.eval()

@torch.no_grad()
def embed_query_clip_text(query: str):
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
    feats = clip_model.get_text_features(**inputs)              # (1,512)
    feats = feats.pooler_output
    feats = feats / feats.norm(dim=-1, keepdim=True)            # cosine-ready
    return feats[0].cpu().numpy().astype(np.float32)

print("✅ CLIP embedder ready")

# ================================
# Utilities
# ================================

def _unwrap(res: dict):
    """Chroma returns lists-of-lists; unwrap the first query."""
    ids   = res.get("ids", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists

def _to_similarity(dists):
    """Convert 'smaller is better' distance to 'larger is better' similarity."""
    d = np.array(dists, dtype=np.float32)
    return 1.0 - d

def _minmax(x):
    """Min-max normalize to [0, 1] with safe handling for constant arrays."""
    x = np.array(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-8:
        return np.ones_like(x)  # all equal -> treat as same confidence
    return (x - lo) / (hi - lo)

# ================================
# Retrieval functions
# ================================

def retrieve_articles(query: str, k: int = 5, where: dict | None = None):
    q_vec = embed_texts([query])[0]  # 384-d
    res = article_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    sims = _to_similarity(dists)
    return ids, docs, metas, sims

def retrieve_images_by_text(query: str, k: int = 5, where: dict | None = None):
    q_vec = embed_query_clip_text(query)  # 512-d
    res = image_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    sims = _to_similarity(dists)
    return ids, docs, metas, sims

print("✅ Retrieval functions ready")

# ================================
# Multimodal fusion
# ================================

def fuse_rank(
    query: str,
    k_text: int = 5,
    k_img: int = 5,
    w_text: float = 0.6,
    w_img: float = 0.4,
    where_text: dict | None = None,
    where_img: dict | None = None,
    top_n: int = 5
):
    # Retrieve per modality
    t_ids, t_docs, t_metas, t_sims = retrieve_articles(query, k=k_text, where=where_text)
    i_ids, i_docs, i_metas, i_sims = retrieve_images_by_text(query, k=k_img, where=where_img)

    # Normalize within modality
    t_norm = _minmax(t_sims)
    i_norm = _minmax(i_sims)

    # Build one mixed candidate list with fused scores
    rows = []
    for j in range(len(t_ids)):
        rows.append({
            "modality": "article",
            "name": t_metas[j].get("name", "N/A") if isinstance(t_metas[j], dict) else "N/A",
            "cuisine": t_metas[j].get("cuisine", "N/A") if isinstance(t_metas[j], dict) else "N/A",
            "price": t_metas[j].get("price_range", "N/A") if isinstance(t_metas[j], dict) else "N/A",
            "rating": t_metas[j].get("rating", "N/A") if isinstance(t_metas[j], dict) else "N/A",
            "description": t_metas[j].get("environment", "N/A") if isinstance(t_metas[j], dict) else "N/A",
            "text_score": float(t_norm[j]),
            "img_score": 0.0,
            "fused": float(w_text * t_norm[j]),
        })

    for j in range(len(i_ids)):
        rows.append({
            "modality": "image",
            "name": i_metas[j].get("name", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "cuisine": i_metas[j].get("cuisine", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "servings": i_metas[j].get("servings", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "prep_time": i_metas[j].get("prep_time", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "cook_time": i_metas[j].get("cook_time", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "description": i_metas[j].get("image_description", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "text_score": 0.0,
            "img_score": float(i_norm[j]),
            "fused": float(w_img * i_norm[j]),
        })

    # Sort by fused score (desc rerank)
    rows.sort(key=lambda r: r["fused"], reverse=True)
    
    # if top_n not specified, return full pool (k_text + k_img)
    if top_n is None:
        return rows

    top_n = max(0, min(int(top_n), len(rows)))
    return rows[:top_n]

def print_fused(rows, title: str, max_chars: int = 90):
    print(f"\n=== {title} ===")
    for idx, r in enumerate(rows, start=1):
        print(
            f"[{idx}] {r['modality']} | cuisine={r['cuisine']} | "
            f"fused={r['fused']:.4f} "
            f"(text={r['text_score']:.4f}, img={r['img_score']:.4f})"
        )

# Agent configurations
agent_configs = {
    "user_profile_generator": {
        "role": "User Profile Generator",
        "goal": "Analyze user restaurant visit history and social media posts to create a comprehensive profile including preferences, dietary restrictions, favorite cuisines, and dining patterns.",
        "backstory": """You are an expert user behavior analyst with 10 years of experience in the food and hospitality industry. 
        You excel at reading between the lines to understand not just what users say they like, but what their actions reveal 
        about their true preferences. You have a talent for identifying patterns in dining behavior, recognizing subtle preferences, 
        and building rich user profiles that capture both explicit and implicit food preferences. You understand that a user's 
        social media posts and check-ins tell a story about their culinary journey, and you're skilled at extracting meaningful 
        insights from unstructured data."""
    },
    "food_trend_analyst": {
        "role": "Food Trend Analyst",
        "goal": "Identify current food trends, popular ingredients, emerging dining concepts, and culinary movements to ensure recommendations are timely and culturally relevant.",
        "backstory": """You are a culinary journalist and trend forecaster who has spent 15 years covering food culture across 
        global markets. You have your finger on the pulse of what's happening in the food world—from viral TikTok recipes to 
        Michelin-starred innovations. You track emerging ingredients like kelp noodles and yuzu, monitor the rise of food 
        movements like plant-based dining and zero-waste cooking, and spot the next big thing before it goes mainstream. 
        You read Eater, Bon Appétit, and industry reports daily, and you know the difference between a fleeting fad and a 
        lasting trend."""
    },
    "food_style_expert": {
        "role": "Food Style Expert",
        "goal": "Analyze cuisine types, regional variations, cooking methods, and flavor profiles to match user preferences with appropriate food styles.",  # Fill this in
        "backstory": """You are a trained chef and culinary anthropologist with expertise in global cuisines. 
        You've cooked in kitchens across five continents and understand the techniques, ingredients, and cultural contexts 
        that define different food traditions. You can distinguish Sichuan from Cantonese, Neapolitan pizza from Roman, 
        and Nashville hot chicken from Buffalo wings. You understand flavor profiles—umami-rich, bright and acidic, 
        rich and creamy—and can map them to user preferences. You respect culinary heritage while staying open to fusion 
        and innovation."""
    },
    "nutrition_expert": {
        "role": "Nutrition Expert",
        "goal": "Evaluate nutritional content, identify allergens, assess dietary restrictions, and ensure recommendations align with users' health and wellness goals.",
        "backstory": """You are a registered dietitian with a master's degree in nutrition science and 8 years of clinical experience. 
        You understand macronutrients, micronutrients, and how different diets (keto, Mediterranean, plant-based, etc.) affect health. 
        You can quickly assess whether a dish fits within dietary restrictions like gluten-free, dairy-free, or low-sodium. 
        You're also sensitive to food allergies and intolerances, and you know how to balance health considerations with the 
        pleasure of eating. You believe that good nutrition doesn't mean sacrificing flavor or enjoyment."""
    },
    "recommendation_expert": {
        "role": "Recommendation Expert",
        "goal": "Synthesize insights from all agents—user profiles, retrieved data, trends, food styles, and nutrition—into cohesive, well-reasoned restaurant and recipe recommendations.",
        "backstory": """You are a recommendation systems architect with experience building personalization engines for major 
        food delivery platforms and recipe apps. You understand how to balance multiple signals—relevance, diversity, novelty, 
        and serendipity—to create recommendations that delight users. You know when to play it safe with familiar favorites 
        and when to suggest something unexpected. You're skilled at synthesizing complex, sometimes conflicting information 
        from multiple sources into clear, actionable recommendations. You write in a warm, engaging tone that makes users 
        excited to try new restaurants and recipes."""
    }
}

# Create a summary of all agents and their tasks
agents_summary = [
    {"agent": "User Profile Generator", "task": "Generate User Profile"},
    {"agent": "RAG Retriever", "task": "Retrieve Relevant Restaurants and Recipes"},
    {"agent": "Food Trend Analyst", "task": "Analyze Food Trends"},
    {"agent": "Food Style Expert", "task": "Analyze Food Styles"},
    {"agent": "Nutrition Expert", "task": "Evaluate Nutrition and Dietary Fit"},
    {"agent": "Recommendation Expert", "task": "Generate Final Recommendations"}
]

print("=" * 80)
print("MULTI-AGENT SYSTEM SUMMARY")
print("=" * 80)
for i, item in enumerate(agents_summary, 1):
    print(f"{i}. {item['agent']:30} → {item['task']}")

print("Agent configurations loaded successfully!")


def call_agent(agent_key: str, user_message: str) -> str:
    """Call an agent with a specific message and return its response."""
    config = agent_configs[agent_key]
    
    system_prompt = f"""You are a {config['role']}.
    
Your goal: {config['goal']}

Your background: {config['backstory']}

Respond with structured, actionable output."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]

    response = json_llm.invoke(messages)
    return response.content

def node_generate_profile(state: dict) -> dict:
    """Generate user profile from input data."""
    print("\n[Phase 1] Generating user profile...")
    
    user_message = f"""Analyze this user data and create a comprehensive profile:

{state['user_input']}

Provide output in JSON format with these keys:
- favorite_cuisines (list)
- dietary_restrictions (list)
- dining_occasions (list)
- price_range (string)
- adventurousness_score (1-10)
- flavor_preferences (list)
- servings (int 2/3/4/6/8)
- rating (float 3.8-5.0)
- summary (string)
"""
    
    try:
        response = call_agent("user_profile_generator", user_message)
        user_profile = json.loads(response)
        print(f"✓ User profile generated: {user_profile.get('summary', 'No summary')}")
    except Exception as e:
        print(f"⚠ Error generating profile: {e}")
        user_profile = {"error": str(e)}
    
    state["user_profile"] = user_profile
    state["workflow_step"] = "profile_generated"
    return state

def node_retrieve_candidates(state: dict) -> dict:
    """Retrieve restaurant and recipe candidates from vector database."""
    print("\n[Phase 2] Retrieving candidates from vector database...")
    
    profile = state["user_profile"]

    # ================================
    # Multimodal fusion (metadata filters)
    # ================================

    q = " ".join(profile.get("favorite_cuisines", []) + profile.get("dietary_restrictions", []) + profile.get("dining_occasions", []) + profile.get("flavor_preferences", []))

    where_articles = {"rating": {"$gte": profile.get("rating", [])}} if profile.get("rating", []) else None   # change to any rating present in your dataset
    where_images   = {"servings": profile.get("servings", [])} if profile.get("servings", []) else None       # optional

    try:
        rows = fuse_rank(
            q,
            k_text=40,
            k_img=40,
            w_text=0.6,
            w_img=0.4,
            where_text=where_articles,
            where_img=where_images,
            top_n=40
        )

        if len(rows) == 0:
            print("⚠️ No results found. Try relaxing filters (cuisine).")
        else:
            print_fused(rows, title="Multimodal fusion (metadata filters)")

        print("🎉 Multimodal Similarity Fusion and Retrieval Ranking COMPLETE")
        restaurants = [r for r in rows if r["modality"] == "article"]
        recipes = [r for r in rows if r["modality"] == "image"]
        print(f"✓ Retrieved {len(restaurants)} restaurants and {len(recipes)} recipes")
    except Exception as e:
        print(f"⚠ Error retrieving candidates: {e}")
        restaurants, recipes = [], []
    
    state["retrieved_restaurants"] = restaurants
    state["retrieved_recipes"] = recipes
    state["workflow_step"] = "candidates_retrieved"
    return state

def node_analyze_trends(state: dict) -> dict:
    """Analyze food trends in the retrieved candidates."""
    print("\n[Phase 3a] Analyzing food trends...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    
    user_message = f"""Analyze current food trends in these options:

Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}

Identify 3-5 relevant trends and explain how they align with modern dining culture.
Return JSON: {{"trends": [{{"name": str, "description": str, "relevance": str}}]}}
"""
    
    try:
        response = call_agent("food_trend_analyst", user_message)
        trend_analysis = json.loads(response)
        print(f"✓ Identified {len(trend_analysis.get('trends', []))} trends")
    except Exception as e:
        print(f"⚠ Error analyzing trends: {e}")
        trend_analysis = {"error": str(e)}
    
    state["trend_analysis"] = trend_analysis
    return state

def node_analyze_styles(state: dict) -> dict:
    """Analyze food styles and flavor profiles."""
    print("\n[Phase 3b] Analyzing food styles...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    profile = state["user_profile"]
    
    user_message = f"""Analyze the food styles of these options:

User Profile: {json.dumps(profile, indent=2)}
Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}

Identify 3-5 relevant cuisine types and food profiles and explain how they match user preferences.
Return JSON: {{"food_styles": [{{"name": str, "description": str, "relevance": str}}]}}
"""  
    
    try:
        response = call_agent("food_style_expert", user_message)
        style_analysis = json.loads(response)
        print(f"✓ Style analysis completed")
    except Exception as e:
        print(f"⚠ Error analyzing styles: {e}")
        style_analysis = {"error": str(e)}
    
    state["style_analysis"] = style_analysis
    return state

def node_evaluate_nutrition(state: dict) -> dict:
    """Evaluate nutritional aspects and dietary compliance."""
    print("\n[Phase 3c] Evaluating nutrition...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    profile = state["user_profile"]
    
    user_message = f"""Evaluate the nutritional fit of these options:

User Profile: {json.dumps(profile, indent=2)}
Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}

Check dietary restrictions, allergens, and nutritional balance.
Return JSON: {{"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}}
"""
    
    try:
        response = call_agent("nutrition_expert", user_message)
        nutrition_analysis = json.loads(response)
        print(f"✓ Nutrition evaluation completed")
    except Exception as e:
        print(f"⚠ Error evaluating nutrition: {e}")
        nutrition_analysis = {"error": str(e)}
    
    state["nutrition_analysis"] = nutrition_analysis
    return state

def node_generate_recommendations(state: dict) -> dict:
    """Synthesize all analyses into final recommendations."""
    print("\n[Phase 4] Generating final recommendations...")
    
    user_message = f"""Synthesize these insights into top 5 restaurant and top 5 recipe recommendations:

User Profile: {json.dumps(state['user_profile'], indent=2)}
Restaurants: {json.dumps(state['retrieved_restaurants'][:10], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:10], indent=2)}
Trends: {json.dumps(state['trend_analysis'], indent=2)}
Styles: {json.dumps(state['style_analysis'], indent=2)}
Nutrition: {json.dumps(state['nutrition_analysis'], indent=2)}

Return JSON:
{{
  "restaurants": [{{"name": str, "cuisine": str, "price": str, "reasoning": str}}],
  "recipes": [{{"name": str, "cuisine": str, "cook_time": str, "reasoning": str}}]
}}

Each reasoning should be 2-3 sentences explaining why it's a great match.
"""
    
    try:
        response = call_agent("recommendation_expert", user_message)
        recommendations = json.loads(response)
        print(f"✓ Generated {len(recommendations.get('restaurants', []))} restaurant recommendations")
        print(f"✓ Generated {len(recommendations.get('recipes', []))} recipe recommendations")
    except Exception as e:
        print(f"⚠ Error generating recommendations: {e}")
        recommendations = {"error": str(e)}
    
    state["final_recommendations"] = recommendations
    state["workflow_step"] = "complete"
    return state

def run_workflow(user_input: str) -> dict:
    """Run the full multi-agent workflow.
    
    Phases:
      1. User Analysis      (sequential)
      2. Data Retrieval      (sequential)
      3. Analysis            (parallel – trends, styles, nutrition)
      4. Synthesis           (sequential)
    """
    
    # Initialize shared state
    # Define the shared state structure as a dictionary.
    # # Every node reads from and writes to this state.
    state = {
        # Input
        "user_input": user_input,
        
        # Phase 1: User Analysis
        "user_profile": {},
        
        # Phase 2: Data Retrieval
        "retrieved_restaurants": [],
        "retrieved_recipes": [],
        
        # Phase 3: Analysis (Parallel)
        "trend_analysis": {},
        "style_analysis": {},
        "nutrition_analysis": {},
        
        # Phase 4: Synthesis
        "final_recommendations": {},
        
        # Metadata
        "workflow_step": "start"
    }
    
    print(f"State structure defined with {len(state)} fields:")
    for key in state:
        print(f"  - {key}")
    
    # Phase 1 – Sequential
    state = node_generate_profile(state)
    
    # Phase 2 – Sequential
    state = node_retrieve_candidates(state)
    
    # Phase 3 – Parallel using ThreadPoolExecutor
    print("\n[Phase 3] Running analysis agents in parallel...")
    
    # Each function needs its own copy of state to read from,
    # and we merge their outputs back afterwards.
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_trends   = executor.submit(node_analyze_trends, dict(state))
        future_styles   = executor.submit(node_analyze_styles, dict(state))
        future_nutrition = executor.submit(node_evaluate_nutrition, dict(state))
        
        result_trends   = future_trends.result()
        result_styles   = future_styles.result()
        result_nutrition = future_nutrition.result()
        
    # Merge parallel results back into state
    state["trend_analysis"]    = result_trends["trend_analysis"]
    state["style_analysis"]    = result_styles["style_analysis"]
    state["nutrition_analysis"] = result_nutrition["nutrition_analysis"]
    
    # Phase 4 – Sequential
    state = node_generate_recommendations(state)
    
    return state

print("✓ Workflow function built successfully!")

def run_recommendation_workflow(preferences: Dict[str, Any], recommendation_type: str) -> Dict[str, Any]:
    """Run the multi-agent workflow and return recommendations.
    
    Args:
        preferences: User preferences extracted from their message
        recommendation_type: "restaurant", "recipe", or "both"
    
    Returns:
        Dictionary with recommendations
    """
    print(f"Running workflow for {recommendation_type} recommendations...")

    try:
        result = run_workflow(preferences)
        recommendations = result["final_recommendations"]
        # Filter based on recommendation type
        if recommendation_type == "restaurant":
            return {"restaurants": recommendations["restaurants"]}
        elif recommendation_type == "recipe":
            return {"recipes": recommendations["recipes"]}
        else:  # both
            return recommendations
    except Exception as e:
        print(f"\nRecommendation workflow requires valid OpenAI API key. Error: {e}")

print("Workflow integration function created!")

def format_recommendations(recommendations: Dict[str, Any]) -> str:
    """Format recommendations for display in the chat."""
    
    output = ""
    
    # Format restaurant recommendations
    if "restaurants" in recommendations and recommendations["restaurants"]:
        output += "🍽️ **Restaurant Recommendations:**\n\n"
        for i, restaurant in enumerate(recommendations["restaurants"], 1):
            output += f"**{i}. {restaurant['name']}**\n"
            output += f"   - Cuisine: {restaurant['cuisine']}\n"
            output += f"   - Price: {restaurant['price']}\n"
            output += f"   - Why: {restaurant['reasoning']}\n\n"
    
    # Format recipe recommendations
    if "recipes" in recommendations and recommendations["recipes"]:
        output += "👨‍🍳 **Recipe Recommendations:**\n\n"
        for i, recipe in enumerate(recommendations["recipes"], 1):
            output += f"**{i}. {recipe['name']}**\n"
            output += f"   - Cuisine: {recipe['cuisine']}\n"
            output += f"   - Cook time: {recipe['cook_time']}\n"
            output += f"   - Why: {recipe['reasoning']}\n\n"
    
    if not output:
        output = "I couldn't generate recommendations. Please try again with more details about your preferences."
    
    return output

print("Formatting function created!")

def recommendation_chatbot(message: str, history: List[Tuple[str, str]]) -> str:
    """Main chatbot function that handles user requests."""
    
    try:
        # Step 1: Classify intent
        intent = classify_intent(message, llm)
        print(f"Classified intent: {intent}")
        
        # Step 2: Handle different intents
        if intent == "clarification":
            return """I'm your food recommendation assistant! I can help you with:
            
🍽️ **Restaurant recommendations** - Tell me your cuisine preferences, dietary restrictions, and occasion
👨‍🍳 **Recipe recommendations** - Let me know what you'd like to cook
📝 **Database management** - Add, update, or delete restaurants and recipes

Just describe what you're looking for, and I'll provide personalized recommendations!"""
        
        elif intent == "database":
            return """To manage the database, please use the tabs above:
            
- **Add Restaurant**: Submit a new restaurant
- **Add Recipe**: Submit a new recipe
- **Edit/Delete**: Modify or remove existing entries

Is there anything else I can help you with?"""
        
        elif intent in ["restaurant", "recipe", "both"]:
            # Step 3: Extract preferences
            preferences = extract_preferences(message, llm)
            print(f"Extracted preferences: {preferences}")
            
            # Step 4: Run workflow
            recommendations = run_recommendation_workflow(preferences, intent)
            
            # Step 5: Format output
            formatted_output = format_recommendations(recommendations)
            
            return formatted_output
        
        else:
            return "I'm not sure how to help with that. Can you rephrase your request?"
    
    except Exception as e:
        return f"I encountered an error: {str(e)}. Please make sure you have set your OpenAI API key."

print("Complete chatbot function created!")

def add_restaurant(name: str, cuisine: str, price: str, location: str, description: str) -> str:
    """Add a new restaurant to the database."""
    # In a real implementation, this would add to the vector database
    print(f"Adding restaurant: {name}")
    return f"✅ Successfully added '{name}' to the database!"

def add_recipe(name: str, cuisine: str, difficulty: str, prep_time: str, ingredients: str, instructions: str) -> str:
    """Add a new recipe to the database."""
    # In a real implementation, this would add to the vector database
    print(f"Adding recipe: {name}")
    return f"✅ Successfully added '{name}' recipe to the database!"

print("Database management functions created!")

# Create the main interface with tabs
with gr.Blocks(title="Food Recommendation Chatbot", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🍽️ Food Recommendation Chatbot
    
    Your personal AI assistant for restaurant and recipe recommendations!
    """)
    
    with gr.Tabs():
        
        # Tab 1: Chat Interface
        with gr.Tab("💬 Chat"):
            chatbot_interface = gr.ChatInterface(
                fn=recommendation_chatbot,
                examples=[
                    "I'm looking for vegetarian restaurants",
                    "Suggest some easy recipes for dinner",
                    "I want spicy Thai food recommendations",
                    "What can you help me with?"
                ],
                title="Chat with the Recommendation Assistant",
                description="Describe your food preferences and I'll recommend restaurants or recipes!"
            )
        
        # Tab 2: Add Restaurant
        with gr.Tab("➕ Add Restaurant"):
            gr.Markdown("### Add a New Restaurant to the Database")
            
            with gr.Row():
                with gr.Column():
                    rest_name = gr.Textbox(label="Restaurant Name")
                    rest_cuisine = gr.Textbox(label="Cuisine Type")
                    rest_price = gr.Dropdown(
                        choices=["$", "$$", "$$$", "$$$$"],
                        label="Price Range"
                    )
                with gr.Column():
                    rest_location = gr.Textbox(label="Location")
                    rest_description = gr.Textbox(
                        label="Description",
                        lines=3
                    )
            
            add_rest_btn = gr.Button("Add Restaurant", variant="primary")
            rest_output = gr.Textbox(label="Status")
            
            add_rest_btn.click(
                fn=add_restaurant,
                inputs=[rest_name, rest_cuisine, rest_price, rest_location, rest_description],
                outputs=rest_output
            )
        
        # Tab 3: Add Recipe
        with gr.Tab("➕ Add Recipe"):
            gr.Markdown("### Add a New Recipe to the Database")
            
            with gr.Row():
                with gr.Column():
                    recipe_name = gr.Textbox(label="Recipe Name")
                    recipe_cuisine = gr.Textbox(label="Cuisine Type")
                    recipe_difficulty = gr.Dropdown(
                        choices=["Easy", "Medium", "Hard"],
                        label="Difficulty"
                    )
                with gr.Column():
                    recipe_time = gr.Textbox(label="Prep Time")
                    recipe_ingredients = gr.Textbox(
                        label="Ingredients (comma-separated)",
                        lines=3
                    )
            
            recipe_instructions = gr.Textbox(
                label="Instructions",
                lines=5
            )
            
            add_recipe_btn = gr.Button("Add Recipe", variant="primary")
            recipe_output = gr.Textbox(label="Status")
            
            add_recipe_btn.click(
                fn=add_recipe,
                inputs=[recipe_name, recipe_cuisine, recipe_difficulty, recipe_time, recipe_ingredients, recipe_instructions],
                outputs=recipe_output
            )
        
        # Tab 4: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This Chatbot
            
            This chatbot uses a multi-agent AI system to provide personalized food recommendations.
            
            ### Features:
            - 🤖 **Intelligent Agents**: Six specialized AI agents work together to analyze your preferences
            - 🔍 **Smart Search**: Vector database retrieval finds the most relevant options
            - 🎯 **Personalized**: Recommendations tailored to your tastes and dietary needs
            - 📝 **Editable Database**: Add your favorite restaurants and recipes
            
            ### How to Use:
            1. Go to the **Chat** tab
            2. Describe what you're looking for (cuisine, dietary restrictions, occasion, etc.)
            3. Receive personalized restaurant or recipe recommendations
            4. Use the **Add** tabs to contribute to the database
            
            ### Technologies:
            - LangChain & LangGraph for multi-agent orchestration
            - OpenAI GPT-4 for language understanding
            - Vector databases for semantic search
            - Gradio for the user interface
            """)

print("Complete interface created!")
print("\nTo launch the chatbot, run: demo.launch()")

# Uncomment the line below to launch the chatbot
demo.launch(share=False)

print("To launch the chatbot interface, uncomment and run the line above.")
print("The chatbot will open in a new browser tab.")