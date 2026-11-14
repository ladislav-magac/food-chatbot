# ================================
# Import environment
# (Dependencies were installed)
# ================================

# Standard library
import asyncio
import logging
from pathlib import Path
from typing import Dict

# Third-party
import numpy as np
# Libraries to create our MCP host application
from fastmcp.client import Client, PythonStdioTransport
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

# Configuration
SERVER_SCRIPT = str(Path(__file__).parent.parent / "chromadb_mcp_server.py")
SYSTEM_PROMPT = "You are a smart, useful agent with tools to access california restaurants and recipes." # TODO

logger = logging.getLogger(__name__)

class CandidatesRetrieverAgent:
    def __init__(self):
        """
        Initialize the Candidates Retriever Agent with the Qwen ModelInference.
        """
        # Initialize the Qwen ModelInference
        self.llm = ChatOpenAI(
            model="qwen2.5:3b",
            temperature=0.7,
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

    # ================================
    # Utilities
    # ================================

    def _unwrap(self, res: dict):
        """Chroma returns lists-of-lists; unwrap the first query."""
        ids   = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return ids, metas, dists

    def _to_similarity(self, dists):
        """Convert 'smaller is better' distance to 'larger is better' similarity."""
        d = np.array(dists, dtype=np.float32)
        return 1.0 - d

    def _minmax(self, x):
        """Min-max normalize to [0, 1] with safe handling for constant arrays."""
        x = np.array(x, dtype=np.float32)
        if x.size == 0:
            return x
        lo, hi = float(x.min()), float(x.max())
        if abs(hi - lo) < 1e-8:
            return np.ones_like(x)  # all equal -> treat as same confidence
        return (x - lo) / (hi - lo)

    # MCP Host — ReAct Agent Loop
    async def call_tool(self, query: str):
        """Connect to the MCP server, discover tools, and run a ReAct loop.
        The LLM decides which tools to call, calls them via the MCP server,
        and repeats until it produces a final text response."""
        transport = PythonStdioTransport(script_path=SERVER_SCRIPT)
    
        async with Client(transport) as client:
            # Discover available tools from the MCP server
            mcp_tools = await client.list_tools()
    
            # Convert MCP tool schemas to OpenAI-style tool definitions for the LLM
            openai_tools = [
                    {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.input_schema,
                    },
                }
                for t in mcp_tools
            ]
    
            model = self.llm.bind_tools(openai_tools)
    
            # Build the message list from chat history and the new user message
            messages = [SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=query)
            ]

            result = None
    
            # ReAct loop — call tools until the LLM returns a plain text reply
            for _ in range(10):
                response = model.invoke(messages)
                messages.append(response)
    
                # No tool calls means the LLM is done — return the final response
                if not response.tool_calls:
                    if result is not None and isinstance(result.data, dict):
                        ids, metas, dists = self._unwrap(result.data)
                        sims = self._to_similarity(dists)
                        return ids, metas, sims
                    raw = response.content
                    if isinstance(raw, list):
                        return " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in raw
                        )
                    return str(raw)
    
                # Execute each tool call via the MCP server and feed results back
                for tool_call in response.tool_calls:
                    result = await client.call_tool(tool_call["name"], tool_call["args"])
                    tool_output = " ".join(
                        item.text if hasattr(item, "text") else str(item)
                        for item in result.content
                    ) if result.content else "(no result)"
                    messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
    
            return "I wasn't able to complete that request. Please try again."

    # ================================
    # Retrieval functions
    # ================================

    def retrieve_articles(self, query: str, k: int = 5, where: dict | None = None):
        return asyncio.run(self.call_tool(f"Retrieve {k} restaurants with query = {query} and where = {where}."))

    def retrieve_images_by_text(self, query: str, k: int = 5, where: dict | None = None):
        return asyncio.run(self.call_tool(f"Retrieve {k} recipes with query = {query} and where = {where}."))

    # ================================
    # Multimodal fusion
    # ================================

    def fuse_rank(
        self,
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
        t_ids, t_metas, t_sims = self.retrieve_articles(query, k=k_text, where=where_text)
        i_ids, i_metas, i_sims = self.retrieve_images_by_text(query, k=k_img, where=where_img)

        # Normalize within modality
        t_norm = self._minmax(t_sims)
        i_norm = self._minmax(i_sims)

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

    def retrieve_candidates(self, user_profile: dict) -> Dict:
        """
        Retrieve restaurant and recipe candidates from vector database.
        """

        logger.debug(f"CandidatesRetriever.retrieve_candidates called with user_profile='{user_profile}'")

        # ================================
        # Multimodal fusion (metadata filters)
        # ================================
        
        q = " ".join(user_profile.get("favorite_cuisines", []) + user_profile.get("dietary_restrictions", []) + user_profile.get("dining_occasions", []) + user_profile.get("flavor_preferences", []))
        
        where_articles = {"rating": {"$gte": user_profile.get("rating", [])}} if user_profile.get("rating", []) else None   # change to any rating present in your dataset
        where_images   = {"servings": user_profile.get("servings", [])} if user_profile.get("servings", []) else None       # optional
        
        try:
            rows = self.fuse_rank(
                q,
                k_text=40,
                k_img=40,
                w_text=0.6,
                w_img=0.4,
                where_text=where_articles,
                where_img=where_images,
                top_n=40
            )
        
            retrieved_restaurants = [r for r in rows if r["modality"] == "article"]
            retrieved_recipes = [r for r in rows if r["modality"] == "image"]
        except Exception as e:
            retrieved_restaurants, retrieved_recipes = [], []
            
        return {
            "retrieved_restaurants": retrieved_restaurants,
            "retrieved_recipes": retrieved_recipes
        }