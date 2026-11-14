# ================================
# Import environment
# (Dependencies were installed)
# ================================

# Standard library
import json
import logging
from typing import Dict, List

# Third-party
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class FoodTrendAnalystAgent:
    def __init__(self):
        """
        Initialize the Food Trend Analyst Agent with the Qwen ModelInference.
        """
        # Initialize the Qwen ModelInference
        self.llm = ChatOpenAI(
            model="qwen2.5:3b",
            temperature=0.7,
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        self.json_llm = self.llm.bind(
            response_format={"type": "json_object"}
        )

    def sanitize_response(self, response_text: str) -> dict:
        """
        Sanitize the LLM's response by loading json.
        """
        return json.loads(response_text)

    def generate_prompts(self, retrieved_restaurants: list[dict], retrieved_recipes: list[dict]) -> List[BaseMessage]:
        """
        Create structured prompts (System Message and Human Message) for the LLM to analyze food trends in the retrieved candidates.
        """
        system_prompt = f"""You are a Food Trend Analyst.
                    
        Your goal: Identify current food trends, popular ingredients, emerging dining concepts, and culinary movements to ensure recommendations are timely and culturally relevant.
                
        Your background: You are a culinary journalist and trend forecaster who has spent 15 years covering food culture across 
        global markets. You have your finger on the pulse of what's happening in the food world—from viral TikTok recipes to 
        Michelin-starred innovations. You track emerging ingredients like kelp noodles and yuzu, monitor the rise of food 
        movements like plant-based dining and zero-waste cooking, and spot the next big thing before it goes mainstream. 
        You read Eater, Bon Appétit, and industry reports daily, and you know the difference between a fleeting fad and a 
        lasting trend.
                
        Respond with structured, actionable output."""
        
        user_message = f"""Analyze current food trends in these options:
                
        Restaurants: {json.dumps(retrieved_restaurants[:5], indent=2)}
        Recipes: {json.dumps(retrieved_recipes[:5], indent=2)}

        Identify 3-5 relevant trends and explain how they align with modern dining culture.
        Return JSON: {{"trends": [{{"name": str, "description": str, "relevance": str}}]}}
        """
                    
        prompts = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return prompts

    def analyze_food_trends(self, retrieved_restaurants: list[dict], retrieved_recipes: list[dict]) -> Dict:
        """
        Analyze food trends in the retrieved candidates.
        """

        logger.debug(f"FoodTrendAnalyst.analyze_food_trends called with retrieved_restaurants='{retrieved_restaurants}' and retrieved_recipes='{retrieved_recipes}'")

        # Create prompts for the LLM
        messages = self.generate_prompts(retrieved_restaurants, retrieved_recipes)
        
        # Call the LLM to generate the answer
        try:
            response = self.json_llm.invoke(
                            messages
                        )
            # Extract the content from the response
            llm_response = response.content

            # Sanitize the response
            trend_analysis = self.sanitize_response(llm_response)

        except Exception as e:
            trend_analysis = {"error": str(e)}

        return {
            "trend_analysis": trend_analysis
        }