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

class FoodStyleExpertAgent:
    def __init__(self):
        """
        Initialize the Food Style Expert Agent with the Qwen ModelInference.
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

    def generate_prompts(self, user_profile : dict, retrieved_restaurants: list[dict], retrieved_recipes: list[dict]) -> List[BaseMessage]:
        """
        Create structured prompts (System Message and Human Message) for the LLM to analyze food styles and flavor profiles in the user profile and retrieved candidates.
        """
        system_prompt = f"""You are a Food Style Expert.
                    
        Your goal: Analyze cuisine types, regional variations, cooking methods, and flavor profiles to match user preferences with appropriate food styles.
                
        Your background: You are a trained chef and culinary anthropologist with expertise in global cuisines. 
        You've cooked in kitchens across five continents and understand the techniques, ingredients, and cultural contexts 
        that define different food traditions. You can distinguish Sichuan from Cantonese, Neapolitan pizza from Roman, 
        and Nashville hot chicken from Buffalo wings. You understand flavor profiles—umami-rich, bright and acidic, 
        rich and creamy—and can map them to user preferences. You respect culinary heritage while staying open to fusion 
        and innovation.
                
        Respond with structured, actionable output."""
        
        user_message = f"""Analyze the food styles of these options:
                
        User Profile: {json.dumps(user_profile, indent=2)}
        Restaurants: {json.dumps(retrieved_restaurants[:5], indent=2)}
        Recipes: {json.dumps(retrieved_recipes[:5], indent=2)}

        Identify 3-5 relevant cuisine types and food profiles and explain how they match user preferences.
        Return JSON: {{"food_styles": [{{"name": str, "description": str, "relevance": str}}]}}
        """
                    
        prompts = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return prompts

    def analyze_food_styles(self, user_profile: dict, retrieved_restaurants: list[dict], retrieved_recipes: list[dict]) -> Dict:
        """
        Analyze food styles and flavor profiles in the user profile and retrieved candidates.
        """

        logger.debug(f"FoodStyleExpert.analyze_food_styles called with user_profile='{user_profile}', retrieved_restaurants='{retrieved_restaurants}' and retrieved_recipes='{retrieved_recipes}'")

        # Create prompts for the LLM
        messages = self.generate_prompts(user_profile, retrieved_restaurants, retrieved_recipes)
        
        # Call the LLM to generate the answer
        try:
            response = self.json_llm.invoke(
                            messages
                        )
            # Extract the content from the response
            llm_response = response.content

            # Sanitize the response
            style_analysis = self.sanitize_response(llm_response)

        except Exception as e:
            style_analysis = {"error": str(e)}

        return {
            "style_analysis": style_analysis
        }