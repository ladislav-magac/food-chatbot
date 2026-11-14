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

class NutritionExpertAgent:
    def __init__(self):
        """
        Initialize the Nutrition Expert Agent with the Qwen ModelInference.
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

    def generate_prompts(self, user_profile: dict, retrieved_restaurants: list[dict], retrieved_recipes: list[dict]) -> List[BaseMessage]:
        """
        Create structured prompts (System Message and Human Message) for the LLM to evaluate nutritional aspects and dietary compliance in the user profile and retrieved candidates.
        """
        system_prompt = f"""You are a Nutrition Expert.
                    
        Your goal: Evaluate nutritional content, identify allergens, assess dietary restrictions, and ensure recommendations align with users' health and wellness goals.
                
        Your background: You are a registered dietitian with a master's degree in nutrition science and 8 years of clinical experience. 
        You understand macronutrients, micronutrients, and how different diets (keto, Mediterranean, plant-based, etc.) affect health. 
        You can quickly assess whether a dish fits within dietary restrictions like gluten-free, dairy-free, or low-sodium. 
        You're also sensitive to food allergies and intolerances, and you know how to balance health considerations with the 
        pleasure of eating. You believe that good nutrition doesn't mean sacrificing flavor or enjoyment.
                
        Respond with structured, actionable output."""
        
        user_message = f"""Evaluate the nutritional fit of these options:
                
        User Profile: {json.dumps(user_profile, indent=2)}
        Restaurants: {json.dumps(retrieved_restaurants[:5], indent=2)}
        Recipes: {json.dumps(retrieved_recipes[:5], indent=2)}

        Check dietary restrictions, allergens, and nutritional balance.
        Return JSON: {{"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}}
        """
                    
        prompts = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return prompts

    def evaluate_nutrition(self, user_profile: dict, retrieved_restaurants: list[dict], retrieved_recipes: list[dict]) -> Dict:
        """
        Evaluate nutritional aspects and dietary compliance in the user profile and retrieved candidates.
        """

        logger.debug(f"NutritionExpert.evaluate_nutrition called with user_profile='{user_profile}', retrieved_restaurants='{retrieved_restaurants}' and retrieved_recipes='{retrieved_recipes}'")

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
            nutrition_analysis = self.sanitize_response(llm_response)

        except Exception as e:
            nutrition_analysis = {"error": str(e)}

        return {
            "nutrition_analysis": nutrition_analysis
        }