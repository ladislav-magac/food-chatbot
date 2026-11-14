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

class RecommendationExpertAgent:
    def __init__(self):
        """
        Initialize the Recommendation Expert Agent with the Qwen ModelInference.
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

    def generate_prompts(self, user_profile: dict, retrieved_restaurants: list[dict], retrieved_recipes: list[dict], trend_analysis: dict, style_analysis: dict, nutrition_analysis: dict) -> List[BaseMessage]:
        """
        Create structured prompts (System Message and Human Message) for the LLM to generate recommendations in the user profile, retrieved candidates, trend analysis, style analysis and nutrition analysis.
        """
        system_prompt = f"""You are a Recommendation Expert.
                    
        Your goal: Synthesize insights from all agents—user profiles, retrieved data, trends, food styles, and nutrition—into cohesive, well-reasoned restaurant and recipe recommendations.
                
        Your background: You are a recommendation systems architect with experience building personalization engines for major 
        food delivery platforms and recipe apps. You understand how to balance multiple signals—relevance, diversity, novelty, 
        and serendipity—to create recommendations that delight users. You know when to play it safe with familiar favorites 
        and when to suggest something unexpected. You're skilled at synthesizing complex, sometimes conflicting information 
        from multiple sources into clear, actionable recommendations. You write in a warm, engaging tone that makes users 
        excited to try new restaurants and recipes.
                
        Respond with structured, actionable output."""
        
        user_message = f"""Synthesize these insights into top 5 restaurant and top 5 recipe recommendations:
                
        User Profile: {json.dumps(user_profile, indent=2)}
        Restaurants: {json.dumps(retrieved_restaurants[:10], indent=2)}
        Recipes: {json.dumps(retrieved_recipes[:10], indent=2)}
        Trends: {json.dumps(trend_analysis, indent=2)}
        Styles: {json.dumps(style_analysis, indent=2)}
        Nutrition: {json.dumps(nutrition_analysis, indent=2)}

        Return JSON:
        {{
        "restaurants": [{{"name": str, "cuisine": str, "price": str, "reasoning": str}}],
        "recipes": [{{"name": str, "cuisine": str, "cook_time": str, "reasoning": str}}]
        }}

        Each reasoning should be 2-3 sentences explaining why it's a great match.
        """
                    
        prompts = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return prompts

    def generate_recommendation(self, user_profile: dict, retrieved_restaurants: list[dict], retrieved_recipes: list[dict], trend_analysis: dict, style_analysis: dict, nutrition_analysis: dict) -> Dict:
        """
        Synthesize all analyses into final recommendations.
        """

        logger.debug(f"RecommendationExpert.generate_recommendation called with user_profile='{user_profile}', retrieved_restaurants='{retrieved_restaurants}', retrieved_recipes='{retrieved_recipes}', trend_analysis='{trend_analysis}', style_analysis='{style_analysis}' and nutrition_analysis='{nutrition_analysis}'")

        # Create prompts for the LLM
        messages = self.generate_prompts(user_profile, retrieved_restaurants, retrieved_recipes, trend_analysis, style_analysis, nutrition_analysis)
        
        # Call the LLM to generate the answer
        try:
            response = self.json_llm.invoke(
                            messages
                        )
            # Extract the content from the response
            llm_response = response.content

            # Sanitize the response
            final_recommendations = self.sanitize_response(llm_response)

        except Exception as e:
            
            final_recommendations = {"error": str(e)}

        return {
            "final_recommendations": final_recommendations
        }