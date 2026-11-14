# ================================
# Import environment
# (Dependencies were installed)
# ================================

# Standard library
import json
import logging
from typing import Any, Dict, List

# Third-party
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class UserProfileGeneratorAgent:
    def __init__(self):
        """
        Initialize the User Profile Generator Agent with the Qwen ModelInference.
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

    def generate_prompts(self, preferences: Dict[str, Any]) -> List[BaseMessage]:
        """
        Create structured prompts (System Message and Human Message) for the LLM to generate user profile from input data.
        """
        system_prompt = f"""You are an User Profile Generator.
                    
        Your goal: Analyze user restaurant visit history and social media posts to create a comprehensive profile including preferences, dietary restrictions, favorite cuisines, and dining patterns.
                
        Your background: You are an expert user behavior analyst with 10 years of experience in the food and hospitality industry. 
        You excel at reading between the lines to understand not just what users say they like, but what their actions reveal 
        about their true preferences. You have a talent for identifying patterns in dining behavior, recognizing subtle preferences, 
        and building rich user profiles that capture both explicit and implicit food preferences. You understand that a user's 
        social media posts and check-ins tell a story about their culinary journey, and you're skilled at extracting meaningful 
        insights from unstructured data.
                
        Respond with structured, actionable output."""
        
        user_message = f"""Analyze this user data and create a comprehensive profile:
                
        {preferences}
                
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
                    
        prompts = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return prompts

    def generate_user_profile(self, preferences: Dict[str, Any]) -> Dict:
        """
        Generate user profile using the input data.
        """

        logger.debug(f"UserProfileGenerator.generate_user_profile called with preferences='{preferences}'")

        # Create prompts for the LLM
        messages = self.generate_prompts(preferences)
        
        # Call the LLM to generate the answer
        try:
            response = self.json_llm.invoke(
                            messages
                        )
            # Extract the content from the response
            llm_response = response.content

            # Sanitize the response
            user_profile = self.sanitize_response(llm_response)

        except Exception as e:
            user_profile = {"error": str(e)}

        return {
            "user_profile": user_profile
        }