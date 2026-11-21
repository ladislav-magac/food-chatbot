# ================================
# Import environment
# (Dependencies were installed)
# ================================

# Standard library
import logging
from typing import List

# Third-party
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class RecommendationEvaluatorAgent:
    def __init__(self):
        """
        Initialize the Recommendation Evaluator Agent with the Qwen ModelInference.
        """
        # Initialize the Qwen ModelInference
        self.llm = ChatOpenAI(
            model="qwen2.5:3b",
            temperature=0.7,
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response.
        """
        return response_text.strip()

    def generate_prompts(self, retrieved_restaurants: list[dict], retrieved_recipes: list[dict], final_recommendations: dict) -> List[BaseMessage]:
        """
        Create structured prompts (System Message and Human Message) for the LLM to evaluate final recommendations against the retrieved restaurants and recipes.
        """
        system_prompt = f"""You are an AI assistant designed to evaluate the accuracy and relevance of final recommendations based on retrieved restaurants and recipes.
                
        Respond with structured, actionable output."""
        
        user_message = f"""**Instructions:**
        - Evaluate the accuracy and relevance of the following final recommendations against the retrieved restaurants and recipes.
        - Check for made up restaurants and recipes (YES/NO)
        - Respond in the exact format specified below without adding any unrelated information.

        **Format:**
        YES/NO

        **Final recommendations:** {final_recommendations}
        **Retrieved restaurants:** {retrieved_restaurants}
        **Retrieved recipes:** {retrieved_recipes}

        **Respond ONLY with the above format.**
        """
                    
        prompts = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return prompts

    def evaluate_recommendation(self, retrieved_restaurants: list[dict], retrieved_recipes: list[dict], final_recommendations: dict) -> str:
        """
        Evaluate the final recommendations against the retrieved candidates.
        """

        logger.debug(f"RecommendationEvaluator.evaluate_recommendation called with retrieved_restaurants='{retrieved_restaurants}', retrieved_recipes='{retrieved_recipes}' and final_recommendations='{final_recommendations}'")

        # Create prompts for the LLM
        messages = self.generate_prompts(retrieved_restaurants, retrieved_recipes, final_recommendations)
        
        # Call the LLM to generate the answer
        try:
            response = self.llm.invoke(
                            messages
                        )
            # Extract the content from the response
            llm_response = response.content

            # Sanitize the response
            made_up = self.sanitize_response(llm_response)

        except Exception as e:
            
            made_up = "error: " + str(e)

        return made_up