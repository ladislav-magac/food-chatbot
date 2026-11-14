from langgraph.graph import StateGraph
from typing import Any, TypedDict, List, Dict
from .user_profile_generator_agent import UserProfileGeneratorAgent
from .candidates_retriever_agent import CandidatesRetrieverAgent
from .food_trend_analyst_agent import FoodTrendAnalystAgent
from .food_style_expert_agent import FoodStyleExpertAgent
from .nutrition_expert_agent import NutritionExpertAgent
from .recommendation_expert_agent import RecommendationExpertAgent
import logging

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    preferences: Dict[str, Any]
    user_profile: dict
    retrieved_restaurants: List[dict]
    retrieved_recipes: List[dict]
    trend_analysis: dict
    style_analysis: dict
    nutrition_analysis: dict
    final_recommendations: dict

class AgentWorkflow:
    def __init__(self):
        self.user_profile_generator = UserProfileGeneratorAgent()
        self.candidates_retriever = CandidatesRetrieverAgent()
        self.food_trend_analyst = FoodTrendAnalystAgent()
        self.food_style_expert = FoodStyleExpertAgent()
        self.nutrition_expert = NutritionExpertAgent()
        self.recommendation_expert = RecommendationExpertAgent()
        self.compiled_workflow = self.build_workflow()  # Compile once during initialization
        
    def build_workflow(self):
        """Create and compile the multi-agent workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("generate_user_profile", self._generate_user_profile_step)
        workflow.add_node("retrieve_candidates", self._retrieve_candidates_step)
        workflow.add_node("analyze_food_trends", self._analyze_food_trends_step)
        workflow.add_node("analyze_food_styles", self._analyze_food_styles_step)
        workflow.add_node("analyze_nutrition", self._analyze_nutrition_step)
        workflow.add_node("generate_final_recommendation", self._generate_final_recommendation_step)
        
        # Define edges
        workflow.set_entry_point("generate_user_profile")
        workflow.add_edge("generate_user_profile", "retrieve_candidates")
        workflow.add_edge("retrieve_candidates", "analyze_food_trends")
        workflow.add_edge("retrieve_candidates", "analyze_food_styles")
        workflow.add_edge("retrieve_candidates", "analyze_nutrition")
        workflow.add_edge("analyze_food_trends", "generate_final_recommendation")
        workflow.add_edge("analyze_food_styles", "generate_final_recommendation")
        workflow.add_edge("analyze_nutrition", "generate_final_recommendation")
        return workflow.compile()
    
    def full_pipeline(self, preferences: Dict[str, Any]):
        try:
            print(f"[DEBUG] Starting full_pipeline with preferences='{preferences}'")
            
            initial_state = AgentState(
                preferences=preferences,
                user_profile={},
                retrieved_restaurants=[],
                retrieved_recipes=[],
                trend_analysis={},
                style_analysis={},
                nutrition_analysis={},
                final_recommendation={}
            )
            
            final_state = self.compiled_workflow.invoke(initial_state)
            
            return {
                "final_recommendations": final_state["final_recommendations"]
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    def _generate_user_profile_step(self, state: AgentState) -> Dict:
        print(f"[DEBUG] Entered _generate_user_profile_step with preferences='{state['preferences']}'")
        result = self.user_profile_generator.generate_user_profile(state["preferences"])
        print("[DEBUG] UserProfileGeneratorAgent generated a user profile.")
        return {"user_profile": result["user_profile"]}

    def _retrieve_candidates_step(self, state: AgentState) -> Dict:
        print(f"[DEBUG] Entered _retrieve_candidates_step with user_profile='{state['user_profile']}'")
        result = self.candidates_retriever.retrieve_candidates(state["user_profile"])
        print("[DEBUG] CandidatesRetrieverAgent retrieved restaurant and recipe candidates from vector database.")
        return {"retrieved_restaurants": result["retrieved_restaurants"], "retrieved_recipes": result["retrieved_recipes"]}

    def _analyze_food_trends_step(self, state: AgentState) -> Dict:
        print(f"[DEBUG] Entered _analyze_food_trends_step with retrieved_restaurants='{state['retrieved_restaurants']}' and retrieved_recipes='{state['retrieved_recipes']}'")
        result = self.food_trend_analyst.analyze_food_trends(state["retrieved_restaurants"], state["retrieved_recipes"])
        print("[DEBUG] FoodTrendAnalystAgent analyzed food trends in the retrieved candidates.")
        return {"trend_analysis": result["trend_analysis"]}

    def _analyze_food_styles_step(self, state: AgentState) -> Dict:
        print(f"[DEBUG] Entered _analyze_food_styles_step with user_profile='{state['user_profile']}', retrieved_restaurants='{state['retrieved_restaurants']}' and retrieved_recipes='{state['retrieved_recipes']}'")
        result = self.food_style_expert.analyze_food_styles(state["user_profile"], state["retrieved_restaurants"], state["retrieved_recipes"])
        print("[DEBUG] FoodStyleExpertAgent analyzed food styles and flavor profiles in the user profile and retrieved candidates.")
        return {"style_analysis": result["style_analysis"]}

    def _analyze_nutrition_step(self, state: AgentState) -> Dict:
        print(f"[DEBUG] Entered _analyze_nutrition_step with user_profile='{state['user_profile']}', retrieved_restaurants='{state['retrieved_restaurants']}' and retrieved_recipes='{state['retrieved_recipes']}'")
        result = self.nutrition_expert.evaluate_nutrition(state["user_profile"], state["retrieved_restaurants"], state["retrieved_recipes"])
        print("[DEBUG] NutritionExpertAgent evaluated nutritional aspects and dietary compliance in the user profile and retrieved candidates.")
        return {"nutrition_analysis": result["nutrition_analysis"]}

    def _generate_final_recommendation_step(self, state: AgentState) -> Dict:
        print(f"[DEBUG] Entered _generate_final_recommendation_step with user_profile='{state['user_profile']}', retrieved_restaurants='{state['retrieved_restaurants']}', retrieved_recipes='{state['retrieved_recipes']}', trend_analysis='{state['trend_analysis']}', style_analysis='{state['style_analysis']}' and nutrition_analysis='{state['nutrition_analysis']}'")
        result = self.recommendation_expert.generate_recommendation(state["user_profile"], state["retrieved_restaurants"], state["retrieved_recipes"], state["trend_analysis"], state["style_analysis"], state["nutrition_analysis"])
        print("[DEBUG] RecommendationExpertAgent synthesized all analyses into final recommendations.")
        return {"final_recommendations": result["final_recommendations"]}