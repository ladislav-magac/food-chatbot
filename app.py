import gradio as gr
import json
from typing import List, Tuple, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Configure OpenAI API
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

llm = ChatOpenAI(model="qwen2.5:3b", temperature=0.7, base_url="http://localhost:11434/v1", api_key="ollama")

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

def run_recommendation_workflow(preferences: Dict[str, Any], recommendation_type: str) -> Dict[str, Any]:
    """Run the multi-agent workflow and return recommendations.
    
    Args:
        preferences: User preferences extracted from their message
        recommendation_type: "restaurant", "recipe", or "both"
    
    Returns:
        Dictionary with recommendations
    """
    
    # For this demo, we'll create mock recommendations
    # In a real implementation, this would call the LangGraph workflow from Lesson 2
    
    print(f"Running workflow for {recommendation_type} recommendations...")
    
    mock_recommendations = {
        "restaurants": [
            {
                "name": "Green Leaf Bistro",
                "cuisine": "Mediterranean",
                "price": "$$",
                "reasoning": "This restaurant perfectly aligns with your preference for healthy, plant-based options and offers a diverse Mediterranean menu with excellent vegetarian choices."
            },
            {
                "name": "Spice Route",
                "cuisine": "Indian",
                "price": "$$",
                "reasoning": "Known for authentic Indian cuisine with extensive vegetarian options. The spice level can be customized to your preference."
            }
        ],
        "recipes": [
            {
                "name": "One-Pot Chickpea Curry",
                "cuisine": "Indian",
                "difficulty": "Easy",
                "reasoning": "A flavorful, protein-rich dish that matches your love for bold flavors. Ready in 30 minutes with simple ingredients."
            },
            {
                "name": "Mediterranean Quinoa Bowl",
                "cuisine": "Mediterranean",
                "difficulty": "Easy",
                "reasoning": "Nutritious and satisfying, this bowl combines your favorite Mediterranean flavors with plant-based protein."
            }
        ]
    }
    
    # Filter based on recommendation type
    if recommendation_type == "restaurant":
        return {"restaurants": mock_recommendations["restaurants"]}
    elif recommendation_type == "recipe":
        return {"recipes": mock_recommendations["recipes"]}
    else:  # both
        return mock_recommendations

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
            output += f"   - Difficulty: {recipe['difficulty']}\n"
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