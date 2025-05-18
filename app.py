import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain.agents import initialize_agent, Tool
from langchain.schema import HumanMessage, AIMessage
import requests

# --- API KEYS ---
GROQ_API_KEY = "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"
NUTRITIONIX_APP_ID = "84bb4d66"
NUTRITIONIX_API_KEY = "31b1d56ea8facf6f1beda13a20a69966"

# --- LangChain LLM using Groq ---
llm = ChatOpenAI(
    model="llama3-70b-8192",
    openai_api_key=GROQ_API_KEY,
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.7,
)

# --- Nutrition Tool ---
@tool
def get_nutrition_data(query: str) -> str:
    """Fetches nutrition information for a given food item."""
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    headers = {
        "x-app-id": NUTRITIONIX_APP_ID,
        "x-app-key": NUTRITIONIX_API_KEY,
        "Content-Type": "application/json"
    }
    data = {"query": query}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        res = response.json()
        if "foods" in res and len(res["foods"]) > 0:
            info = res["foods"][0]
            return (
                f"{info['food_name'].title()} - "
                f"{info['nf_calories']} kcal, "
                f"{info['nf_protein']}g protein, "
                f"{info['nf_total_carbohydrate']}g carbs, "
                f"{info['nf_total_fat']}g fat."
            )
        else:
            return "No data found for the given food."
    return "Error contacting Nutritionix API."

# --- Agent Setup ---
tools = [
    Tool.from_function(
        func=get_nutrition_data,
        name="NutritionixTool",
        description="Fetches nutrition information for a given food item."
    )
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools,
    llm,
    agent="chat-conversational-react-description",
    verbose=True,
    memory=memory,
)

# --- Streamlit UI ---
st.set_page_config(page_title="LangChain Nutrition Assistant", layout="centered")
st.title("🥗 AI Smart Nutrition Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar preferences
st.sidebar.header("User Preferences")
goal = st.sidebar.selectbox("Your goal", ["Lose Weight", "Gain Muscle", "Maintain Health"])
diet = st.sidebar.selectbox("Diet Type", ["No Preference", "Vegetarian", "Vegan", "Keto"])
allergies = st.sidebar.text_input("Allergies", placeholder="e.g. nuts, dairy")

user_input = st.text_input("Ask me anything about food or nutrition:")

if st.button("Submit") and user_input:
    with st.spinner("Thinking..."):
        if not memory.chat_memory.messages:
            system_message = (
                f"User goal: {goal.lower()}. "
                f"Diet preference: {diet.lower()}. "
                f"Allergies: {allergies.lower() if allergies else 'none'}."
            )
            memory.chat_memory.add_message(HumanMessage(content=system_message))

        memory.chat_memory.add_message(HumanMessage(content=user_input))
        result = agent.run(user_input)
        memory.chat_memory.add_message(AIMessage(content=result))

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Assistant", result))

for speaker, msg in st.session_state.chat_history:
    st.chat_message(speaker).write(msg)
