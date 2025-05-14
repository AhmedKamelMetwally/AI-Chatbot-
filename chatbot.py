import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools, AgentType, initialize_agent
from langchain.agents import AgentExecutor
from langchain.chains import ConversationalRetrievalChain

# Initialize Ollama model (Gemma)
llm = ChatOllama(model="gemma")

# Initialize conversation memory to maintain context
memory = ConversationBufferMemory()

# Load Wikipedia API 
tools = load_tools(llm=llm, tool_names=["wikipedia"])

#  Agent with Wikipedia tool
agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# deployment
st.title("Ahmed Kamel Chatbot")

# Displaying chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Displaying the conversation history
for message in st.session_state.history:
    if message["role"] == "user":
        st.write(f"You: {message['text']}")
    else:
        st.write(f"Ahmed Kamel chatbot: {message['text']}")


user_input = st.text_input("You: ")

if user_input:
    # Adding user input to the conversation history
    st.session_state.history.append({"role": "user", "text": user_input})

    # Generating the response from the chatbot using rag
    response = agent.run(user_input)

    # Adding the bot response to the conversation history
    st.session_state.history.append({"role": "bot", "text": response})

    # Displaying the bot's response
    st.write(f"Ahmed Kamel chatbot: {response}")


