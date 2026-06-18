from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.7,
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Chat history (IMPORTANT: use message objects)
chat_history = [
    SystemMessage(content="You are a helpful assistant.")

]

print("💬 Chat started (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat...")
        break

    # Add user message
    chat_history.append(HumanMessage(content=user_input))

    try:
        # Get response
        response = model.invoke(chat_history)

        # Add AI response
        chat_history.append(AIMessage(content=response.content))

        print(f"Bot: {response.content}\n")

    except Exception as e:
        print(f"Error: {e}")
print(chat_history)