from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.9,
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    
]
result=model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)