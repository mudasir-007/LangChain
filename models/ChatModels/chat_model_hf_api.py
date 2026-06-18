from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

print("Step 1")
load_dotenv()

print("Step 2")
print("Token exists:", os.getenv("HUGGINGFACEHUB_API_TOKEN") is not None)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=1.1,
    task="text-generation"
)

print("Step 3")

model = ChatHuggingFace(llm=llm)

print("Step 4")

result = model.invoke("write a poem about cricket in 5 lines?")

print("Step 5")
print(result.content)