from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",  # ✅ Supported model
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=1.5,
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("write a poem about cricket in 5 lines?")
print(result.content)

