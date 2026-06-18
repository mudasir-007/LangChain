from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
import os
load_dotenv()

llm = HuggingFacePipeline(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=1.1,
    task="text_generation"
)
model =ChatHuggingFace(llm)

result=model.invoke("why i learn this Shit ")


print(result.content)