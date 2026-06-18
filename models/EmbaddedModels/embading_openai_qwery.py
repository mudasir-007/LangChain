from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=32,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
result=openai_embeddings.embed_query("What is the capital of India?")
print(str(result))