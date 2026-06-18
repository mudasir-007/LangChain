from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=32,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
documents = [
    "The capital of India is New Delhi.",
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo."
]
embeddings = openai_embeddings.embed_documents(documents)
print(str(embeddings))