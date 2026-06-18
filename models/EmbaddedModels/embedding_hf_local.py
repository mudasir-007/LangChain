from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # ✅ model → model_name
    
)

documents = [
    "The capital of India is New Delhi.",
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo."
]

text = "What is the capital of India?"
vector = embeddings.embed_query(text)
# print(str(vector))
doc_embeddings = embeddings.embed_documents(documents)
print(str(doc_embeddings))