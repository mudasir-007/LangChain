from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=32
)

# Documents
documents = [
    "The capital of India is New Delhi.",
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo."
]

# Query
query = "What is the capital of India?"

# Generate embeddings
doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

# print(cosine_similarity([query_embedding], doc_embeddings) )

# Compute similarity
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

# Get most similar document
most_similar_doc_index = np.argmax(similarities)

# Output
print(f"Most similar document: {documents[most_similar_doc_index]}")