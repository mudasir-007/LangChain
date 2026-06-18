from langchain_exa import ExaSearchRetriever
from dotenv import load_dotenv
import os

load_dotenv()

# Same as: llm = OpenAI(model="gpt-3.5-turbo")
exa = ExaSearchRetriever(
    exa_api_key=os.getenv("EXA_API_KEY"),
    k=3,
    highlights=True
)

# Same as: result = llm.invoke("Hello, how are you?")
result = exa.invoke("Hello, how are you?")

# Clean print instead of raw output
for doc in result:
    print("Title:", doc.metadata.get('title'))
    print("URL:", doc.metadata.get('url'))
    highlights = doc.metadata.get('highlights', [])
    if highlights:
        print("Answer:", highlights[0][:300])
    print("-" * 50)