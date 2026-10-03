from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

os.environ["USER_AGENT"] = "my-langchain-app/1.0"  # Fix the USER_AGENT warning too

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.3
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

url = "https://www.amazon.in/Apple-2025-MacBook-Laptop-10%E2%80%91core/dp/B0FWD7JSX7"
loader = WebBaseLoader(url)
docs = loader.load()

# Truncate to first 3000 characters to stay within token limits
page_text = docs[0].page_content[:3000]

prompt = ChatPromptTemplate.from_messages([
    ("human", "Answer the following question:\n{question}\n\nBased on this text:\n{text}")
])

chain = prompt | model | parser

result = chain.invoke({
    "question": "What is the peak brightness and price of this product",
    "text": page_text
})
print(result)