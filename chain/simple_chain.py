from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.3  # lower = more predictable
)

model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate([
    "Generate 5 facts about {topic}",
])

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "football"})

# visualize pipeline
print(result)