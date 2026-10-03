from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.3,
    max_new_tokens=512,
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

template1 = ChatPromptTemplate.from_template(
    "Write a detailed report on topic {topic}"  
)

template2 = ChatPromptTemplate.from_template(
    "Write a five-line summary of the following text:\n{text}"  
)
parser=StrOutputParser()

chain=template1| model | parser | template2 |model | parser

result=chain.invoke({ "topic":"Black Hole"})
print(result)