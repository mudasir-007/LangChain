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

prompt1 = ChatPromptTemplate.from_messages([
    ("human", "Write a detailed report about: {topic}")
])

# Prompt 2: Report → Summary
prompt2 = ChatPromptTemplate.from_messages([
    ("human", "Summarize the following report in 3 lines:\n{report}")
])

parser = StrOutputParser()

chain = prompt1 | model | parser| prompt2 | model | parser

result=chain.invoke({"topic" : "cricket"})

print(result)