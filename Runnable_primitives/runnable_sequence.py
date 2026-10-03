from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.3
)

model = ChatHuggingFace(llm=llm)

prompt1 = ChatPromptTemplate.from_messages([
    ("human", "write a joke on {topic}")
])
promp2=ChatPromptTemplate.from_messages([
    "Explain the Following joke- {text}"
])

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser , promp2 , model , parser)

print(chain.invoke({"topic": "AI"}))  # ✅ "AI" as a string 