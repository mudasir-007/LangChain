from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.3
)
model = ChatHuggingFace(llm=llm)
parser=StrOutputParser()


loader = TextLoader("/Users/mudasirwani/Desktop/langchain-models/Document_Loaders/cricket.txt", encoding="utf-8")
docs = loader.load()

prompt = ChatPromptTemplate.from_messages([
    "Human"',Write a summary for the following poem - \n {poem}'
])


print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))
