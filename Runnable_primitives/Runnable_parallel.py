from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.3
)

model = ChatHuggingFace(llm=llm)

prompt1 = ChatPromptTemplate.from_messages([
    ("human", "Generate a tweet about  {topic}")
])
prompt2 =ChatPromptTemplate.from_messages([
    "Generate a  linkedin_post about - {topic}"
])
parser=StrOutputParser()

parallel_chain=RunnableParallel({
    "tweet":RunnableSequence(prompt1,model,parser),
    "linkedin_post":RunnableSequence(prompt2,model,parser)
})
result=parallel_chain.invoke({"topic":"AI"})

print(result["tweet"])
print(result["linkedin_post"])