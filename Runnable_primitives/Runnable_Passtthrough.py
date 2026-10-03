from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.3
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

prompt1 = ChatPromptTemplate.from_messages([
    ("human", "write a joke on {topic}")
])
prompt2=ChatPromptTemplate.from_messages([
    "Explain the Following joke- {text}"
])

joke_gen_chain= RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
    "joke": RunnablePassthrough(),
    "Explination": RunnableSequence(prompt2,model,parser)
})
final_chain=RunnableSequence(joke_gen_chain,parallel_chain)

result=final_chain.invoke({"topic":"cricket"})
print(result)


