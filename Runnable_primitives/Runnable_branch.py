from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough,RunnableParallel,RunnableLambda,RunnableBranch

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.3
)
model = ChatHuggingFace(llm=llm)

prompt1 = ChatPromptTemplate.from_messages([
    "Human"',Write a detailed report on {topic}'
])

prompt2 = ChatPromptTemplate.from_messages([
    "Human",'Summarize the following text \n {text}'
])
parser=StrOutputParser()
report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))
