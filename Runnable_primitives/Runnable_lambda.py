from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough,RunnableParallel,RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.3
)
model = ChatHuggingFace(llm=llm)
def word_count(text):
    """Count the number of words in the text"""
    return len(text.split())


prompt = ChatPromptTemplate.from_messages([
    ("human", "write a joke on {topic}")
])

parser = StrOutputParser()

joke_generate_chain= RunnableSequence(prompt,model,parser)

parallel_chain=RunnableParallel(
    {
        "joke":RunnablePassthrough(),
        "word_count":RunnableLambda(word_count)
    }
)
final_chain=RunnableSequence(joke_generate_chain,parallel_chain)
result=final_chain.invoke({"topic":"AI"})
print(result)