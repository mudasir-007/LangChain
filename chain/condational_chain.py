from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel,RunnableBranch ,RunnableLambda         # ✅ fixed import
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=1024,                                        
    temperature=0.3
)
model = ChatHuggingFace(llm=llm) 
parser=StrOutputParser()
class feedback(BaseModel):
    sentiment: Literal["positive", "negative"]=Field(description="Give the sentiment of feedback")
parser2=PydanticOutputParser(pydantic_object=feedback)

prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful sentiment classification assistant."),
    ("human", "Classify the sentiment of the following feedback text into positive or negative\n{feedback}\n{format_instruction}")
]).partial(
    format_instruction=parser2.get_format_instructions()
)

classifier_chain=prompt1 | model | parser2

prompt2 = ChatPromptTemplate.from_messages([
    'Write an appropriate response to this positive feedback \n {feedback}',
])

prompt3 = ChatPromptTemplate.from_messages([
    'Write an appropriate response to this negative feedback \n {feedback}',
])

chain1=prompt2 | model | parser
chain2 =prompt3 | model | parser

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", chain1),   # ✅ positive → chain1
    (lambda x: x.sentiment == "negative", chain2),   # ✅ negative → chain2
    RunnableLambda(lambda x: "could not find sentiment")  # ✅ fixed typo
)
chain=classifier_chain | branch_chain
result=chain.invoke({"feedback":"this is phone"})

print(result)