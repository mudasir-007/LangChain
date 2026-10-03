from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.3
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# Prompt 1: Topic → Report
prompt1 = ChatPromptTemplate.from_messages([
    
    ( "Write a detailed report about: {topic}")
])

# Prompt 2: Report → Summary
prompt2 = ChatPromptTemplate.from_messages([
    ("Summarize the following report:\n{report}")
])

# ✅ SINGLE CHAIN (this is what you wanted)
chain = prompt1 | model | parser | prompt2 | model | parser

# Run
result = chain.invoke({"topic": "cricket"})

print(result)












# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# # LLM setup
# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation",
#     max_new_tokens=256,
#     temperature=0.3
# )

# model = ChatHuggingFace(llm=llm)
# parser = StrOutputParser()

# # Prompt 1: Topic → Report
# prompt1 = ChatPromptTemplate.from_messages([
#     ("system", "You are an expert writer."),
#     ("human", "Write a detailed and structured report about: {topic}")
# ])

# # Prompt 2: Report → Summary
# prompt2 = ChatPromptTemplate.from_messages([
#     ("system", "You are a skilled editor."),
#     ("human", "Summarize the following report:\n{report}")
# ])

# # Chains
# chain1 = prompt1 | model | parser
# chain2 = prompt2 | model | parser

# # Step 1: Generate report
# report = chain1.invoke({"topic": "cricket"})

# # Step 2: Generate summary from report
# summary = chain2.invoke({"report": report})

# # Output
# print("REPORT:\n", report)
# print("\nSUMMARY:\n", summary)