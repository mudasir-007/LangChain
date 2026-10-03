from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.9,
    max_new_tokens=512,
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = ChatPromptTemplate.from_template(
    "Give me the name, age, cricketer, and city of a random famous person.\n"
    "Return ONLY valid JSON.\n{format_instructions}",
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# ✅ Best way (use LCEL chain)
chain = template | model | parser

result = chain.invoke({})
print(result)

# prompt= template.format()
# result=model.invoke(prompt)
# final_result=parser.parse(result)
# print(final_result)