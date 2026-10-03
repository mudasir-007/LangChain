from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser
from langchain_core.output_parsers.structured import ResponseSchema
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.3,
    max_new_tokens=256
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template=(
        "Give exactly 3 facts about {topic}.\n"
        "Return ONLY valid JSON.\n"
        "{format_instruction}"
    ),
    input_variables=['topic'],
    partial_variables={
        'format_instruction': parser.get_format_instructions()
    }
)

chain = template | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)
