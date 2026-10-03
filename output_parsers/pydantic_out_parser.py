from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.exceptions import OutputParserException

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=256,        # Ensure enough tokens for JSON output
    temperature=0.9          # Lower temp = more structured/predictable output
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person, must be greater than 18')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=(
        'Generate the name, age and city of a footballer {place} person.\n'
        'You must respond with ONLY valid JSON. No extra text.\n'
        '{format_instruction}'
    ),
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

try:
    final_result = chain.invoke({'place': 'portgal'})
    print(final_result)
except OutputParserException as e:
    print(f"Parsing failed: {e}")