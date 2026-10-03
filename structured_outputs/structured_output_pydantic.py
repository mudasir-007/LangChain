from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, EmailStr, Field
from dotenv import load_dotenv
import os

from typing import Optional, Literal  # ✅ FIXED

# Load env variables
load_dotenv()

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.9,
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class ProductReview(BaseModel):
    key_themes: list[str] = Field(description="Key themes in the review")
    summary: str = Field(description="Short summary")
    sentiment: Literal['pos', 'neg', 'neut'] = Field(description="Overall sentiment")
    pros: Optional[list[str]] = Field(default=None)
    cons: Optional[list[str]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    email: Optional[EmailStr] = Field(default=None)

structured_model = model.with_structured_output(ProductReview)

result = structured_model.invoke(
    "The stainless steel water bottle is a practical and reliable everyday essential..."
)

print(result)