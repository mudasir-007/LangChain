from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.9,max_completion_tokens=1000)
result=model.invoke("What is the capital of France?")
print(result.content)   