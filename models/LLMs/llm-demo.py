from langchain_openai import OpenAI

from dotenv import load_dotenv 
load_dotenv()
llm=OpenAI(model="gpt-3.5-turbo", temperature=0.9)
result=llm.invoke("Hello, how are you?")
print(result)

# prompt=PromptTemplate.from_template("What year did the world cup start?")
# response=llm(prompt)
# print(response)