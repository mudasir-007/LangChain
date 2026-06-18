from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.9,
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


st.header("Chat with HuggingFace LLM")

user_input = st.text_input("Enter your message:")

if st.button("Send") and user_input:
    with st.spinner("Generating..."):
        try:
            result = model.invoke([HumanMessage(content=user_input)])
            st.success(result.content)
        except Exception as e:
            st.error(f"Error: {e}")