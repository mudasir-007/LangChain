import os
import streamlit as st

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from prompt_generator import load_research_prompt

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

# Streamlit UI
st.header('Research Tool')

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

template = load_research_prompt(on_warning=st.warning)

# Button action
if st.button("Summarize"):
    chain = template | model
    try:
        result = chain.invoke(
            {
                "paper_input": paper_input,
                "style_input": style_input,
                "length_input": length_input,
            }
        )
        st.write(result.content)
    except Exception as exc:
        st.error(f"Model request failed: {exc}")