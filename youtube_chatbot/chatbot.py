"""
YouTube Transcript RAG Pipeline
-------------------------------
Fetch a YouTube transcript, chunk it, embed it into a FAISS vector store,
and answer questions about the video using a Hugging Face LLM.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Iterable

from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
load_dotenv()


@dataclass(frozen=True)
class Config:
    """Runtime configuration for the pipeline."""

    video_id: str = "LPZh9BOjkQs"          # <-- single source of truth
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_repo_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 4
    temperature: float = 0.2
    max_new_tokens: int = 512
    language: str = "en"

    @property
    def hf_token(self) -> str:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise EnvironmentError(
                "HF_TOKEN is not set. Add it to your environment or .env file."
            )
        return token


CONFIG = Config()


# ── 1. FETCH TRANSCRIPT ───────────────────────────────────────────────────────
def fetch_transcript(video_id: str, language: str = "en") -> str:
    """Fetch and concatenate the transcript of a YouTube video."""
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id, languages=[language])
        transcript = " ".join(chunk.text for chunk in transcript_list)
    except TranscriptsDisabled as exc:
        raise RuntimeError(f"Captions are disabled for video {video_id}.") from exc
    except NoTranscriptFound as exc:
        raise RuntimeError(
            f"No {language!r} transcript found for video {video_id}."
        ) from exc

    logger.info("Transcript fetched (%d characters).", len(transcript))
    return transcript


# ── 2. SPLIT INTO CHUNKS ──────────────────────────────────────────────────────
def split_transcript(
    transcript: str, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split a raw transcript into overlapping documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.create_documents([transcript])
    logger.info("Created %d chunks.", len(chunks))
    return chunks


# ── 3. VECTOR STORE ───────────────────────────────────────────────────────────
def build_vector_store(chunks: list[Document], embed_model: str) -> FAISS:
    """Embed documents and build an in‑memory FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    store = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS vector store ready.")
    return store


# ── 4. LLM ────────────────────────────────────────────────────────────────────
def build_llm(cfg: Config) -> ChatHuggingFace:
    """Instantiate the HF Inference endpoint as a chat model."""
    endpoint = HuggingFaceEndpoint(
        repo_id=cfg.llm_repo_id,
        huggingfacehub_api_token=cfg.hf_token,
        task="conversational",           # required for instruct models
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
    )
    return ChatHuggingFace(llm=endpoint)


# ── 5. PROMPT ─────────────────────────────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Answer ONLY from the provided transcript context. "
            "If the context is insufficient, say you don't know.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def format_docs(docs: Iterable[Document]) -> str:
    """Join retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


# ── 6. BUILD CHAIN ────────────────────────────────────────────────────────────
def build_chain(cfg: Config):
    """Wire retriever + prompt + LLM into a runnable chain."""
    transcript = fetch_transcript(cfg.video_id, cfg.language)
    chunks = split_transcript(transcript, cfg.chunk_size, cfg.chunk_overlap)
    vector_store = build_vector_store(chunks, cfg.embed_model)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": cfg.retriever_k},
    )
    llm = build_llm(cfg)

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    return parallel_chain | PROMPT | llm | StrOutputParser()


# ── 7. ENTRYPOINT ─────────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "Is nuclear fusion discussed in this video? If yes, what was discussed?",
    "Who is Demis?",
    "Can you summarize the video?",
]


def main() -> int:
    try:
        chain = build_chain(CONFIG)
    except EnvironmentError as exc:
        logger.error("%s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("Failed to build chain: %s", exc)
        return 1

    for question in EXAMPLE_QUESTIONS:
        logger.info("Q: %s", question)
        answer = chain.invoke(question)
        print(f"A: {answer}\n{'-' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())





# import os
# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser

# # ── CONFIG ────────────────────────────────────────────────────────────────────
# HF_TOKEN    = os.environ.get("HF_TOKEN", "your_huggingface_token_here")
# VIDEO_ID    = "Gfr50f6ZBvo"   # only the ID, not the full URL

# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # free, no token needed
# LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"       # free via HF Inference API
# # ─────────────────────────────────────────────────────────────────────────────


# # ── 1. FETCH TRANSCRIPT ───────────────────────────────────────────────────────
# try:
#     transcript_list = YouTubeTranscriptApi.get_transcript(VIDEO_ID, languages=["en"])
#     transcript = " ".join(chunk["text"] for chunk in transcript_list)
#     print("Transcript fetched successfully.\n")
# except TranscriptsDisabled:
#     raise RuntimeError("No captions available for this video.")


# # ── 2. SPLIT INTO CHUNKS ──────────────────────────────────────────────────────
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# chunks   = splitter.create_documents([transcript])
# print(f"Total chunks created: {len(chunks)}\n")


# # ── 3. EMBEDDINGS + VECTOR STORE ─────────────────────────────────────────────
# embeddings   = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
# vector_store = FAISS.from_documents(chunks, embeddings)
# print("Vector store ready.\n")


# # ── 4. RETRIEVER ──────────────────────────────────────────────────────────────
# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 4}
# )


# # ── 5. LLM (HuggingFace Inference API — no local download) ───────────────────
# llm = HuggingFaceEndpoint(
#     repo_id=LLM_REPO_ID,
#     huggingfacehub_api_token=HF_TOKEN,
#     temperature=0.2,
#     max_new_tokens=512,
# )


# # ── 6. PROMPT ─────────────────────────────────────────────────────────────────
# prompt = PromptTemplate(
#     template="""
# You are a helpful assistant.
# Answer ONLY from the provided transcript context.
# If the context is insufficient, just say you don't know.

# {context}
# Question: {question}
# """,
#     input_variables=["context", "question"],
# )


# # ── 7. HELPER ─────────────────────────────────────────────────────────────────
# def format_docs(retrieved_docs):
#     return "\n\n".join(doc.page_content for doc in retrieved_docs)


# # ── 8. CHAIN ──────────────────────────────────────────────────────────────────
# parallel_chain = RunnableParallel({
#     "context":  retriever | RunnableLambda(format_docs),
#     "question": RunnablePassthrough(),
# })

# parser     = StrOutputParser()
# main_chain = parallel_chain | prompt | llm | parser


# # ── 9. RUN EXAMPLE QUERIES ───────────────────────────────────────────────────
# if __name__ == "__main__":
#     questions = [
#         "is the topic of nuclear fusion discussed in this video? if yes then what was discussed",
#         "who is Demis",
#         "Can you summarize the video",
#     ]

#     for q in questions:
#         print(f"Q: {q}")
#         print(f"A: {main_chain.invoke(q)}\n")
#         print("-" * 60)
