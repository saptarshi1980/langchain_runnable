from typing import Optional, List
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

import os


def analyze_question_relevance(
    video_id: str,
    question: str,
    model_dir: str,
    groq_api_key: str,
    similarity_threshold: float = 0.8  # Relaxed threshold
) -> str:
    """
    Analyzes whether a question is related to the topic of a YouTube video transcript (assumes aviation/tech focus).

    Parameters:
        video_id (str): YouTube video ID (not full URL)
        question (str): The question to analyze
        model_dir (str): Local directory for HuggingFace embedding model (e.g., all-MiniLM-L6-v2)
        groq_api_key (str): Your GROQ API key
        similarity_threshold (float): Distance threshold (lower = closer, e.g., 0.8)

    Returns:
        str: LLM's determination of whether the question is related
    """

    # Step 1: Fetch transcript
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        return "No captions available for this video."

    # Step 2: Chunk the transcript
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Step 3: Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_dir,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Step 4: Build FAISS index
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Step 5: Get top 4 relevant chunks (with scores)
    docs_with_scores = vector_store.similarity_search_with_score(question, k=4)

    # Step 6: Filter based on distance threshold (lower = more similar)
    filtered_docs = [doc for doc, score in docs_with_scores if score <= similarity_threshold]

    # Fallback: if no chunks pass the threshold, use the top 2 anyway
    if not filtered_docs:
        print("[INFO] No chunks passed threshold. Using fallback top 2 chunks.")
        filtered_docs = [doc for doc, score in docs_with_scores[:2]]

    # Step 7: Combine filtered docs into context
    context_text = "\n\n".join(doc.page_content for doc in filtered_docs)

    print("\n--- Final Context Passed to LLM ---\n")
    print(context_text[:2000] + "\n...[truncated]\n")

    # Step 8: Initialize LLM (Groq + LLaMA)
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.4,
        api_key=groq_api_key
    )

    # Step 9: Build prompt
    prompt = PromptTemplate.from_template("""
    Analyze if this question relates to the context's domain (e.g., aviation, AI careers, or technology).
    If related, summarize an answer using the context.
    If unrelated, respond ONLY with: "This query is unrelated to the content."

    Context: {context}
    Question: {question}
    """)

    # Step 10: Run chain
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "context": context_text,
        "question": question
    })

    return response


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    model_dir = "D:/huggingface_models/all-MiniLM-L6-v2"  # Or your own HF model path

    video_id = "aPObz17OyH8"  # Example video ID
    question = "can you summarize the video in 10 points and briefly discuss each points?"

    result = analyze_question_relevance(video_id, question, model_dir, api_key)

    print("\n--- RESULT ---\n")
    print(result)
