import os
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from transcript_downloader import fetch_transcript



load_dotenv()



OUTPUT_DIR = "downloaded_transcript"
api_key = os.getenv("GROQ_API_KEY")

video_url="https://www.youtube.com/watch?v=oCx6f3wNP1w&t=371s"

os.makedirs(OUTPUT_DIR, exist_ok=True)


transcript_file_path = fetch_transcript(video_url, preferred_languages=['hi'])

if transcript_file_path:
    print(f"\nTranscript file downloaded at: {transcript_file_path}")


# Initialize Groq with a supported model
llm = ChatGroq(
    temperature=0.2,
    model_name="llama3-70b-8192",
    api_key= api_key
)

# Load Hindi transcript
with open(transcript_file_path, "r", encoding="utf-8") as f:
    hindi_transcript = f.read()

# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=300,
    length_function=len
)
chunks = text_splitter.split_text(hindi_transcript)

# Define the translation prompt
translation_prompt = ChatPromptTemplate.from_template("""
You are an expert Hindi-to-English translator.  
Convert this Hindi educational content into clear, structured English notes:

{transcript}

Guidelines:
1. Keep notes concise but informative.
2. Maintain original meaning accurately.
3. Use bullet points where helpful.
4. Organize content logically with headings if needed.
""")

# Process chunks with rate limiting
notes = []
for i, chunk in enumerate(chunks):
    try:
        result = translation_prompt | llm
        response = result.invoke({"transcript": chunk})
        notes.append(response.content)
        print(f"✅ Processed chunk {i+1}/{len(chunks)}")
        
        # Avoid rate limits
        if i < len(chunks) - 1:
            time.sleep(2)
        
    except Exception as e:
        print(f"❌ Error in chunk {i+1}: {str(e)}")
        notes.append(f"[ERROR IN CHUNK {i+1}: {str(e)}]")

# Combine initial results
initial_notes = "\n\n--- NOTES CONTINUED ---\n\n".join(notes)

# Save initial output
with open("student_notes_english_initial.txt", "w", encoding="utf-8") as f:
    f.write(initial_notes)

print("✨ Initial translation complete! Starting refinement process...")

# Now refine the notes with a second LLM pass
refinement_prompt = ChatPromptTemplate.from_template("""
You are an expert educational content editor. Your task is to improve these study notes:

{initial_notes}

Improvement guidelines:
1. Correct any translation errors or awkward phrasing
2. Ensure technical terms are accurately translated
3. Improve organization with clear headings and subheadings
4. Add structure with bullet points or numbered lists where helpful
5. Remove redundant information
6. Highlight key concepts and important points
7. Ensure consistency in terminology
8. Make sure the notes flow logically

Return the refined version with clear section breaks and markdown-style formatting for better readability.
""")

# Split the initial notes into chunks for refinement (may need different chunk size)
refinement_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,  # Slightly larger as we're working with English now
    chunk_overlap=400,
    length_function=len
)
refinement_chunks = refinement_splitter.split_text(initial_notes)

# Process refinement chunks
refined_notes = []
for i, chunk in enumerate(refinement_chunks):
    try:
        result = refinement_prompt | llm
        response = result.invoke({"initial_notes": chunk})
        refined_notes.append(response.content)
        print(f"✅ Refined chunk {i+1}/{len(refinement_chunks)}")
        
        # Avoid rate limits
        if i < len(refinement_chunks) - 1:
            time.sleep(2)
        
    except Exception as e:
        print(f"❌ Error refining chunk {i+1}: {str(e)}")
        refined_notes.append(f"[ERROR IN REFINING CHUNK {i+1}: {str(e)}]")

# Combine refined results
final_notes = "\n\n--- REFINED NOTES CONTINUED ---\n\n".join(refined_notes)

# Save final output
with open("student_notes_english_refined.txt", "w", encoding="utf-8") as f:
    f.write(final_notes)

print("🎉 Note refinement complete! Final version saved.")