# transcript_downloader.py

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
from urllib.parse import urlparse, parse_qs
import os


def extract_video_id(url):
    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc:
        return parse_qs(parsed_url.query)['v'][0]
    elif "youtu.be" in parsed_url.netloc:
        return parsed_url.path.lstrip('/')
    raise ValueError("Invalid YouTube URL")

OUTPUT_DIR = "downloaded_transcript"

def fetch_transcript(video_url, preferred_languages=['en', 'hi'], output_dir=OUTPUT_DIR):
    try:
        video_id = extract_video_id(video_url)

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try preferred languages in order
        transcript = None
        for lang in preferred_languages:
            try:
                transcript = transcript_list.find_transcript([lang])
                break
            except NoTranscriptFound:
                continue

        if not transcript:
            raise NoTranscriptFound("Transcript not found in preferred languages.")

        transcript_data = transcript.fetch()
        formatter = TextFormatter()
        text_formatted = formatter.format_transcript(transcript_data)

        # Save to file
        filename = f"{video_id}_transcript.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text_formatted)

        print(f"Transcript saved to: {filepath}")
        return filepath  # <=== RETURNING THE FILE PATH

    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.")
    except Exception as e:
        print("Error:", e)
