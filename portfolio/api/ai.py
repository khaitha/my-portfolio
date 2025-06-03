import io
import os
import traceback
import warnings

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List

import pdfplumber
from transformers import logging as tf_logging

# Correct import for Google’s GenAI client:
from google import genai

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# — suppress pdfplumber CropBox warnings —
warnings.filterwarnings("ignore", message="CropBox missing from /Page")

# — quiet transformers logs (we only use it to silence pdfplumber output) —
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
tf_logging.set_verbosity_error()

# — load your Google API key from the environment —
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

# Instantiate the client correctly:
client = genai.Client(api_key=GOOGLE_API_KEY)

# Shared “system prompt” for both endpoints:
SYSTEM_PROMPT = (
    "Analyze the user’s text: if it’s a question, provide a concise answer that includes "
    "all relevant details; if it’s a statement, produce a summary covering every key point. "
    "You may use bullet points to enhance readability. Do not restate these instructions or include section headings."
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # → lock this down in production
    allow_methods=["POST"],
    allow_headers=["*"],
)


def extract_text_from_pdf(data: bytes) -> str:
    """
    Extract all page text from a PDF (as one big string).
    """
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "".join(page.extract_text() or "" for page in pdf.pages)


def clean_response(raw: str) -> str:
    """
    Remove any accidental repetition of SYSTEM_PROMPT or stray markdown markers.
    """
    cleaned = raw.replace("**", "").strip()
    if cleaned.startswith(SYSTEM_PROMPT):
        cleaned = cleaned[len(SYSTEM_PROMPT) :].strip()
    return cleaned


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    1) Read the uploaded PDF bytes
    2) Extract text
    3) Send “SYSTEM_PROMPT + PDF text” to Google’s GenAI
    4) Return the “cleaned” result
    """
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDFs allowed")

    data = await file.read()
    text = extract_text_from_pdf(data)

    if not text.strip():
        return {"error": "No text extracted from PDF."}

    try:
        # Build a single prompt string: system prompt + user text
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {text}\nAssistant:"

        # Call Google GenAI’s generate_content endpoint
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        raw_out = response.text or ""
        result = clean_response(raw_out)
        return {"result": result}

    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "Processing failed")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatHistoryRequest(BaseModel):
    messages: List[ChatMessage]


@app.post("/chat")
async def chat(request: ChatHistoryRequest):
    """
    1) Take the array of {role, content} messages from the client
    2) Prepend SYSTEM_PROMPT and fold everything into one prompt string
    3) Send the concatenated prompt to Gemini
    4) Return Gemini’s reply
    """
    # Build a single prompt string from the history
    pieces = [SYSTEM_PROMPT]
    for msg in request.messages:
        prefix = "User:" if msg.role == "user" else "Assistant:"
        pieces.append(f"{prefix} {msg.content}")
    # Let the assistant continue from the last “Assistant:” turn
    pieces.append("Assistant:")

    prompt = "\n\n".join(pieces)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        raw_out = response.text or ""
        result = clean_response(raw_out)
        return {"result": result}

    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "Chat processing failed")
