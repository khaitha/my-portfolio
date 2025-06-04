import io
import os
import traceback
import warnings
import time
import gc
import logging
import asyncio

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List

import pdfplumber
from transformers import logging as tf_logging

# Correct import for Google's GenAI client:
from google import genai

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Shared "system prompt" for both endpoints:
SYSTEM_PROMPT = (
    "Analyze the user's text: if it's a question, provide a concise answer that includes "
    "all relevant details; if it's a statement, produce a summary covering every key point. The shorter the better."
    "You may use bullet points to enhance readability. Do not restate these instructions or include section headings."
)

CHAT_PROMPT = (
    "You are a helpful assistant. Respond to the user's messages with concise and relevant information. Most of the time you are sarcastic, but always helpful. Also keep the reply short and to the point. "
)

app = FastAPI()

# Semaphore to limit concurrent processing
processing_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent uploads

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://goldfish-app-84zag.ondigitalocean.app",  # Your frontend URL
        "http://localhost:3000",  # For local development
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Timeout middleware
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        # Set timeout for requests (30 seconds)
        return await asyncio.wait_for(call_next(request), timeout=30.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

def extract_text_from_pdf(data: bytes) -> str:
    """
    Extract text with better memory management.
    """
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text_parts = []
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                # Force garbage collection for large files
                if (i + 1) % 10 == 0:  # Every 10 pages
                    gc.collect()
            return "".join(text_parts)
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise HTTPException(500, f"PDF processing failed: {str(e)}")
    finally:
        # Ensure cleanup
        data = None
        gc.collect()

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
    3) Send "SYSTEM_PROMPT + PDF text" to Google's GenAI
    4) Return the "cleaned" result
    """
    async with processing_semaphore:
        # File size limit (10MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        
        if file.content_type != "application/pdf":
            raise HTTPException(400, "Only PDFs allowed")

        logger.info(f"Processing file: {file.filename}")
        
        data = await file.read()
        
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(413, "File too large. Maximum size is 10MB.")
        
        text = extract_text_from_pdf(data)

        if not text.strip():
            return {"error": "No text extracted from PDF."}

        try:
            # Build a single prompt string: system prompt + user text
            prompt = f"{SYSTEM_PROMPT}\n\nUser: {text}\nAssistant:"

            # Call Google GenAI's generate_content endpoint
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )

            raw_out = response.text or ""
            result = clean_response(raw_out)
            logger.info("File processing completed successfully")
            return {"result": result}

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
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
    4) Return Gemini's reply
    """
    try:
        logger.info("Processing chat request")
        
        # Build a single prompt string from the history
        pieces = [CHAT_PROMPT]
        for msg in request.messages:
            prefix = "User:" if msg.role == "user" else "Assistant:"
            pieces.append(f"{prefix} {msg.content}")
        # Let the assistant continue from the last "Assistant:" turn
        pieces.append("Assistant:")

        prompt = "\n\n".join(pieces)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        raw_out = response.text or ""
        result = clean_response(raw_out)
        logger.info("Chat processing completed successfully")
        return {"result": result}

    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(500, "Chat processing failed")
