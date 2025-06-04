import io
import os
import traceback
import warnings
import time
import gc
import logging
import asyncio
import psutil  # Add this import

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

# Memory monitoring helper functions
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024
    }

def log_memory(stage: str):
    """Log memory usage at a specific stage"""
    mem = get_memory_usage()
    logger.info(f"MEMORY [{stage}]: RSS={mem['rss_mb']:.1f}MB, VMS={mem['vms_mb']:.1f}MB, %={mem['percent']:.1f}%, Available={mem['available_mb']:.1f}MB")

# Global variable to track total requests (potential memory leak source)
request_count = 0

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

# Reduce concurrent processing to save memory
processing_semaphore = asyncio.Semaphore(2)  # Reduced from 3 to 2

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://goldfish-app-84zag.ondigitalocean.app",  # Your frontend URL
        "http://localhost:3000",  # For local development
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Memory check middleware
@app.middleware("http")
async def memory_check_middleware(request: Request, call_next):
    global request_count
    request_count += 1
    
    # Log memory at start of request
    log_memory(f"REQUEST_{request_count}_START")
    
    # Check if memory is too high before processing
    mem = get_memory_usage()
    if mem['percent'] > 85:  # If memory usage > 85%
        logger.warning(f"High memory usage detected: {mem['percent']:.1f}% - Rejecting request")
        raise HTTPException(503, "Server temporarily overloaded due to high memory usage")
    
    response = await call_next(request)
    
    # Log memory after request
    log_memory(f"REQUEST_{request_count}_END")
    
    return response

# Timeout middleware
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        # Reduce timeout to prevent memory accumulation
        return await asyncio.wait_for(call_next(request), timeout=20.0)  # Reduced from 30 to 20
    except asyncio.TimeoutError:
        log_memory("TIMEOUT_ERROR")
        raise HTTPException(status_code=504, detail="Request timeout")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

# Memory monitoring endpoint
@app.get("/memory")
async def get_memory_stats():
    """Get detailed memory statistics"""
    mem = get_memory_usage()
    system_mem = psutil.virtual_memory()
    
    return {
        "process_memory_mb": mem['rss_mb'],
        "process_memory_percent": mem['percent'],
        "system_total_mb": system_mem.total / 1024 / 1024,
        "system_available_mb": system_mem.available / 1024 / 1024,
        "system_used_percent": system_mem.percent,
        "request_count": request_count,
        "gc_stats": {
            "collections": gc.get_stats(),
            "count": gc.get_count()
        }
    }

# Force garbage collection endpoint
@app.post("/gc")
async def force_garbage_collection():
    """Force garbage collection - use only for debugging"""
    log_memory("BEFORE_FORCED_GC")
    collected = gc.collect()
    log_memory("AFTER_FORCED_GC")
    return {"collected_objects": collected, "message": "Garbage collection completed"}

def extract_text_from_pdf(data: bytes) -> str:
    """
    Extract text with enhanced memory management and monitoring.
    """
    log_memory("PDF_EXTRACT_START")
    
    try:
        data_size_mb = len(data) / 1024 / 1024
        logger.info(f"Processing PDF of size: {data_size_mb:.2f} MB")
        
        # Check memory before PDF processing
        mem = get_memory_usage()
        if mem['percent'] > 80:
            raise HTTPException(503, f"Insufficient memory to process PDF. Current usage: {mem['percent']:.1f}%")
        
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text_parts = []
            total_pages = len(pdf.pages)
            logger.info(f"PDF has {total_pages} pages")
            
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                    
                # More aggressive garbage collection
                if (i + 1) % 5 == 0:  # Every 5 pages instead of 10
                    log_memory(f"BEFORE_GC_PAGE_{i+1}")
                    gc.collect()
                    log_memory(f"AFTER_GC_PAGE_{i+1}")
                    
                    # Check memory after GC
                    mem = get_memory_usage()
                    if mem['percent'] > 85:
                        logger.warning(f"High memory usage during PDF processing: {mem['percent']:.1f}%")
                        # Continue but with caution
            
            final_text = "".join(text_parts)
            text_size_mb = len(final_text.encode('utf-8')) / 1024 / 1024
            logger.info(f"Total extracted text size: {text_size_mb:.2f} MB")
            
            # Clear text_parts immediately
            text_parts.clear()
            text_parts = None
            
            log_memory("PDF_EXTRACT_END")
            return final_text
            
    except Exception as e:
        log_memory("PDF_EXTRACT_ERROR")
        logger.error(f"PDF processing failed: {str(e)}")
        raise HTTPException(500, f"PDF processing failed: {str(e)}")
    finally:
        # Aggressive cleanup
        log_memory("PDF_CLEANUP_START")
        data = None
        gc.collect()
        log_memory("PDF_CLEANUP_END")

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
    Enhanced upload with detailed memory monitoring and limits
    """
    log_memory("UPLOAD_START")
    
    async with processing_semaphore:
        # Reduce file size limit to 3MB to save memory
        MAX_FILE_SIZE = 3 * 1024 * 1024  # 3MB instead of 10MB
        
        if file.content_type != "application/pdf":
            raise HTTPException(400, "Only PDFs allowed")

        logger.info(f"Processing file: {file.filename}")
        log_memory("BEFORE_FILE_READ")
        
        data = await file.read()
        log_memory("AFTER_FILE_READ")
        
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(413, "File too large. Maximum size is 3MB.")
        
        text = extract_text_from_pdf(data)
        log_memory("AFTER_TEXT_EXTRACTION")
        
        # Clear data immediately after extraction
        data = None
        gc.collect()

        if not text.strip():
            return {"error": "No text extracted from PDF."}

        try:
            # Limit text size to prevent memory issues
            MAX_TEXT_SIZE = 50000  # 50KB of text
            if len(text) > MAX_TEXT_SIZE:
                text = text[:MAX_TEXT_SIZE] + "... (truncated due to size limit)"
                logger.info(f"Text truncated to {MAX_TEXT_SIZE} characters")
            
            # Build a single prompt string: system prompt + user text
            prompt = f"{SYSTEM_PROMPT}\n\nUser: {text}\nAssistant:"
            
            # Clear original text
            text = None
            gc.collect()
            
            log_memory("BEFORE_AI_CALL")

            # Call Google GenAI's generate_content endpoint
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            
            log_memory("AFTER_AI_CALL")

            raw_out = response.text or ""
            result = clean_response(raw_out)
            
            # Clear variables immediately
            prompt = None
            raw_out = None
            response = None
            gc.collect()
            
            log_memory("UPLOAD_SUCCESS")
            logger.info("File processing completed successfully")
            return {"result": result}

        except Exception as e:
            log_memory("UPLOAD_ERROR")
            logger.error(f"Processing failed: {str(e)}")
            traceback.print_exc()
            raise HTTPException(500, "Processing failed")
        finally:
            log_memory("UPLOAD_CLEANUP")
            gc.collect()

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatHistoryRequest(BaseModel):
    messages: List[ChatMessage]

@app.post("/chat")
async def chat(request: ChatHistoryRequest):
    """
    Enhanced chat with memory monitoring and limits
    """
    log_memory("CHAT_START")
    
    try:
        logger.info("Processing chat request")
        
        # Limit chat history length to prevent memory issues
        MAX_HISTORY_LENGTH = 10
        if len(request.messages) > MAX_HISTORY_LENGTH:
            request.messages = request.messages[-MAX_HISTORY_LENGTH:]
            logger.info(f"Chat history truncated to last {MAX_HISTORY_LENGTH} messages")
        
        # Build a single prompt string from the history
        pieces = [CHAT_PROMPT]
        for msg in request.messages:
            prefix = "User:" if msg.role == "user" else "Assistant:"
            pieces.append(f"{prefix} {msg.content}")
        pieces.append("Assistant:")

        prompt = "\n\n".join(pieces)
        
        # Limit prompt size
        MAX_PROMPT_SIZE = 10000  # 10KB
        if len(prompt) > MAX_PROMPT_SIZE:
            prompt = prompt[-MAX_PROMPT_SIZE:]
            logger.info(f"Prompt truncated to {MAX_PROMPT_SIZE} characters")
        
        log_memory("BEFORE_CHAT_AI_CALL")

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        
        log_memory("AFTER_CHAT_AI_CALL")
        
        raw_out = response.text or ""
        result = clean_response(raw_out)
        
        # Clear variables
        prompt = None
        raw_out = None
        response = None
        gc.collect()
        
        log_memory("CHAT_SUCCESS")
        logger.info("Chat processing completed successfully")
        return {"result": result}

    except Exception as e:
        log_memory("CHAT_ERROR")
        logger.error(f"Chat processing failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(500, "Chat processing failed")
    finally:
        log_memory("CHAT_CLEANUP")
        gc.collect()
