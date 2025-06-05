import io
import os
import traceback
import warnings
import time
import gc
import logging
import asyncio
import psutil

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List, Optional

import pdfplumber

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
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024
    }

def log_memory(stage: str):
    """Log memory usage at a specific stage"""
    mem = get_memory_usage()
    logger.info(f"MEMORY [{stage}]: RSS={mem['rss_mb']:.1f}MB, VMS={mem['vms_mb']:.1f}MB, %={mem['percent']:.1f}%, Available={mem['available_mb']:.1f}MB")

# Global variable to track total requests and store PDF context
request_count = 0
pdf_context_store = {}  # Store PDF content by session/user

# Enhanced system prompts - UPDATED for mid-conversation PDF uploads
PDF_SUMMARY_PROMPT = (
    "You are an expert document analyzer. Provide a CONCISE scrape of this document . "
    "Focus on:\n"
    "• Main topic and purpose\n"
    "• Key findings or conclusions\n"
    "• Important data or statistics (if any)\n"
    "• Practical implications\n\n"
    "Keep it brief and actionable. Use clear, simple language.\n\n"
    "Document content:\n"
)

CHAT_WITH_PDF_PROMPT = (
    "You are a helpful AI assistant with access to a PDF document that the user has uploaded during our conversation. "
    "Answer questions about the document content, provide clarifications, and help the user understand the material. "
    "Always reference specific parts of the document when relevant. "
    "If the user asks something not covered in the document, let them know politely. "
    "Keep responses concise but informative.\n\n"
    "PDF CONTENT:\n{pdf_content}\n\n"
    "CONVERSATION:\n"
)

GENERAL_CHAT_PROMPT = (
    "You are a helpful assistant. Respond to the user's messages with concise and relevant information. If user asks about document beyond its content, use your knowledge to help assist them. "
    "Be friendly and always helpful. Keep replies short and to the point. "
)

app = FastAPI()

# Reduce concurrent processing to save memory
processing_semaphore = asyncio.Semaphore(2)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://goldfish-app-84zag.ondigitalocean.app",
        "http://localhost:3000",
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
    
    response = await call_next(request)
    
    # Log memory after request
    log_memory(f"REQUEST_{request_count}_END")
    
    return response

# Timeout middleware
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=30.0)
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
        "pdf_contexts_stored": len(pdf_context_store)
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
    """Extract text with enhanced memory management and monitoring."""
    log_memory("PDF_EXTRACT_START")
    
    try:
        data_size_mb = len(data) / 1024 / 1024
        logger.info(f"Processing PDF of size: {data_size_mb:.2f} MB")
        
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text_parts = []
            total_pages = len(pdf.pages)
            logger.info(f"PDF has {total_pages} pages")
            
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {i+1} ---\n{page_text}\n")
                    
                # Garbage collection every 5 pages
                if (i + 1) % 5 == 0:
                    gc.collect()
            
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
        log_memory("PDF_CLEANUP_START")
        data = None
        gc.collect()
        log_memory("PDF_CLEANUP_END")

def clean_response(raw: str) -> str:
    """Clean up AI response"""
    cleaned = raw.replace("**", "").strip()
    return cleaned

def generate_session_id() -> str:
    """Generate a simple session ID"""
    return f"session_{int(time.time())}_{request_count}"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Enhanced PDF upload that can be used mid-conversation
    """
    log_memory("UPLOAD_START")
    
    async with processing_semaphore:
        # File size limit
        MAX_FILE_SIZE = 3 * 1024 * 1024  # 3MB
        
        if file.content_type != "application/pdf":
            raise HTTPException(400, "Only PDFs allowed")

        logger.info(f"Processing file: {file.filename}")
        log_memory("BEFORE_FILE_READ")
        
        data = await file.read()
        log_memory("AFTER_FILE_READ")
        
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(413, "File too large. Maximum size is 3MB.")
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(data)
        log_memory("AFTER_TEXT_EXTRACTION")
        
        # Clear data immediately after extraction
        data = None
        gc.collect()

        if not pdf_text.strip():
            return {"error": "No text extracted from PDF."}

        try:
            # Reduce text size for CONCISE summary
            MAX_TEXT_SIZE = 40000  # Reduced for shorter summaries
            if len(pdf_text) > MAX_TEXT_SIZE:
                truncated_text = pdf_text[:MAX_TEXT_SIZE] + "\n\n... (Document continues but was truncated for processing)"
                logger.info(f"Text truncated to {MAX_TEXT_SIZE} characters for summarization")
            else:
                truncated_text = pdf_text
            
            # Generate CONCISE summary
            prompt = f"{PDF_SUMMARY_PROMPT}{truncated_text}\n\nRemember: Keep the summary to 3-4 paragraphs maximum."
            
            log_memory("BEFORE_AI_CALL")

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            
            log_memory("AFTER_AI_CALL")

            summary = response.text or ""
            cleaned_summary = clean_response(summary)
            
            # Generate session ID and store PDF context for chat
            session_id = generate_session_id()
            
            # Store the full PDF text (not truncated) for chat context
            MAX_STORAGE_SIZE = 30000  # Reduced to save memory
            if len(pdf_text) > MAX_STORAGE_SIZE:
                stored_text = pdf_text[:MAX_STORAGE_SIZE] + "\n\n... (Document continues)"
            else:
                stored_text = pdf_text
                
            pdf_context_store[session_id] = {
                "content": stored_text,
                "filename": file.filename,
                "timestamp": time.time(),
                "summary": cleaned_summary
            }
            
            # Clean up old sessions (keep only last 5)
            if len(pdf_context_store) > 5:
                oldest_sessions = sorted(pdf_context_store.keys(), 
                                       key=lambda x: pdf_context_store[x]["timestamp"])[:len(pdf_context_store)-5]
                for old_session in oldest_sessions:
                    del pdf_context_store[old_session]
            
            # Clear variables
            prompt = None
            truncated_text = None
            pdf_text = None
            response = None
            gc.collect()
            
            log_memory("UPLOAD_SUCCESS")
            logger.info("File processing completed successfully")
            
            # Return response optimized for mid-conversation use
            return {
                "result": cleaned_summary,
                "session_id": session_id,
                "filename": file.filename,
                "message": f"PDF '{file.filename}' uploaded successfully! I can now answer questions about this document.",
                "is_mid_conversation": True  # Flag to indicate this can be used mid-conversation
            }

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
    session_id: Optional[str] = None  # Optional session ID for PDF context

@app.post("/chat")
async def chat(request: ChatHistoryRequest):
    """
    Enhanced chat with PDF context support and session switching
    """
    log_memory("CHAT_START")
    
    try:
        logger.info(f"Processing chat request with session_id: {request.session_id}")
        
        # Check if we have PDF context for this session
        pdf_context = None
        if request.session_id and request.session_id in pdf_context_store:
            pdf_context = pdf_context_store[request.session_id]
            logger.info(f"Found PDF context for session: {request.session_id} (file: {pdf_context['filename']})")
        else:
            if request.session_id:
                logger.warning(f"Session ID provided but not found: {request.session_id}")
                return {
                    "result": f"Sorry, I couldn't find the PDF session '{request.session_id}'. Please upload the PDF again.",
                    "has_pdf_context": False,
                    "pdf_filename": None,
                    "error": "session_not_found"
                }
        
        # Keep full chat history when switching contexts (no truncation for continuity)
        MAX_HISTORY_LENGTH = 12  # Increased to maintain conversation flow
        if len(request.messages) > MAX_HISTORY_LENGTH:
            # Keep first few and last few messages to maintain context
            first_messages = request.messages[:3]
            last_messages = request.messages[-(MAX_HISTORY_LENGTH-3):]
            request.messages = first_messages + last_messages
            logger.info(f"Chat history optimized: kept first 3 and last {MAX_HISTORY_LENGTH-3} messages")
        
        # Build prompt based on whether we have PDF context
        if pdf_context:
            # Chat with PDF context - maintain conversation flow
            base_prompt = CHAT_WITH_PDF_PROMPT.format(pdf_content=pdf_context["content"])
            pieces = [base_prompt]
            
            # Add context about the document
            pieces.append(f"Document: '{pdf_context['filename']}' (uploaded during this conversation)")
            pieces.append("---")
            pieces.append("Previous conversation continues below:")
        else:
            # General chat without PDF context
            pieces = [GENERAL_CHAT_PROMPT]
            pieces.append("Continuing our conversation:")
        
        # Add conversation history
        for msg in request.messages:
            prefix = "User:" if msg.role == "user" else "Assistant:"
            pieces.append(f"{prefix} {msg.content}")
        pieces.append("Assistant:")

        prompt = "\n\n".join(pieces)
        
        # More generous prompt size limits for conversation continuity
        MAX_PROMPT_SIZE = 18000 if pdf_context else 10000
        if len(prompt) > MAX_PROMPT_SIZE:
            # If too large, prioritize recent conversation
            if pdf_context:
                # Keep recent messages and truncate PDF content
                recent_conversation = "\n\n".join(pieces[-8:])  # Last 8 exchanges
                truncated_pdf = pdf_context["content"][:6000] + "...(truncated)"
                prompt = CHAT_WITH_PDF_PROMPT.format(pdf_content=truncated_pdf) + "\n\n" + recent_conversation
            else:
                prompt = prompt[-MAX_PROMPT_SIZE:]
            logger.info(f"Prompt truncated to {len(prompt)} characters")
        
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
        pieces = None
        gc.collect()
        
        log_memory("CHAT_SUCCESS")
        logger.info("Chat processing completed successfully")
        
        return {
            "result": result,
            "has_pdf_context": pdf_context is not None,
            "pdf_filename": pdf_context["filename"] if pdf_context else None,
            "session_id": request.session_id,
            "context_maintained": True  # Indicates conversation context was preserved
        }

    except Exception as e:
        log_memory("CHAT_ERROR")
        logger.error(f"Chat processing failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(500, "Chat processing failed")
    finally:
        log_memory("CHAT_CLEANUP")
        gc.collect()

# Add endpoint to switch PDF context mid-conversation
@app.post("/switch-context")
async def switch_context(request: dict):
    """
    Switch PDF context mid-conversation
    """
    session_id = request.get("session_id")
    
    if not session_id or session_id not in pdf_context_store:
        raise HTTPException(404, "Session not found")
    
    context = pdf_context_store[session_id]
    return {
        "message": f"Switched context to '{context['filename']}'",
        "session_id": session_id,
        "filename": context["filename"],
        "summary_preview": context["summary"][:200] + "..." if len(context["summary"]) > 200 else context["summary"]
    }

# Endpoint to get available PDF sessions
@app.get("/sessions")
async def get_sessions():
    """Get available PDF sessions"""
    sessions = []
    for session_id, context in pdf_context_store.items():
        sessions.append({
            "session_id": session_id,
            "filename": context["filename"],
            "timestamp": context["timestamp"],
            "summary_preview": context["summary"][:200] + "..." if len(context["summary"]) > 200 else context["summary"]
        })
    
    # Sort by timestamp (newest first)
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"sessions": sessions}

# Endpoint to clear old sessions
@app.post("/clear-sessions")
async def clear_sessions():
    """Clear all stored PDF sessions"""
    global pdf_context_store
    pdf_context_store.clear()
    gc.collect()
    return {"message": "All PDF sessions cleared"}
