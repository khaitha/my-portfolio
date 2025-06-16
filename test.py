import io
import os
import traceback
import warnings
import time
import gc
import logging
import asyncio
import psutil
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List, Optional, Dict, Any, Tuple

import pdfplumber

# HTTP-based imports for search
import requests
import urllib.parse
from bs4 import BeautifulSoup

# Correct import for Google's GenAI client:
import google.generativeai as genai_config

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress pdfplumber CropBox warnings
warnings.filterwarnings("ignore", message="CropBox missing from /Page")

# Load your Google API key from the environment
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
#if not GOOGLE_API_KEY:
   # raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

# Configure Google AI for both services
#genai_config.configure(api_key=GOOGLE_API_KEY)

# Create a shared model instance for chat and upload functionality
chat_model = genai_config.GenerativeModel('gemini-2.0-flash')
def create_intelligent_chunks(pdf_text: str, chunk_size: int = 2000, overlap: int = 200) -> List[Dict]:
    """Create overlapping chunks with metadata"""
    chunks = []
    start = 0
    chunk_id = 1
    
    while start < len(pdf_text):
        end = start + chunk_size
        
        # Try to break at sentence boundaries near the target size
        if end < len(pdf_text):
            # Look for sentence end within last 200 chars
            sentence_end = pdf_text.rfind('.', end - 200, end)
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk_text = pdf_text[start:end].strip()
        
        if chunk_text:
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "start_pos": start,
                "end_pos": end,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split())
            })
            chunk_id += 1
        
        # Move start position with overlap
        start = end - overlap if end < len(pdf_text) else end
        
        if start >= len(pdf_text):
            break
    
    return chunks
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

print(extract_text_from_pdf(open("paper.pdf", "rb").read()))