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
from typing import Literal, List, Optional, Dict

import pdfplumber

# Selenium imports for search
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

# Correct import for Google's GenAI client:
from google import genai
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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

# Configure Google AI for both services
genai_config.configure(api_key=GOOGLE_API_KEY)
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

# Global variables
request_count = 0
pdf_context_store = {}  # Store PDF content by session/user

# Enhanced system prompts - UPDATED for mid-conversation PDF uploads
PDF_SUMMARY_PROMPT = (
    "You are an expert document analyzer. Provide a DETAILED scrape of this document. "
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
    "If the user asks something not covered in the document, If user asks about document beyond its content, use your knowledge. If not possible, politely inform them.\n"
    "Keep responses concise but informative.\n\n"
    "PDF CONTENT:\n{pdf_content}\n\n"
    "CONVERSATION:\n"
)

GENERAL_CHAT_PROMPT = (
    "You are a helpful assistant. Respond to the user's messages with concise and relevant information."
    "Be friendly and always helpful. Keep replies short and to the point. "
)

# Create the main FastAPI app
app = FastAPI(title="Combined Portfolio API", description="AI Chat and Search API")

# Reduce concurrent processing to save memory
processing_semaphore = asyncio.Semaphore(2)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memory check middleware
@app.middleware("http")
async def memory_check_middleware(request: Request, call_next):
    global request_count
    request_count += 1
    
    log_memory(f"REQUEST_{request_count}_START")
    response = await call_next(request)
    log_memory(f"REQUEST_{request_count}_END")
    
    return response

# Timeout middleware
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        timeout = 45.0 if "/search" in str(request.url) else 30.0
        return await asyncio.wait_for(call_next(request), timeout=timeout)
    except asyncio.TimeoutError:
        log_memory("TIMEOUT_ERROR")
        raise HTTPException(status_code=504, detail="Request timeout")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "combined", "timestamp": time.time()}

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
        "pdf_contexts_stored": len(pdf_context_store),
        "service": "combined"
    }

# AI Chat Models
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatHistoryRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: Optional[str] = None

class UploadResponse(BaseModel):
    result: str
    session_id: str
    filename: str
    message: str
    is_mid_conversation: bool

class ChatResponse(BaseModel):
    result: str
    has_pdf_context: bool
    pdf_filename: Optional[str]
    session_id: Optional[str]
    context_maintained: bool

# Search Models
class SearchRequest(BaseModel):
    query: str
    num_results: Optional[int] = 5

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    rank: int

class SearchResponse(BaseModel):
    query: str
    ai_response: str
    sources: List[SearchResult]
    search_time: float

# AI Chat Helper Functions
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

# Search Engine Class
class AISearchEngine:
    def __init__(self):
        # Configure Google AI model
        self.model = genai_config.GenerativeModel('gemini-2.0-flash')
        
        # Setup Chrome options for minimal footprint
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--disable-images')
        self.chrome_options.add_argument('--disable-javascript')
        self.chrome_options.add_argument('--disable-web-security')
        self.chrome_options.add_argument('--disable-features=VizDisplayCompositor')
        self.chrome_options.add_argument('--disable-extensions')
        self.chrome_options.add_argument('--disable-plugins')
        self.chrome_options.add_argument('--log-level=3')
        self.chrome_options.add_argument('--silent')
        self.chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        self.chrome_options.add_experimental_option('useAutomationExtension', False)
        self.chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    def search_duckduckgo(self, query: str, num_results: int = 5) -> List[Dict]:
        """Quick search without heavy scraping"""
        search_results = []
        driver = None
        
        try:
            log_memory("SELENIUM_START")
            driver = webdriver.Chrome(options=self.chrome_options)
            wait = WebDriverWait(driver, 15)
            
            logger.info(f"Navigating to DuckDuckGo for query: {query}")
            driver.get("https://duckduckgo.com")
            
            # Search
            search_box = wait.until(EC.presence_of_element_located((By.NAME, "q")))
            search_box.clear()
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)
            
            time.sleep(3)  # Wait for results
            log_memory("AFTER_SEARCH")
            
            # Extract results quickly
            results = driver.find_elements(By.CSS_SELECTOR, "[data-testid='result']")
            logger.info(f"Found {len(results)} search result elements")
            
            for i, result in enumerate(results[:num_results]):
                try:
                    title_element = result.find_element(By.CSS_SELECTOR, "h2 a")
                    title = title_element.text.strip()
                    url = title_element.get_attribute("href")
                    
                    # Get snippet without deep scraping
                    snippet = ""
                    try:
                        snippet_element = result.find_element(By.CSS_SELECTOR, "[data-result='snippet']")
                        snippet = snippet_element.text.strip()[:300]  # Limit snippet
                    except:
                        # Try alternative snippet selectors
                        try:
                            snippet_element = result.find_element(By.CSS_SELECTOR, ".result__snippet")
                            snippet = snippet_element.text.strip()[:300]
                        except:
                            pass
                    
                    if title and url:
                        search_results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "rank": i + 1
                        })
                        logger.info(f"Extracted result {i+1}: {title[:50]}...")
                except Exception as e:
                    logger.warning(f"Error extracting result {i}: {e}")
                    continue
            
            log_memory("SEARCH_EXTRACTION_COMPLETE")
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            log_memory("SEARCH_ERROR")
        finally:
            if driver:
                driver.quit()
            log_memory("SELENIUM_CLEANUP")
            gc.collect()
        
        return search_results
    
    def generate_response(self, query: str, search_results: List[Dict]) -> str:
        """Generate AI response using only search results"""
        if not search_results:
            return "No search results found for your query. Please try a different search term."
        
        try:
            log_memory("AI_GENERATION_START")
            
            # Create context from search results only
            context = "Search Results:\n"
            for i, result in enumerate(search_results):
                context += f"[{i+1}] {result['title']}\n"
                context += f"URL: {result['url']}\n"
                if result['snippet']:
                    context += f"Summary: {result['snippet']}\n"
                context += "\n"
            
            prompt = f"""
You are an AI search assistant similar to Perplexity. Based on the search results below, provide a comprehensive and well-structured answer to the user's question.

{context}

User Question: {query}

Instructions:
- Provide a clear, informative answer that directly addresses the user's question
- Use information from the search results provided above
- Include relevant citations using [1], [2], [3], etc. referring to the numbered search results
- Structure your response with proper paragraphs for readability
- Be concise but thorough - aim for 2-4 paragraphs
- If the search results don't fully answer the question, mention what information is available
- Focus on the most relevant and credible information from the sources

Answer:
"""
            
            # Generate response
            response = self.model.generate_content(prompt)
            ai_response = response.text or "Unable to generate response from search results."
            
            # Clean response
            cleaned_response = ai_response.replace("**", "").strip()
            
            log_memory("AI_GENERATION_COMPLETE")
            
            # Clear variables
            prompt = None
            context = None
            response = None
            gc.collect()
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            log_memory("AI_GENERATION_ERROR")
            return f"Error generating AI response: {str(e)}"

# Initialize search engine
search_engine = AISearchEngine()

# AI CHAT ENDPOINTS
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Enhanced PDF upload that can be used mid-conversation"""
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
            return UploadResponse(
                result=cleaned_summary,
                session_id=session_id,
                filename=file.filename,
                message=f"PDF '{file.filename}' uploaded successfully! I can now answer questions about this document.",
                is_mid_conversation=True  # Flag to indicate this can be used mid-conversation
            )

        except Exception as e:
            log_memory("UPLOAD_ERROR")
            logger.error(f"Processing failed: {str(e)}")
            traceback.print_exc()
            raise HTTPException(500, "Processing failed")
        finally:
            log_memory("UPLOAD_CLEANUP")
            gc.collect()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatHistoryRequest):
    """Enhanced chat with PDF context support and session switching"""
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
                return ChatResponse(
                    result=f"Sorry, I couldn't find the PDF session '{request.session_id}'. Please upload the PDF again.",
                    has_pdf_context=False,
                    pdf_filename=None,
                    session_id=request.session_id,
                    context_maintained=False
                )
        
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
        
        return ChatResponse(
            result=result,
            has_pdf_context=pdf_context is not None,
            pdf_filename=pdf_context["filename"] if pdf_context else None,
            session_id=request.session_id,
            context_maintained=True  # Indicates conversation context was preserved
        )

    except Exception as e:
        log_memory("CHAT_ERROR")
        logger.error(f"Chat processing failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(500, "Chat processing failed")
    finally:
        log_memory("CHAT_CLEANUP")
        gc.collect()

# SEARCH ENDPOINTS
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform AI-powered search"""
    log_memory("SEARCH_ENDPOINT_START")
    
    async with processing_semaphore:
        start_time = time.time()
        
        try:
            query = request.query.strip()
            if not query:
                raise HTTPException(400, "Query cannot be empty")
            
            if len(query) > 500:
                raise HTTPException(400, "Query too long. Maximum 500 characters.")
            
            logger.info(f"Processing search request: {query}")
            
            # Get search results
            search_results = search_engine.search_duckduckgo(query, request.num_results)
            
            if not search_results:
                return SearchResponse(
                    query=query,
                    ai_response="No search results found. Please try rephrasing your query or using different keywords.",
                    sources=[],
                    search_time=time.time() - start_time
                )
            
            logger.info(f"Found {len(search_results)} search results")
            
            # Generate AI response
            ai_response = search_engine.generate_response(query, search_results)
            
            # Convert to response format
            sources = [
                SearchResult(
                    title=result["title"],
                    url=result["url"],
                    snippet=result["snippet"],
                    rank=result["rank"]
                )
                for result in search_results
            ]
            
            search_time = time.time() - start_time
            logger.info(f"Search completed in {search_time:.2f} seconds")
            
            log_memory("SEARCH_ENDPOINT_SUCCESS")
            
            return SearchResponse(
                query=query,
                ai_response=ai_response,
                sources=sources,
                search_time=search_time
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Search processing failed: {str(e)}")
            log_memory("SEARCH_ENDPOINT_ERROR")
            raise HTTPException(500, f"Search processing failed: {str(e)}")
        finally:
            log_memory("SEARCH_ENDPOINT_CLEANUP")
            gc.collect()

# ADDITIONAL ENDPOINTS
@app.post("/switch-context")
async def switch_context(request: dict):
    """Switch PDF context mid-conversation"""
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

@app.post("/clear-sessions")
async def clear_sessions():
    """Clear all stored PDF sessions"""
    global pdf_context_store
    pdf_context_store.clear()
    gc.collect()
    return {"message": "All PDF sessions cleared"}

@app.post("/gc")
async def force_garbage_collection():
    """Force garbage collection - use only for debugging"""
    log_memory("BEFORE_FORCED_GC")
    collected = gc.collect()
    log_memory("AFTER_FORCED_GC")
    return {"collected_objects": collected, "message": "Garbage collection completed", "service": "combined"}

# Cleanup task for expired sessions
async def cleanup_expired_sessions():
    """Periodically clean up expired PDF sessions"""
    while True:
        try:
            current_time = time.time()
            expired_sessions = []
            
            for session_id, context in pdf_context_store.items():
                # Remove sessions older than 2 hours
                if current_time - context["timestamp"] > 7200:  # 2 hours
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del pdf_context_store[session_id]
                logger.info(f"Cleaned up expired session: {session_id}")
            
            if expired_sessions:
                gc.collect()
                log_memory("SESSION_CLEANUP")
            
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
        
        # Wait 30 minutes before next cleanup
        await asyncio.sleep(1800)

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(cleanup_expired_sessions())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)