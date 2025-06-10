import os
import time
import gc
import logging
import asyncio
import psutil
import warnings
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

# Correct import for Google's GenAI client (same as ai.py):
from google import genai

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", message="CropBox missing from /Page")

# Load Google API key from the environment (same as ai.py)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

# Instantiate the client correctly (same as ai.py):
client = genai.Client(api_key=GOOGLE_API_KEY)

# Memory monitoring helper functions (same as ai.py)
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

app = FastAPI()

# Reduce concurrent processing to save memory (same as ai.py)
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

# Memory check middleware (same as ai.py)
@app.middleware("http")
async def memory_check_middleware(request: Request, call_next):
    global request_count
    request_count += 1
    
    # Log memory at start of request
    log_memory(f"SEARCH_REQUEST_{request_count}_START")
    
    response = await call_next(request)
    
    # Log memory after request
    log_memory(f"SEARCH_REQUEST_{request_count}_END")
    
    return response

# Timeout middleware (same as ai.py)
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=45.0)  # Longer timeout for search
    except asyncio.TimeoutError:
        log_memory("SEARCH_TIMEOUT_ERROR")
        raise HTTPException(status_code=504, detail="Search request timeout")

# Health check endpoint (same as ai.py)
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

# Memory monitoring endpoint (same as ai.py)
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
        "service": "search"
    }

# Force garbage collection endpoint (same as ai.py)
@app.post("/gc")
async def force_garbage_collection():
    """Force garbage collection - use only for debugging"""
    log_memory("BEFORE_FORCED_GC")
    collected = gc.collect()
    log_memory("AFTER_FORCED_GC")
    return {"collected_objects": collected, "message": "Garbage collection completed"}

# Pydantic models
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

def clean_response(raw: str) -> str:
    """Clean up AI response (same as ai.py)"""
    cleaned = raw.replace("**", "").strip()
    return cleaned

class AISearchEngine:
    def __init__(self):
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
            
            log_memory("BEFORE_SEARCH_AI_CALL")
            
            # Generate response using the same client as ai.py
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            
            log_memory("AFTER_SEARCH_AI_CALL")
            
            ai_response = response.text or "Unable to generate response from search results."
            
            # Clean response (same as ai.py)
            cleaned_response = clean_response(ai_response)
            
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

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform AI-powered search
    """
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)