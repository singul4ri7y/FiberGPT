#!/usr/bin/env python3
"""
Unified web chat server - serves both UI and API.
Simulates a multi-GPU streaming backend using a Mock Engine.
"""

import argparse
import json
import asyncio
import logging
import random
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass

# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 1
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

# Configuration Mock
parser = argparse.ArgumentParser(description='FiberGPT Web Server')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='Number of (simulated) GPUs')
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Default temperature')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Default top-k')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Default max tokens')
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
args, _ = parser.parse_known_args() # Use known args to avoid conflict with uvicorn if run directly

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- Mock Engine & Components ---

class MockTokenizer:
    def encode(self, text):
        return [ord(c) for c in text] # Simple mock encoding
    
    def decode(self, tokens):
        return "".join([chr(t) for t in tokens])
    
    def encode_special(self, special_token):
        return 0 # Placeholder
    
    def get_bos_token_id(self):
        return 1

class MockEngine:
    def __init__(self):
        pass

    async def generate_mock(self, prompt_tokens, max_new_tokens):
        """Simulates token generation delay."""
        # Generic response text based on prompt length or content, or random
        responses = [
            "Here is a snippet of code demonstrating the concept:",
            "FiberGPT is optimized for efficiency and speed.",
            "I can help you with that! Let's break it down step by step.",
            "The quick brown fox jumps over the lazy dog."
        ]
        
        # Pick a response or generate generic text
        full_text = random.choice(responses) + " " + " ".join(["bla"] * (max_new_tokens // 5))
        
        for char in full_text:
            await asyncio.sleep(0.02) # Simulate inference time per token
            yield ord(char)

@dataclass
class Worker:
    gpu_id: int
    engine: MockEngine
    tokenizer: MockTokenizer

class WorkerPool:
    def __init__(self, num_gpus: int = 1):
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self):
        print(f"Initializing worker pool with {self.num_gpus} (Simulated) GPUs...")
        for gpu_id in range(self.num_gpus):
            print(f"Loading Mock Model on GPU {gpu_id}...")
            # Simulate load time
            await asyncio.sleep(0.5) 
            
            worker = Worker(
                gpu_id=gpu_id,
                engine=MockEngine(),
                tokenizer=MockTokenizer()
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)
        print(f"All {self.num_gpus} workers initialized!")

    async def acquire_worker(self) -> Worker:
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        await self.available_workers.put(worker)

# --- API Models ---

class ChatMessage(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None

# --- Logic ---

def validate_chat_request(request: ChatRequest):
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Too many messages. Max {MAX_MESSAGES_PER_REQUEST}")

    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} is empty")
        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(status_code=400, detail=f"Message {i} too long. Max {MAX_MESSAGE_LENGTH}")
        total_length += msg_length
        
        if message.role not in ["user", "assistant", "system"]:
             raise HTTPException(status_code=400, detail=f"Invalid role in message {i}")

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(status_code=400, detail=f"Conversation too long. Max {MAX_TOTAL_CONVERSATION_LENGTH}")

async def generate_stream(worker: Worker, request: ChatRequest) -> AsyncGenerator[str, None]:
    """Generate mock streaming response."""
    # Determine basic response based on last message (simple logic for demo)
    last_msg = request.messages[-1].content.lower()
    
    response_content = ""
    if "hello" in last_msg:
        response_content = "Hello there! I am FiberGPT, running on a mock backend. How can I help?"
    elif "time" in last_msg:
        response_content = f"The current server time is {time.strftime('%H:%M:%S')}."
    else:
        response_content = f"I received your request: '{request.messages[-1].content}'. As a simulated engine, I am streaming this response back to you token by token."

    # Simulate token-by-token generation
    for char in response_content:
        await asyncio.sleep(0.03) # intentional delay for effect
        # Yield SSE format data
        # JSON structure: {"token": "c", "gpu": 0}
        yield f"data: {json.dumps({'token': char, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
    
    # Done signal
    yield f"data: {json.dumps({'done': True})}\n\n"

# --- FastAPI App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize()
    logger.info(f"Server ready at http://{args.host}:{args.port}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.get("/style.css")
async def get_style():
    return FileResponse("style.css")

@app.get("/script.js")
async def get_script():
    return FileResponse("script.js")

@app.get("/health")
async def health():
    pool = getattr(app.state, 'worker_pool', None)
    return {
        "status": "ok",
        "ready": pool is not None and len(pool.workers) > 0,
        "available_workers": pool.available_workers.qsize() if pool else 0
    }

@app.get("/stats")
async def stats():
    pool = app.state.worker_pool
    return {
        "total_workers": len(pool.workers),
        "available_workers": pool.available_workers.qsize(),
        "busy_workers": len(pool.workers) - pool.available_workers.qsize(),
        "workers": [{"gpu_id": w.gpu_id, "device": "MockDevice"} for w in pool.workers]
    }

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    validate_chat_request(request)
    
    logger.info(f"Incoming Request: {len(request.messages)} messages")
    
    pool = app.state.worker_pool
    worker = await pool.acquire_worker()
    
    try:
        async def stream_wrapper():
            try:
                async for chunk in generate_stream(worker, request):
                    yield chunk
            finally:
                await pool.release_worker(worker)
                logger.info(f"Worker {worker.gpu_id} released.")

        return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
    except Exception as e:
        await pool.release_worker(worker)
        logger.error(f"Error during generation: {e}")
        raise e

if __name__ == "__main__":
    import uvicorn
    print(f"Starting mocked FiberGPT server on port {args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
