"""FastAPI application for the knowledge assistant."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from app.config import config
from app.data.processor import DataProcessor
from app.chatbot.backend import ChatbotBackend
from app.llms.gateway import LLMGateway
from app.monitoring.metrics import MetricsManager
from langchain.embeddings import HuggingFaceEmbeddings

app = FastAPI(title="Knowledge Assistant")

# Initialize components
embeddings = HuggingFaceEmbeddings(model_endpoint=config.embeddings.endpoint)
data_processor = DataProcessor(config)
llm_gateway = LLMGateway(config, embeddings)
chatbot = ChatbotBackend(config, llm_gateway.llm, data_processor)
metrics = MetricsManager()

class QueryRequest(BaseModel):
    query: str

class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    metadata: List[Dict[str, Any]]

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base."""
    try:
        response = await chatbot.process_input(request.query)
        return response
    except Exception as e:
        metrics.record_error()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_document(request: DocumentRequest):
    """Ingest a new document into the knowledge base."""
    try:
        await data_processor.process_text(request.content, request.metadata)
        return {"status": "success"}
    except Exception as e:
        metrics.record_error()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/directory")
async def ingest_directory(directory_path: str):
    """Ingest all documents from a directory."""
    try:
        await data_processor.process_directory(directory_path)
        return {"status": "success"}
    except Exception as e:
        metrics.record_error()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
