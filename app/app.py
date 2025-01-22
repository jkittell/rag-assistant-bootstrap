from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import CrateDB
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.cache import CrateDBCache, CrateDBSemanticCache
from langchain.globals import set_llm_cache
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.prometheus import PrometheusMetricReader
import logging
import logging_loki
import time
import socket
import os
from dotenv import load_dotenv
from app.config import config

# Load environment variables
load_dotenv()

# Set up Loki handler
loki_handler = logging_loki.LokiHandler(
    url=config.observability.loki_url,
    tags={
        "service": "rag-assistant-bootstrap",
        "host": socket.gethostname()
    },
    version="1",
)

# Configure logging
logging.basicConfig(
    level=config.observability.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.addHandler(loki_handler)

# Set up tracing (goes to Tempo)
tracer_provider = TracerProvider()
otlp_exporter = OTLPSpanExporter(
    endpoint=config.observability.tempo_endpoint,
    insecure=True
)
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(tracer_provider)
LangchainInstrumentor().instrument()

# Set up metrics (goes to Prometheus)
metrics.set_meter_provider(MeterProvider())
meter = metrics.get_meter(__name__)

# Create metrics
llm_requests = meter.create_counter(
    name="langchain_llm_requests_total",
    description="Total number of LLM requests"
)

llm_response_time = meter.create_histogram(
    name="langchain_llm_response_time_seconds",
    description="LLM response time in seconds",
    unit="s",
)

vector_search_time = meter.create_histogram(
    name="langchain_vectorstore_search_time_seconds",
    description="Vector store search time in seconds",
    unit="s",
)

error_counter = meter.create_counter(
    name="langchain_errors_total",
    description="Total number of errors"
)

token_usage = meter.create_counter(
    name="langchain_token_usage_total",
    description="Total number of tokens used",
    attributes={"type": "total"}
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_endpoint=config.embeddings.endpoint
)

# Initialize standard cache (exact matches)
standard_cache = CrateDBCache(
    connection_string=config.vector_store.connection_string,
    table_name=config.vector_store.cache_table
)

# Initialize semantic cache (similar queries)
semantic_cache = CrateDBSemanticCache(
    connection_string=config.vector_store.connection_string,
    table_name=config.vector_store.semantic_cache_table,
    embedding=embeddings,
    score_threshold=config.vector_store.semantic_cache_threshold
)

# Set the standard cache as default
set_llm_cache(standard_cache)

def initialize_llm():
    """Initialize LLM based on configuration."""
    if config.llm.type.lower() == "openai":
        return ChatOpenAI(
            model=config.llm.openai_model,
            openai_api_key=config.llm.openai_api_key,
            temperature=config.llm.temperature
        )
    else:  # default to ollama
        return Ollama(
            base_url=config.llm.ollama_base_url,
            model=config.llm.ollama_model,
            temperature=config.llm.temperature
        )

# Initialize components
llm = initialize_llm()

# Connect to CrateDB for vector storage
vectorstore = CrateDB(
    connection_string=config.vector_store.connection_string,
    embeddings=embeddings,
    table_name="documents"
)

# Create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

async def process_query(query: str):
    start_time = time.time()
    tracer = trace.get_tracer(__name__)
    
    try:
        with tracer.start_as_current_span("process_query") as span:
            span.set_attribute("query", query)
            
            # First check standard cache (exact matches)
            with tracer.start_span("standard_cache_lookup") as cache_span:
                cached_result = standard_cache.lookup(query)
                if cached_result:
                    logger.info(f"Standard cache hit for query: {query}")
                    process_time = time.time() - start_time
                    store_interaction(query, cached_result, process_time, cache_type="standard")
                    return cached_result

            # Then check semantic cache (similar queries)
            with tracer.start_span("semantic_cache_lookup") as cache_span:
                cached_result = semantic_cache.lookup(query)
                if cached_result:
                    logger.info(f"Semantic cache hit for query: {query}")
                    process_time = time.time() - start_time
                    store_interaction(query, cached_result, process_time, cache_type="semantic")
                    return cached_result

            # If no cache hit, proceed with normal processing
            llm_requests.add(1)
            
            with tracer.start_span("qa_chain") as qa_span:
                result = qa_chain({"query": query})
                answer = result["result"]
            
            process_time = time.time() - start_time
            llm_response_time.record(process_time)
            
            # Store in both caches
            with tracer.start_span("cache_store") as cache_span:
                standard_cache.update(query, answer)
                semantic_cache.update(query, answer)
            
            store_interaction(query, answer, process_time, cache_type="none")
            return answer
            
    except Exception as e:
        error_counter.add(1)
        logger.error(f"Error processing query: {str(e)}")
        raise

def store_interaction(query, result, process_time, cache_type="none"):
    """Store interaction details in CrateDB"""
    try:
        with trace.get_tracer(__name__).start_span("store_interaction") as span:
            vectorstore.client.execute("""
                INSERT INTO interactions (
                    query, 
                    response, 
                    process_time,
                    cache_type,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                query,
                str(result),
                process_time,
                cache_type,
                time.time()
            ))
            logger.info(f"Stored interaction with cache_type: {cache_type}")
    except Exception as e:
        logger.error(f"Error storing interaction: {str(e)}")
        error_counter.add(1)