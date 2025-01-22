"""Configuration management for the LangChain application."""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class LLMConfig:
    """LLM configuration settings."""
    type: str = os.getenv("LLM_TYPE", "ollama")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # OpenAI specific
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Ollama specific
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://ollama.ai-stack:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama2")

@dataclass
class VectorStoreConfig:
    """Vector store configuration settings."""
    connection_string: str = os.getenv(
        "CRATEDB_CONNECTION_STRING", 
        "crate://ai-cratedb.ai-stack:4200"
    )
    cache_table: str = os.getenv("CACHE_TABLE_NAME", "llm_cache")
    semantic_cache_table: str = os.getenv("SEMANTIC_CACHE_TABLE_NAME", "semantic_cache")
    semantic_cache_threshold: float = float(
        os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.8")
    )

@dataclass
class ObservabilityConfig:
    """Observability configuration settings."""
    tempo_endpoint: str = os.getenv(
        "TEMPO_ENDPOINT", 
        "http://tempo.monitoring:4317"
    )
    loki_url: str = os.getenv(
        "LOKI_URL",
        "http://loki.monitoring:3100/loki/api/v1/push"
    )
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

@dataclass
class EmbeddingsConfig:
    """Embeddings configuration settings."""
    endpoint: str = os.getenv(
        "SENTENCE_TRANSFORMERS_ENDPOINT",
        "http://sentence-transformers.ai-stack:8080"
    )

@dataclass
class Config:
    """Main application configuration."""
    llm: LLMConfig = LLMConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()

# Create a global config instance
config = Config()
