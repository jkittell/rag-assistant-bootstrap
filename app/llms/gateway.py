"""LLM Gateway for managing different LLM providers and configurations."""
from typing import Optional
from langchain.chat_models import ChatOpenAI
from langchain.llms import Ollama
from langchain.cache import CrateDBCache, CrateDBSemanticCache
from langchain.globals import set_llm_cache
from opentelemetry import trace

class LLMGateway:
    def __init__(self, config, embeddings):
        self.config = config
        self.embeddings = embeddings
        self.tracer = trace.get_tracer(__name__)
        
        # Initialize caches
        self._setup_caches()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
    
    def _setup_caches(self):
        """Setup standard and semantic caches."""
        # Standard cache for exact matches
        standard_cache = CrateDBCache(
            connection_string=self.config.vector_store.connection_string,
            table_name=self.config.vector_store.cache_table
        )
        
        # Semantic cache for similar queries
        semantic_cache = CrateDBSemanticCache(
            connection_string=self.config.vector_store.connection_string,
            table_name=self.config.vector_store.semantic_cache_table,
            embedding=self.embeddings,
            score_threshold=self.config.vector_store.semantic_cache_threshold
        )
        
        # Set standard cache as default
        set_llm_cache(standard_cache)
        self.semantic_cache = semantic_cache
    
    def _initialize_llm(self):
        """Initialize LLM based on configuration."""
        if self.config.llm.type.lower() == "openai":
            return ChatOpenAI(
                model=self.config.llm.openai_model,
                openai_api_key=self.config.llm.openai_api_key,
                temperature=self.config.llm.temperature
            )
        else:  # default to ollama
            return Ollama(
                base_url=self.config.llm.ollama_base_url,
                model=self.config.llm.ollama_model,
                temperature=self.config.llm.temperature
            )
    
    async def get_completion(self, prompt: str) -> str:
        """Get completion from LLM with caching."""
        with self.tracer.start_span("llm_completion") as span:
            span.set_attribute("prompt", prompt)
            
            # Check semantic cache
            cached_response = self.semantic_cache.lookup(prompt)
            if cached_response:
                span.set_attribute("cache_hit", "semantic")
                return cached_response
            
            # Get response from LLM
            response = await self.llm.agenerate([prompt])
            
            # Update caches
            self.semantic_cache.update(prompt, response.generations[0][0].text)
            
            return response.generations[0][0].text
