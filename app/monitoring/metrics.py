"""Monitoring and metrics collection for the knowledge assistant."""
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader

class MetricsManager:
    def __init__(self):
        self.meter = metrics.get_meter(__name__)
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Initialize metrics collectors."""
        # LLM metrics
        self.llm_requests = self.meter.create_counter(
            name="knowledge_assistant_llm_requests_total",
            description="Total number of LLM requests"
        )
        
        self.llm_response_time = self.meter.create_histogram(
            name="knowledge_assistant_llm_response_time_seconds",
            description="LLM response time in seconds",
            unit="s",
        )
        
        self.token_usage = self.meter.create_counter(
            name="knowledge_assistant_token_usage_total",
            description="Total number of tokens used",
            attributes={"type": "total"}
        )
        
        # Vector store metrics
        self.vector_search_time = self.meter.create_histogram(
            name="knowledge_assistant_vectorstore_search_time_seconds",
            description="Vector store search time in seconds",
            unit="s",
        )
        
        self.vector_store_size = self.meter.create_up_down_counter(
            name="knowledge_assistant_vectorstore_documents_total",
            description="Total number of documents in vector store"
        )
        
        # Cache metrics
        self.cache_hits = self.meter.create_counter(
            name="knowledge_assistant_cache_hits_total",
            description="Total number of cache hits",
            attributes={"type": "total"}
        )
        
        self.cache_misses = self.meter.create_counter(
            name="knowledge_assistant_cache_misses_total",
            description="Total number of cache misses",
            attributes={"type": "total"}
        )
        
        # Error metrics
        self.error_counter = self.meter.create_counter(
            name="knowledge_assistant_errors_total",
            description="Total number of errors"
        )
    
    def record_llm_request(self, duration: float, tokens: int):
        """Record LLM request metrics."""
        self.llm_requests.add(1)
        self.llm_response_time.record(duration)
        self.token_usage.add(tokens)
    
    def record_vector_search(self, duration: float):
        """Record vector store search metrics."""
        self.vector_search_time.record(duration)
    
    def record_cache_result(self, hit: bool):
        """Record cache hit/miss."""
        if hit:
            self.cache_hits.add(1)
        else:
            self.cache_misses.add(1)
    
    def record_error(self):
        """Record an error occurrence."""
        self.error_counter.add(1)
    
    def update_vector_store_size(self, count: int):
        """Update vector store document count."""
        self.vector_store_size.add(count)
