# RAG Assistant Bootstrap

A production-ready template for building RAG (Retrieval Augmented Generation) applications with LangChain and CrateDB, featuring:

- 🔍 RAG with CrateDB vector store
- 🤖 Flexible LLM support (OpenAI/Ollama)
- 📊 Observability stack (OpenTelemetry, Prometheus, Loki)
- 💾 Semantic caching for efficiency
- 🚀 Kubernetes deployment ready
- 🔒 Secure configuration management

## Features

- **Production-Ready RAG Implementation**:
  - Document ingestion and processing
  - Vector storage with CrateDB
  - Semantic search and retrieval
  - Context-aware response generation
- **Flexible LLM Integration**: Easy switching between OpenAI and Ollama
- **Production Observability**:
  - Distributed tracing with OpenTelemetry
  - Metrics with Prometheus
  - Logging with Loki
- **Advanced Caching**: Both standard and semantic caching support
- **Kubernetes Ready**: Includes Helm charts and DevSpace configuration

## Quick Start

1. Clone the template:
   ```bash
   git clone https://github.com/yourusername/rag-assistant-bootstrap.git
   cd rag-assistant-bootstrap
   ```

2. Set up environment:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your LLM:
   - For OpenAI:
     ```
     LLM_TYPE=openai
     OPENAI_API_KEY=your-api-key
     OPENAI_MODEL=gpt-4.0-mini
     ```
   - For Ollama:
     ```
     LLM_TYPE=ollama
     OLLAMA_BASE_URL=http://localhost:11434
     OLLAMA_MODEL=llama2
     ```

## Project Structure

```
rag-assistant-bootstrap/
├── app/
│   ├── data/           # Data processing and vector store management
│   ├── chatbot/        # RAG implementation and response generation
│   ├── llms/           # LLM provider management
│   ├── monitoring/     # Observability and metrics
│   ├── config.py       # Configuration management
│   └── main.py         # FastAPI application
├── chart/              # Helm chart for Kubernetes deployment
├── .env.example        # Example environment configuration
├── requirements.txt    # Python dependencies
├── Dockerfile         # Container definition
└── devspace.yaml      # DevSpace configuration
```

## Configuration

All configuration is managed through environment variables. See `.env.example` for available options.

### LLM Configuration
- `LLM_TYPE`: Choose between 'openai' or 'ollama'
- `LLM_TEMPERATURE`: Model temperature (0.0-1.0)
- Model-specific settings in `.env.example`

### Infrastructure Configuration
- Vector store (CrateDB) settings
- Observability endpoints
- Logging configuration

## Development

1. Start local services:
   ```bash
   devspace dev
   ```

2. Run the application:
   ```bash
   python -m app.main
   ```

## Deployment

### Local Kubernetes
```bash
devspace deploy
```

### Production
1. Update values in `chart/values.yaml`
2. Deploy using Helm:
   ```bash
   helm install rag-assistant ./chart
   ```

## Monitoring

- Traces: Available in Tempo
- Metrics: Prometheus endpoints at `/metrics`
- Logs: Aggregated in Loki

## Contributing

1. Fork the repository
2. Create your feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.