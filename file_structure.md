# ABC Project File Structure

```
abc/
├── .DS_Store
├── .env
├── environment.yml
├── file_structure.md
└── rag_app/
    ├── .DS_Store
    ├── Dockerfile
    ├── api_server.py
    ├── rag_orchestrator.py
    ├── run_api_server.py
    ├── run_tests.py
    ├── requirements.txt
    ├── requirements_api.txt
    ├── docker-compose.yml
    ├── api_documentation.md
    ├── adapters/
    │   ├── __init__.py
    │   ├── autogen_adapter.py
    │   ├── dspy_adapter.py
    │   ├── langgraph_adapter.py
    │   └── llamaindex_adapter.py
    ├── config/
    │   ├── core_config.py
    │   └── settings.py
    ├── core/
    │   ├── registry.py
    │   └── tracing.py
    ├── monitoring/
    │   ├── .DS_Store
    │   ├── alert_rules.yml
    │   ├── prometheus.yml
    │   └── grafana/
    │       ├── .DS_Store
    │       ├── dashboards/
    │       │   ├── dashboard.yml
    │       │   ├── docker-agent.json
    │       │   └── real-time-metrics.json
    │       └── datasources/
    │           ├── prometheus.yml
    │           └── site24x7.yml
    ├── nginx/
    │   └── nginx.conf
    ├── providers/
    │   ├── .DS_Store
    │   ├── .vectordb.yml
    │   ├── __init__.py
    │   ├── framework_factory.py
    │   ├── llm_factory.py
    │   ├── langchain_framework.py
    │   ├── llamaindex_framework.py
    │   ├── autogen_framework.py
    │   ├── crewai_framework.py
    │   ├── neo4j_framework.py
    │   ├── dspy_framework.py
    │   ├── cleanlab_framework.py
    │   ├── aws_bedrock_framework.py
    │   ├── litellm_framework.py
    │   ├── vercel_framework.py
    │   └── vector_store_factory.py
    ├── routes/
    │   ├── api.py
    │   └── web.py
    ├── services/
    │   ├── agent_service.py
    │   ├── enhanced_metrics_service.py
    │   ├── metrics_collector.py
    │   └── metrics_service.py
    ├── static/
    │   ├── css/
    │   │   └── style.css
    │   └── js/
    │       └── app.js
    ├── templates/
    │   ├── base.html
    │   ├── enhanced_metrics.html
    │   ├── logs.html
    │   ├── metrics.html
    │   ├── playground.html
    │   ├── real_time_dashboard.html
    │   ├── traces.html
    │   └── traces_detail.html
    └── tests/
        ├── .DS_Store
        ├── __init__.py
        ├── conftest.py
        ├── api/
        │   └── test_api_endpoints.py
        └── providers/
            ├── test_frameworks.py
            ├── test_llm.py
            └── test_vectordb.py
```

## Project Overview

This is a Flask-based RAG (Retrieval-Augmented Generation) application with comprehensive monitoring and observability features, organized under the `rag_app` directory.

### Main Application Files

- **`api_server.py`** - API server implementation
- **`rag_orchestrator.py`** - RAG orchestrator with comprehensive error handling
- **`run_api_server.py`** - Script to run the API server
- **`run_tests.py`** - Test runner script

### Configuration & Dependencies

- **`.env`** - Environment variables configuration
- **`requirements.txt`** & **`requirements_api.txt`** - Python dependencies
- **`environment.yml`** - Conda environment configuration
- **`Dockerfile`** & **`docker-compose.yml`** - Containerization setup

### Core Architecture

- **`adapters/`** - Framework adapters (LangGraph, DSPy, AutoGen, LlamaIndex)
- **`providers/`** - Factory classes for LLM, vector database, and framework providers
- **`services/`** - Business logic services (metrics, agents)
- **`routes/`** - Web and API route handlers
- **`core/`** - Core functionality (tracing, registry)

### Web Interface

- **`templates/`** - HTML templates for web dashboard
- **`static/`** - CSS and JavaScript assets
- **`nginx/`** - Nginx configuration

### Monitoring & Observability

- **`monitoring/`** - Prometheus, Grafana, and Site24x7 configurations
- **`core/tracing.py`** - Tracing functionality

### Testing

- **`tests/`** - Comprehensive test suite with organized structure
  - **`api/`** - API endpoint tests
  - **`providers/`** - Provider-specific tests
  - **`conftest.py`** - Pytest configuration

This is a production-ready RAG application with comprehensive monitoring, observability, and a web interface for managing and testing different AI frameworks and LLM providers.
