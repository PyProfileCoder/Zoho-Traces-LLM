"""
🚀 ULTIMATE FLASK RAG API SERVER - PRODUCTION READY
Enterprise-grade Flask API with comprehensive RAG integration
"""

import os
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

# Flask and extensions
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

# Validation and serialization
from pydantic import BaseModel, ValidationError, Field

# Observability and monitoring
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our backend modules with error handling
try:
    from rag_orchestrator import create_rag_orchestrator
    from config.core_config import config_manager
    from config.settings import settings
    from observability.trace_setup import ObservabilityManager, RAGObservability
    from core.tracing import tracing_manager
    from services.agent_service import agent_service
    from services.enhanced_metrics_service import enhanced_metrics_service
    from routes.api import api_bp
    from routes.web import web_bp
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    # Create minimal components for testing
    tracing_manager = None
    agent_service = None
    enhanced_metrics_service = None

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus Metrics
REQUEST_COUNT = Counter('flask_rag_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('flask_rag_request_duration_seconds', 'Request duration', ['endpoint'])
ACTIVE_REQUESTS = Gauge('flask_rag_active_requests', 'Active requests')
RAG_QUERIES = Counter('rag_queries_total', 'Total RAG queries', ['provider', 'framework', 'status'])
RAG_LATENCY = Histogram('rag_query_latency_seconds', 'RAG query latency', ['provider', 'framework'])
ERROR_COUNT = Counter('flask_rag_errors_total', 'Total errors', ['error_type'])

# Global application state
app_state = {
    "observability_manager": None,
    "rag_observability": None,
    "executor": None,
    "startup_time": datetime.now(),
}

# Pydantic Models for Request/Response Validation
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000, description="The query to process")
    llm_provider: Optional[str] = Field(default="openai", description="LLM provider to use")
    framework: Optional[str] = Field(default="langchain", description="Framework to use")
    vector_store: Optional[str] = Field(default="faiss", description="Vector store to use")
    model_name: Optional[str] = Field(default="gpt-4o", description="Model name to use")

class QueryResponse(BaseModel):
    query_id: str = Field(..., description="Unique query identifier")
    answer: str = Field(..., description="Generated answer")
    metadata: Dict[str, Any] = Field(..., description="Query metadata")
    trace_id: Optional[str] = Field(default=None, description="Trace identifier")
    processing_time: float = Field(..., description="Processing time in seconds")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component health status")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: datetime = Field(..., description="Error timestamp")
    trace_id: Optional[str] = Field(default=None, description="Trace identifier")

def create_app() -> Flask:
    """🏗️ Create and configure the Flask application with all integrations"""
    app = Flask(__name__)

    # Configure Flask
    app.config.update({
        'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-secret-change-in-production'),
        'MAX_CONTENT_LENGTH': 50 * 1024 * 1024,  # 50MB max file size
        'JSON_SORT_KEYS': False,
        'JSONIFY_PRETTYPRINT_REGULAR': True
    })

    # Proxy fix for production deployment
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    # CORS configuration
    CORS(app,
         origins=["http://localhost:3000", "http://localhost:5000", "http://127.0.0.1:5000"],
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
         supports_credentials=True)

    # Rate limiting
    limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["1000 per hour", "100 per minute"],
    storage_uri=os.environ.get('REDIS_URL', 'memory://')
)

    # Initialize observability
    setup_observability(app)

    # Initialize backend components
    initialize_backend_components()

    # Register middleware and routes
    register_middleware(app)
    register_error_handlers(app)
    register_routes(app, limiter)

    logger.info("🚀 Flask RAG API Server initialized successfully")
    return app

def setup_observability(app: Flask):
    """📊 Setup comprehensive observability and tracing"""
    try:
        # Initialize our custom observability
        if config_manager:
            app_state["observability_manager"] = ObservabilityManager(config_manager)
            app_state["rag_observability"] = RAGObservability(app_state["observability_manager"])
            app_state["observability_manager"].initialize()
            logger.info("✅ Observability setup completed")
    except Exception as e:
        logger.warning(f"⚠️ Observability setup failed: {e}")
        # Create mock observability to prevent errors
        app_state["observability_manager"] = type('MockObs', (), {'is_initialized': False})()
        app_state["rag_observability"] = type('MockRAGObs', (), {
            'trace_rag_query': lambda self, func: func
        })()

def initialize_backend_components():
    """🔧 Initialize backend components"""
    try:
        # Initialize thread pool for async operations
        app_state["executor"] = ThreadPoolExecutor(max_workers=10, thread_name_prefix="rag-worker")
        logger.info("✅ Backend components initialized successfully")
    except Exception as e:
        logger.error(f"❌ Backend initialization failed: {e}")

def register_middleware(app: Flask):
    """🛡️ Register middleware for request processing"""
    @app.before_request
    def before_request():
        """Pre-request processing"""
        g.start_time = time.time()
        g.request_id = str(uuid.uuid4())
        g.trace_id = request.headers.get('X-Trace-ID', g.request_id)
        ACTIVE_REQUESTS.inc()

    @app.after_request
    def after_request(response):
        """Post-request processing"""
        duration = time.time() - g.start_time
        ACTIVE_REQUESTS.dec()
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.endpoint or 'unknown',
            status=response.status_code
        ).inc()
        REQUEST_DURATION.labels(endpoint=request.endpoint or 'unknown').observe(duration)
        
        # Add response headers
        response.headers['X-Request-ID'] = g.request_id
        response.headers['X-Response-Time'] = f"{duration:.3f}s"
        response.headers['X-API-Version'] = "1.0.0"
        
        return response

def register_error_handlers(app: Flask):
    """🚨 Register comprehensive error handlers"""
    @app.errorhandler(ValidationError)
    def handle_validation_error(e):
        """Handle Pydantic validation errors"""
        ERROR_COUNT.labels(error_type='validation').inc()
        error_response = ErrorResponse(
            error="Validation failed",
            error_code="VALIDATION_ERROR",
            timestamp=datetime.now(),
            trace_id=getattr(g, 'trace_id', None)
        )
        return jsonify(error_response.dict()), 400

    @app.errorhandler(500)
    def handle_internal_error(e):
        """Handle internal server errors"""
        ERROR_COUNT.labels(error_type='internal').inc()
        error_response = ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.now(),
            trace_id=getattr(g, 'trace_id', None)
        )
        return jsonify(error_response.dict()), 500

def register_routes(app: Flask, limiter):
    """🛣️ Register all API routes"""
    
    # Register blueprints if available
    try:
        if 'api_bp' in globals():
            app.register_blueprint(api_bp)
        if 'web_bp' in globals():
            app.register_blueprint(web_bp)
    except Exception as e:
        logger.warning(f"⚠️ Could not register blueprints: {e}")

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """🏥 Comprehensive health check endpoint"""
        try:
            components = {
                "agent_service": "healthy" if agent_service else "unavailable",
                "metrics_service": "healthy" if enhanced_metrics_service else "unavailable",
                "tracing": "healthy" if tracing_manager else "unavailable",
                "observability": "healthy" if app_state["observability_manager"] else "unavailable"
            }
            
            uptime = (datetime.now() - app_state["startup_time"]).total_seconds()
            
            response = HealthResponse(
                status="healthy" if all(status != "unavailable" for status in components.values()) else "degraded",
                timestamp=datetime.now(),
                version="1.0.0",
                components=components
            )
            return jsonify(response.dict())
            
        except Exception as e:
            logger.error("❌ Health check failed", error=str(e))
            error_response = HealthResponse(
                status="unhealthy",
                timestamp=datetime.now(),
                version="1.0.0",
                components={"error": str(e)}
            )
            return jsonify(error_response.dict()), 503

    @app.route('/api/query', methods=['POST'])
    @limiter.limit("50 per minute")
    def process_query():
        """🧠 Main RAG query processing endpoint"""
        try:
            # Validate request
            request_data = QueryRequest(**request.get_json())
            
            if not agent_service:
                raise RuntimeError("Agent service not available")

            start_time = time.time()
            
            # Process query
            result = agent_service.execute_query(request_data.dict())
            processing_time = time.time() - start_time

            # Prepare response
            query_id = str(uuid.uuid4())
            response = QueryResponse(
                query_id=query_id,
                answer=result.get('answer', ''),
                metadata={
                    **result.get('metadata', {}),
                    'request_params': request_data.dict(),
                    'processing_time': processing_time
                },
                trace_id=g.trace_id,
                processing_time=processing_time
            )

            return jsonify(response.dict())

        except Exception as e:
            logger.error("❌ Query processing failed", error=str(e))
            error_response = ErrorResponse(
                error=f"Query processing failed: {str(e)}",
                error_code="QUERY_ERROR",
                timestamp=datetime.now(),
                trace_id=g.trace_id
            )
            return jsonify(error_response.dict()), 500

    @app.route('/metrics', methods=['GET'])
    def prometheus_metrics():
        """📊 Prometheus metrics endpoint"""
        from prometheus_client import generate_latest
        return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001)
