"""
Observability and Tracing Setup Module
Provides centralized observability management with Site24x7 integration
"""

import os
import logging
import structlog
from typing import Dict, Any, Optional
from datetime import datetime
import requests
import json

logger = structlog.get_logger(__name__)

class ObservabilityManager:
    """Centralized observability manager with Site24x7 integration"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.is_initialized = False
        self.tracer = None
        self.site24x7_api_key = os.getenv('SITE24X7_API_KEY')
        self.site24x7_endpoint = "https://www.site24x7.com/api/traces"
        
    def initialize(self):
        """Initialize observability components"""
        try:
            self._setup_tracing()
            self._setup_site24x7()
            self.is_initialized = True
            logger.info("✅ Observability initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Observability initialization failed: {e}")
            self.is_initialized = False
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            
            # Set up tracer provider
            trace.set_tracer_provider(TracerProvider())
            self.tracer = trace.get_tracer(__name__)
            
            # Setup OTLP exporter if configured
            otlp_endpoint = os.getenv('OTLP_ENDPOINT')
            if otlp_endpoint:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                span_processor = BatchSpanProcessor(otlp_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)
                
        except ImportError:
            logger.warning("OpenTelemetry not available, using mock tracer")
            self.tracer = MockTracer()
    
    def _setup_site24x7(self):
        """Setup Site24x7 integration"""
        if self.site24x7_api_key:
            logger.info("✅ Site24x7 integration configured")
        else:
            logger.warning("⚠️ Site24x7 API key not configured")
    
    def send_to_site24x7(self, trace_data: Dict[str, Any]):
        """Send trace data to Site24x7"""
        if not self.site24x7_api_key:
            return
            
        try:
            headers = {
                'Authorization': f'Bearer {self.site24x7_api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                self.site24x7_endpoint,
                headers=headers,
                json=trace_data,
                timeout=5
            )
            
            if response.status_code == 200:
                logger.debug("📊 Trace sent to Site24x7 successfully")
            else:
                logger.warning(f"⚠️ Site24x7 API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Failed to send to Site24x7: {e}")

class MockTracer:
    """Mock tracer for when OpenTelemetry is not available"""
    
    def start_as_current_span(self, name: str):
        return MockSpan(name)

class MockSpan:
    """Mock span for testing"""
    
    def __init__(self, name: str):
        self.name = name
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def set_attribute(self, key: str, value: Any):
        pass

class RAGObservability:
    """RAG-specific observability wrapper"""
    
    def __init__(self, obs_manager: ObservabilityManager):
        self.obs_manager = obs_manager
        
    def trace_rag_query(self, func):
        """Decorator for tracing RAG queries"""
        def wrapper(*args, **kwargs):
            if not self.obs_manager.is_initialized:
                return func(*args, **kwargs)
                
            with self.obs_manager.tracer.start_as_current_span("rag_query") as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("status", "success")
                    
                    # Send to Site24x7
                    trace_data = {
                        "timestamp": datetime.now().isoformat(),
                        "service": "rag-system",
                        "operation": func.__name__,
                        "status": "success",
                        "metadata": {
                            "args_count": len(args),
                            "kwargs_keys": list(kwargs.keys())
                        }
                    }
                    self.obs_manager.send_to_site24x7(trace_data)
                    
                    return result
                except Exception as e:
                    span.set_attribute("status", "error")
                    span.set_attribute("error", str(e))
                    raise
        return wrapper
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get observability metrics"""
        return {
            "initialized": self.obs_manager.is_initialized,
            "site24x7_configured": bool(self.obs_manager.site24x7_api_key),
            "tracer_active": self.obs_manager.tracer is not None
        }
