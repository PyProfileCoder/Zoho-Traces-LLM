"""
Simplified Tracing Module
Provides centralized tracing interface
"""

import logging
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)

class TracingManager:
    """Simple tracing manager for backward compatibility"""
    
    def __init__(self):
        self._traces: Dict[str, Dict] = {}
        self._lock = Lock()
        
    def start_trace(self, request_data: Dict[str, Any]) -> str:
        """Start a new trace"""
        trace_id = str(uuid.uuid4())
        
        with self._lock:
            self._traces[trace_id] = {
                'trace_id': trace_id,
                'start_time': time.time(),
                'timestamp': datetime.now().isoformat(),
                'request_data': request_data,
                'steps': [],
                'status': 'running'
            }
        
        logger.info(f"📝 Started trace: {trace_id}")
        return trace_id
    
    def add_step(self, trace_id: str, step_name: str, data: Dict[str, Any]):
        """Add a step to the trace"""
        if trace_id not in self._traces:
            return
            
        with self._lock:
            self._traces[trace_id]['steps'].append({
                'step_name': step_name,
                'timestamp': datetime.now().isoformat(),
                'data': data
            })
    
    def end_trace(self, trace_id: str, status: str, error: str = None):
        """End a trace"""
        if trace_id not in self._traces:
            return
            
        with self._lock:
            trace = self._traces[trace_id]
            trace['end_time'] = time.time()
            trace['duration'] = trace['end_time'] - trace['start_time']
            trace['status'] = status
            if error:
                trace['error'] = error
        
        logger.info(f"✅ Ended trace: {trace_id} - {status}")
    
    def get_trace(self, trace_id: str) -> Optional[Dict]:
        """Get a specific trace"""
        return self._traces.get(trace_id)
    
    def get_all_traces(self) -> List[Dict]:
        """Get all traces"""
        with self._lock:
            return list(self._traces.values())
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for backward compatibility"""
        with self._lock:
            total_traces = len(self._traces)
            completed = len([t for t in self._traces.values() if t['status'] == 'completed'])
            failed = len([t for t in self._traces.values() if t['status'] == 'failed'])
            
            return {
                'session_id': 'tracing-session',
                'total_requests': total_traces,
                'completed_requests': completed,
                'failed_requests': failed,
                'success_rate': (completed / max(total_traces, 1)) * 100,
                'average_duration': 0,
                'total_tokens_used': 0,
                'active_requests': len([t for t in self._traces.values() if t['status'] == 'running'])
            }

# Global tracing manager instance
tracing_manager = TracingManager()

def initialize_tracing(config=None):
    """Initialize tracing - backward compatibility function"""
    logger.info("✅ Tracing manager initialized")
    return tracing_manager

def get_tracer():
    """Get tracer - backward compatibility function"""
    return tracing_manager
