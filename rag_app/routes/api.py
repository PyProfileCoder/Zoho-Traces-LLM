"""
API Routes with Framework Integration and Site24x7 Metrics
Comprehensive REST API with framework analytics and observability
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import structlog

# Import services with error handling
try:
    from services.agent_service import agent_service
    from services.enhanced_metrics_service import enhanced_metrics_service
    from core.tracing import tracing_manager
except ImportError as e:
    print(f"⚠️ Import warning in API routes: {e}")
    agent_service = None
    enhanced_metrics_service = None
    tracing_manager = None

logger = structlog.get_logger()

api_bp = Blueprint('api', __name__, url_prefix='/api')

# ============================================================================
# CORE RAG ENDPOINTS
# ============================================================================

@api_bp.route('/generate', methods=['POST'])
def generate():
    """Enhanced generate endpoint with comprehensive framework support"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validate required fields
        required_fields = ['framework', 'model', 'vector_store', 'query']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Validate framework availability
        if agent_service:
            available_configs = agent_service.get_available_configurations()
            if data['framework'] not in available_configs.get('frameworks', []):
                return jsonify({
                    'error': f"Framework '{data['framework']}' not available",
                    'available_frameworks': available_configs.get('frameworks', [])
                }), 400

        # Execute query with enhanced error handling
        if agent_service:
            result = agent_service.execute_query(data)
            
            # Enhance response with framework metadata
            if 'framework_metadata' in result:
                result['framework_info'] = {
                    'type': result['framework_metadata'].get('framework_type'),
                    'complexity_score': result['framework_metadata'].get('complexity_score'),
                    'performance_tier': result['framework_metadata'].get('performance_tier')
                }
            
            return jsonify(result)
        else:
            return jsonify({'error': 'Agent service not available'}), 503

    except Exception as e:
        logger.error("API generate error", error=str(e))
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'endpoint': '/api/generate'
        }), 500

@api_bp.route('/configurations', methods=['GET'])
def get_configurations():
    """Get enhanced configurations with framework metadata"""
    try:
        if agent_service:
            configs = agent_service.get_available_configurations()
            
            # Add API metadata
            configs['api_info'] = {
                'version': '2.0',
                'supported_features': [
                    'framework_selection',
                    'model_selection',
                    'vector_store_selection',
                    'performance_analytics',
                    'quality_assessment'
                ],
                'total_frameworks': len(configs.get('frameworks', [])),
                'total_models': len(configs.get('models', [])),
                'total_vector_stores': len(configs.get('vector_stores', []))
            }
            
            return jsonify(configs)
        else:
            return jsonify({
                'error': 'Agent service not available',
                'fallback_config': {
                    'frameworks': ['mock'],
                    'models': ['mock-model'],
                    'vector_stores': ['mock-store']
                }
            }), 503

    except Exception as e:
        logger.error("API configurations error", error=str(e))
        return jsonify({'error': str(e)}), 500

# ============================================================================
# FRAMEWORK-SPECIFIC ENDPOINTS
# ============================================================================

@api_bp.route('/frameworks', methods=['GET'])
def get_frameworks():
    """Get detailed framework information"""
    try:
        if agent_service:
            configs = agent_service.get_available_configurations()
            framework_details = {}
            
            for framework in configs.get('frameworks', []):
                metadata = configs.get('framework_metadata', {}).get(framework, {})
                framework_details[framework] = {
                    'name': framework,
                    'type': metadata.get('type', 'unknown'),
                    'complexity': metadata.get('complexity', 'medium'),
                    'use_cases': metadata.get('best_use_cases', []),
                    'performance_tier': metadata.get('performance_tier', 'medium'),
                    'description': _get_framework_description(framework)
                }
            
            return jsonify({
                'frameworks': framework_details,
                'total_count': len(framework_details),
                'categories': _get_framework_categories(framework_details)
            })
        else:
            return jsonify({'error': 'Agent service not available'}), 503

    except Exception as e:
        logger.error("API frameworks error", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/frameworks/<framework_name>/performance', methods=['GET'])
def get_framework_performance(framework_name):
    """Get performance metrics for specific framework"""
    try:
        if agent_service:
            performance_data = agent_service.get_framework_performance_metrics()
            framework_perf = performance_data['framework_performance'].get(framework_name)
            
            if framework_perf:
                return jsonify({
                    'framework': framework_name,
                    'performance': framework_perf,
                    'ranking': _get_framework_ranking(framework_name, performance_data),
                    'recommendations': _get_framework_recommendations(framework_name, framework_perf)
                })
            else:
                return jsonify({
                    'framework': framework_name,
                    'message': 'No performance data available',
                    'suggestion': 'Execute some queries to generate performance metrics'
                }), 404
        else:
            return jsonify({'error': 'Agent service not available'}), 503

    except Exception as e:
        logger.error("API framework performance error", error=str(e))
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ENHANCED METRICS ENDPOINTS
# ============================================================================

@api_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """Get comprehensive real-time metrics with framework analytics"""
    try:
        hours = request.args.get('hours', 24, type=int)
        
        if enhanced_metrics_service:
            metrics = enhanced_metrics_service.get_real_time_metrics(hours)
            
            # Add endpoint metadata
            metrics['api_metadata'] = {
                'endpoint': '/api/metrics',
                'time_range_hours': hours,
                'generated_at': datetime.now().isoformat(),
                'includes_framework_analytics': 'framework_analytics' in metrics
            }
            
            return jsonify(metrics)
        else:
            return jsonify({'error': 'Enhanced metrics service not available'}), 503

    except Exception as e:
        logger.error("API metrics error", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/metrics/enhanced', methods=['GET'])
def get_enhanced_metrics():
    """Get enhanced metrics with comprehensive framework analysis"""
    try:
        days = request.args.get('days', 7, type=int)
        
        if enhanced_metrics_service:
            metrics = enhanced_metrics_service.get_enhanced_metrics(days)
            return jsonify(metrics)
        else:
            return jsonify({'error': 'Enhanced metrics service not available'}), 503

    except Exception as e:
        logger.error("API enhanced metrics error", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/metrics/frameworks', methods=['GET'])
def get_framework_analytics():
    """Get comprehensive framework analytics"""
    try:
        days = request.args.get('days', 7, type=int)
        
        if enhanced_metrics_service:
            analytics = enhanced_metrics_service.get_framework_analytics(days)
            return jsonify(analytics)
        else:
            return jsonify({'error': 'Enhanced metrics service not available'}), 503

    except Exception as e:
        logger.error("API framework analytics error", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/metrics/tokens', methods=['GET'])
def get_token_metrics():
    """Get token usage metrics with framework breakdown"""
    try:
        days = request.args.get('days', 7, type=int)
        
        if enhanced_metrics_service:
            token_data = enhanced_metrics_service.get_token_usage_data(days)
            return jsonify(token_data)
        else:
            return jsonify({'error': 'Enhanced metrics service not available'}), 503

    except Exception as e:
        logger.error("API token metrics error", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/metrics/costs', methods=['GET'])
def get_cost_metrics():
    """Get cost metrics with framework and model breakdown"""
    try:
        days = request.args.get('days', 7, type=int)
        model = request.args.get('model', 'gpt-4o-mini')
        
        if enhanced_metrics_service:
            cost_data = enhanced_metrics_service.get_cost_data(days, model)
            return jsonify(cost_data)
        else:
            return jsonify({'error': 'Enhanced metrics service not available'}), 503

    except Exception as e:
        logger.error("API cost metrics error", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/metrics/latency', methods=['GET'])
def get_latency_metrics():
    """Get latency metrics with framework performance analysis"""
    try:
        days = request.args.get('days', 7, type=int)
        
        if enhanced_metrics_service:
            latency_data = enhanced_metrics_service.get_latency_data(days)
            return jsonify(latency_data)
        else:
            return jsonify({'error': 'Enhanced metrics service not available'}), 503

    except Exception as e:
        logger.error("API latency metrics error", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/metrics/models', methods=['GET'])
def get_model_metrics():
    """Get comprehensive model and framework usage breakdown"""
    try:
        if enhanced_metrics_service:
            model_data = enhanced_metrics_service.get_model_usage_breakdown()
            return jsonify(model_data)
        else:
            return jsonify({'error': 'Enhanced metrics service not available'}), 503

    except Exception as e:
        logger.error("API model metrics error", error=str(e))
