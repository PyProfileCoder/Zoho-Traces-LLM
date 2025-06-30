"""
Enhanced Metrics Service with Framework Integration
Comprehensive metrics collection with Site24x7 integration and framework-specific analytics
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import structlog
import os
import json
import requests
from .metrics_collector import metrics_collector, MetricData

logger = structlog.get_logger()

class Site24x7Publisher:
    """Site24x7 metrics publisher with enhanced framework data"""
    
    def __init__(self, api_key: str):
        self.endpoint = "https://www.site24x7.com/api/metrics"
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "RAG-Observability-Service/1.0"
        })

    def publish(self, metrics: Dict[str, Any]):
        """Publish metrics to Site24x7 with enhanced error handling"""
        try:
            # Enhance metrics with framework-specific data
            enhanced_metrics = self._enhance_metrics_for_site24x7(metrics)
            
            response = self.session.post(
                self.endpoint,
                json={"metrics": enhanced_metrics, "timestamp": datetime.now().isoformat()},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✅ Metrics published to Site24x7 successfully")
            else:
                logger.warning(f"⚠️ Site24x7 API returned status {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Site24x7 publishing error: {str(e)}")

    def _enhance_metrics_for_site24x7(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance metrics with framework-specific metadata"""
        enhanced = metrics.copy()
        enhanced.update({
            "service": "rag-observability",
            "environment": os.getenv("FLASK_ENV", "development"),
            "framework_metrics": {
                "supported_frameworks": ["langchain", "llamaindex", "autogen", "crewai", "cleanlab"],
                "active_providers": enhanced.get("active_providers", []),
                "framework_performance": enhanced.get("framework_performance", {})
            }
        })
        return enhanced

class EnhancedMetricsService:
    """Enhanced metrics service with comprehensive framework integration"""

    def __init__(self):
        self.collector = metrics_collector
        
        # Initialize Site24x7 publisher
        site24x7_api_key = os.getenv('SITE24X7_API_KEY')
        if site24x7_api_key:
            self.site24x7 = Site24x7Publisher(site24x7_api_key)
            logger.info("✅ Site24x7 integration enabled")
        else:
            self.site24x7 = None
            logger.warning("⚠️ Site24x7 API key not configured")

    def record_trace_metrics(self, trace_data: Dict[str, Any]) -> MetricData:
        """Record comprehensive metrics for a trace execution with framework analysis"""
        try:
            # Enhance trace data with framework-specific metrics
            enhanced_trace_data = self._enhance_trace_data(trace_data)
            
            # Record metrics using collector
            metric_record = self.collector.record_metrics(enhanced_trace_data)
            
            # Publish to Site24x7 if configured
            if self.site24x7:
                self._publish_framework_metrics(enhanced_trace_data, metric_record)
            
            logger.info(
                "📊 Trace metrics recorded",
                trace_id=trace_data.get('trace_id'),
                framework=trace_data.get('framework'),
                tokens=metric_record.total_tokens,
                cost=metric_record.total_cost
            )
            
            return metric_record
            
        except Exception as e:
            logger.error(f"❌ Failed to record trace metrics: {e}")
            # Return a default metric to avoid breaking the flow
            return self._create_fallback_metric(trace_data, str(e))

    def _enhance_trace_data(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance trace data with framework-specific information"""
        enhanced = trace_data.copy()
        
        framework = trace_data.get('framework', 'unknown')
        
        # Add framework-specific metadata
        framework_metadata = {
            'framework_type': self._get_framework_type(framework),
            'complexity_score': self._calculate_complexity_score(trace_data),
            'performance_tier': self._get_performance_tier(framework),
            'feature_usage': self._analyze_feature_usage(trace_data)
        }
        
        enhanced['framework_metadata'] = framework_metadata
        return enhanced

    def _get_framework_type(self, framework: str) -> str:
        """Categorize framework by type"""
        framework_types = {
            'langchain': 'chain_based',
            'llamaindex': 'index_based',
            'autogen': 'multi_agent',
            'crewai': 'crew_based',
            'cleanlab': 'data_quality',
            'neo4j': 'graph_based',
            'dspy': 'programmatic'
        }
        return framework_types.get(framework.lower(), 'unknown')

    def _calculate_complexity_score(self, trace_data: Dict[str, Any]) -> float:
        """Calculate complexity score based on query and response characteristics"""
        try:
            query = trace_data.get('query', '')
            response = trace_data.get('response', '')
            
            # Factors for complexity calculation
            query_length = len(query.split())
            response_length = len(response.split())
            duration = trace_data.get('duration', 0)
            tokens = trace_data.get('tokens_used', 0)
            
            # Normalize and combine factors
            complexity = (
                min(query_length / 50, 1.0) * 0.2 +
                min(response_length / 200, 1.0) * 0.3 +
                min(duration / 10, 1.0) * 0.3 +
                min(tokens / 1000, 1.0) * 0.2
            )
            
            return round(complexity, 3)
            
        except Exception as e:
            logger.warning(f"⚠️ Complexity calculation failed: {e}")
            return 0.5

    def _get_performance_tier(self, framework: str) -> str:
        """Get performance tier for framework"""
        performance_tiers = {
            'langchain': 'standard',
            'llamaindex': 'high',
            'autogen': 'variable',
            'crewai': 'high',
            'cleanlab': 'analytical',
            'dspy': 'optimized'
        }
        return performance_tiers.get(framework.lower(), 'standard')

    def _analyze_feature_usage(self, trace_data: Dict[str, Any]) -> Dict[str, bool]:
        """Analyze which features were used in the trace"""
        features = {
            'vector_search': bool(trace_data.get('vector_store')),
            'multi_step': trace_data.get('duration', 0) > 5,
            'high_token_usage': trace_data.get('tokens_used', 0) > 500,
            'error_handling': bool(trace_data.get('error')),
            'quality_assessment': 'cleanlab' in trace_data.get('framework', '').lower()
        }
        return features

    def _publish_framework_metrics(self, trace_data: Dict[str, Any], metric_record: MetricData):
        """Publish framework-specific metrics to Site24x7"""
        try:
            framework_metrics = {
                'trace_id': trace_data.get('trace_id'),
                'framework': trace_data.get('framework'),
                'framework_metadata': trace_data.get('framework_metadata', {}),
                'performance_metrics': {
                    'duration_ms': metric_record.latency_ms,
                    'tokens_used': metric_record.total_tokens,
                    'cost_usd': metric_record.total_cost,
                    'status': metric_record.status
                },
                'timestamp': datetime.now().isoformat()
            }
            
            self.site24x7.publish(framework_metrics)
            
        except Exception as e:
            logger.error(f"❌ Failed to publish framework metrics: {e}")

    def _create_fallback_metric(self, trace_data: Dict[str, Any], error: str) -> MetricData:
        """Create fallback metric when recording fails"""
        return MetricData(
            timestamp=datetime.now(),
            trace_id=trace_data.get('trace_id', ''),
            framework=trace_data.get('framework', 'unknown'),
            model=trace_data.get('model', 'unknown'),
            vector_store=trace_data.get('vector_store', 'unknown'),
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
            latency_ms=0.0,
            status='failed',
            error_message=error
        )

    def get_framework_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive framework analytics"""
        try:
            real_time_data = self.collector.get_real_time_metrics(days * 24)
            recent_traces = real_time_data['recent_traces']
            
            # Framework usage analysis
            framework_stats = {}
            for trace in recent_traces:
                framework = trace.get('framework', 'unknown')
                if framework not in framework_stats:
                    framework_stats[framework] = {
                        'total_requests': 0,
                        'total_tokens': 0,
                        'total_cost': 0.0,
                        'avg_latency': 0.0,
                        'success_rate': 0.0,
                        'latencies': [],
                        'successful_requests': 0
                    }
                
                stats = framework_stats[framework]
                stats['total_requests'] += 1
                stats['total_tokens'] += trace.get('total_tokens', 0)
                stats['total_cost'] += trace.get('total_cost', 0.0)
                stats['latencies'].append(trace.get('latency_ms', 0))
                
                if trace.get('status') == 'completed':
                    stats['successful_requests'] += 1
            
            # Calculate averages and success rates
            for framework, stats in framework_stats.items():
                if stats['total_requests'] > 0:
                    stats['avg_latency'] = sum(stats['latencies']) / len(stats['latencies'])
                    stats['success_rate'] = (stats['successful_requests'] / stats['total_requests']) * 100
                    stats['cost'] = round(stats['total_cost'], 4)
                del stats['latencies']  # Remove raw data
            
            return {
                'framework_performance': framework_stats,
                'top_performing_framework': self._get_top_framework(framework_stats),
                'framework_recommendations': self._get_framework_recommendations(framework_stats),
                'analysis_period_days': days,
                'total_frameworks_used': len(framework_stats)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get framework analytics: {e}")
            return {'error': str(e)}

    def _get_top_framework(self, framework_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Determine top performing framework"""
        if not framework_stats:
            return {}
        
        # Score frameworks based on success rate, latency, and usage
        scored_frameworks = []
        for framework, stats in framework_stats.items():
            score = (
                stats['success_rate'] * 0.4 +
                (1000 / max(stats['avg_latency'], 1)) * 0.3 +  # Lower latency = higher score
                min(stats['total_requests'] / 10, 10) * 0.3  # Usage factor
            )
            scored_frameworks.append((framework, score, stats))
        
        if scored_frameworks:
            top_framework, score, stats = max(scored_frameworks, key=lambda x: x[1])
            return {
                'framework': top_framework,
                'score': round(score, 2),
                'stats': stats
            }
        
        return {}

    def _get_framework_recommendations(self, framework_stats: Dict[str, Any]) -> List[str]:
        """Generate framework recommendations based on performance"""
        recommendations = []
        
        for framework, stats in framework_stats.items():
            if stats['success_rate'] < 90:
                recommendations.append(f"Consider investigating {framework} reliability (success rate: {stats['success_rate']:.1f}%)")
            
            if stats['avg_latency'] > 5000:  # 5 seconds
                recommendations.append(f"Optimize {framework} performance (avg latency: {stats['avg_latency']:.0f}ms)")
            
            if stats['total_cost'] > 1.0:  # $1 threshold
                recommendations.append(f"Monitor {framework} costs (total: ${stats['total_cost']:.2f})")
        
        if not recommendations:
            recommendations.append("All frameworks performing within acceptable parameters")
        
        return recommendations

    def get_real_time_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get real-time metrics with framework enhancements"""
        try:
            base_metrics = self.collector.get_real_time_metrics(hours)
            
            # Enhance with framework-specific data
            enhanced_metrics = base_metrics.copy()
            enhanced_metrics['framework_analytics'] = self.get_framework_analytics(hours // 24 or 1)
            
            return enhanced_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to get real-time metrics: {e}")
            return self._get_fallback_metrics()

    def get_enhanced_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get enhanced metrics with comprehensive framework analysis"""
        try:
            # Get base metrics
            real_time_data = self.collector.get_real_time_metrics(days * 24)
            framework_analytics = self.get_framework_analytics(days)
            
            # Build enhanced response
            summary = real_time_data['summary']
            time_series = real_time_data['time_series']
            
            enhanced_metrics = {
                'session_id': f'enhanced-session-{datetime.now().strftime("%Y%m%d")}',
                'total_requests': summary['total_requests'],
                'completed_requests': summary['successful_requests'],
                'failed_requests': summary['failed_requests'],
                'success_rate': summary['success_rate'],
                'average_duration': summary['avg_latency_ms'] / 1000,
                'total_tokens_used': summary['total_tokens'],
                'active_requests': 0,  # Would need real-time tracking
                
                # Enhanced framework analytics
                'framework_analytics': framework_analytics,
                
                # Token usage with framework breakdown
                'token_usage': {
                    'total_input_tokens': summary['total_input_tokens'],
                    'total_output_tokens': summary['total_output_tokens'],
                    'total_tokens': summary['total_tokens'],
                    'daily_data': {
                        'labels': time_series['labels'],
                        'input_tokens': time_series['input_tokens'],
                        'output_tokens': time_series['output_tokens'],
                        'total_tokens': time_series['total_tokens']
                    }
                },
                
                # Cost analysis
                'cost_analysis': {
                    'total_input_cost': summary['total_input_cost'],
                    'total_output_cost': summary['total_output_cost'],
                    'total_cost': summary['total_cost'],
                    'daily_data': {
                        'labels': time_series['labels'],
                        'input_costs': time_series['input_costs'],
                        'output_costs': time_series['output_costs'],
                        'total_costs': time_series['total_costs']
                    }
                },
                
                # Latency analysis
                'latency_analysis': {
                    'avg_latency_ms': summary['avg_latency_ms'],
                    'max_latency_ms': max(time_series['latencies']) if time_series['latencies'] else 0,
                    'min_latency_ms': min(time_series['latencies']) if time_series['latencies'] else 0,
                    'daily_data': {
                        'labels': time_series['labels'],
                        'latencies': time_series['latencies']
                    }
                },
                
                'time_range_days': days,
                'last_updated': datetime.now().isoformat()
            }
            
            return enhanced_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to get enhanced metrics: {e}")
            return self._get_fallback_enhanced_metrics(days)

    def get_token_usage_data(self, days: int = 7) -> Dict[str, Any]:
        """Get token usage data with framework breakdown"""
        try:
            real_time_data = self.collector.get_real_time_metrics(days * 24)
            time_series = real_time_data['time_series']
            
            # Add framework breakdown
            framework_token_usage = {}
            for trace in real_time_data['recent_traces']:
                framework = trace.get('framework', 'unknown')
                if framework not in framework_token_usage:
                    framework_token_usage[framework] = {
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'total_tokens': 0
                    }
                
                framework_token_usage[framework]['input_tokens'] += trace.get('input_tokens', 0)
                framework_token_usage[framework]['output_tokens'] += trace.get('output_tokens', 0)
                framework_token_usage[framework]['total_tokens'] += trace.get('total_tokens', 0)
            
            return {
                'labels': time_series['labels'],
                'input_tokens': time_series['input_tokens'],
                'output_tokens': time_series['output_tokens'],
                'total_tokens': time_series['total_tokens'],
                'framework_breakdown': framework_token_usage
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get token usage data: {e}")
            return {'labels': [], 'input_tokens': [], 'output_tokens': [], 'total_tokens': [], 'framework_breakdown': {}}

    def get_cost_data(self, days: int = 7, model: str = 'gpt-4o-mini') -> Dict[str, Any]:
        """Get cost data with framework and model breakdown"""
        try:
            real_time_data = self.collector.get_real_time_metrics(days * 24)
            time_series = real_time_data['time_series']
            
            # Framework cost breakdown
            framework_costs = {}
            for trace in real_time_data['recent_traces']:
                framework = trace.get('framework', 'unknown')
                if framework not in framework_costs:
                    framework_costs[framework] = {
                        'input_cost': 0.0,
                        'output_cost': 0.0,
                        'total_cost': 0.0
                    }
                
                framework_costs[framework]['input_cost'] += trace.get('input_cost', 0.0)
                framework_costs[framework]['output_cost'] += trace.get('output_cost', 0.0)
                framework_costs[framework]['total_cost'] += trace.get('total_cost', 0.0)
            
            return {
                'labels': time_series['labels'],
                'input_costs': time_series['input_costs'],
                'output_costs': time_series['output_costs'],
                'total_costs': time_series['total_costs'],
                'model': model,
                'pricing': self.collector.token_costs.get(model, self.collector.token_costs['gpt-4o-mini']),
                'framework_breakdown': framework_costs
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get cost data: {e}")
            return {
                'labels': [], 'input_costs': [], 'output_costs': [], 'total_costs': [],
                'model': model, 'pricing': {'input': 0.001, 'output': 0.002},
                'framework_breakdown': {}
            }

    def get_latency_data(self, days: int = 7) -> Dict[str, Any]:
        """Get latency data with framework performance analysis"""
        try:
            real_time_data = self.collector.get_real_time_metrics(days * 24)
            time_series = real_time_data['time_series']
            latencies = time_series['latencies']
            
            # Framework latency breakdown
            framework_latencies = {}
            for trace in real_time_data['recent_traces']:
                framework = trace.get('framework', 'unknown')
                if framework not in framework_latencies:
                    framework_latencies[framework] = []
                framework_latencies[framework].append(trace.get('latency_ms', 0))
            
            # Calculate framework averages
            framework_avg_latencies = {}
            for framework, latency_list in framework_latencies.items():
                if latency_list:
                    framework_avg_latencies[framework] = {
                        'avg_latency': sum(latency_list) / len(latency_list),
                        'max_latency': max(latency_list),
                        'min_latency': min(latency_list),
                        'sample_count': len(latency_list)
                    }
            
            return {
                'labels': time_series['labels'],
                'latencies': latencies,
                'avg_latency': sum(latencies) / len(latencies) if latencies else 0,
                'max_latency': max(latencies) if latencies else 0,
                'min_latency': min(latencies) if latencies else 0,
                'framework_performance': framework_avg_latencies
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get latency data: {e}")
            return {
                'labels': [], 'latencies': [], 'avg_latency': 0, 'max_latency': 0, 'min_latency': 0,
                'framework_performance': {}
            }

    def get_model_usage_breakdown(self) -> Dict[str, Any]:
        """Get comprehensive model and framework usage breakdown"""
        try:
            real_time_data = self.collector.get_real_time_metrics(24 * 7)
            recent_traces = real_time_data['recent_traces']
            
            model_stats = {}
            framework_model_matrix = {}
            
            for trace in recent_traces:
                model = trace['model']
                framework = trace.get('framework', 'unknown')
                
                # Model stats
                if model not in model_stats:
                    model_stats[model] = {
                        'requests': 0,
                        'tokens': 0,
                        'cost': 0.0,
                        'total_latency': 0.0,
                        'count': 0,
                        'frameworks_used': set()
                    }
                
                stats = model_stats[model]
                stats['requests'] += 1
                stats['tokens'] += trace['total_tokens']
                stats['cost'] += trace['total_cost']
                stats['total_latency'] += trace['latency_ms']
                stats['count'] += 1
                stats['frameworks_used'].add(framework)
                
                # Framework-model matrix
                if framework not in framework_model_matrix:
                    framework_model_matrix[framework] = {}
                if model not in framework_model_matrix[framework]:
                    framework_model_matrix[framework][model] = 0
                framework_model_matrix[framework][model] += 1
            
            # Calculate averages and convert sets to lists
            for model, stats in model_stats.items():
                if stats['count'] > 0:
                    stats['avg_latency'] = int(stats['total_latency'] / stats['count'])
                else:
                    stats['avg_latency'] = 0
                del stats['total_latency']
                del stats['count']
                stats['cost'] = round(stats['cost'], 4)
                stats['frameworks_used'] = list(stats['frameworks_used'])
            
            return {
                'model_performance': model_stats,
                'framework_model_matrix': framework_model_matrix,
                'analysis_summary': {
                    'total_models_used': len(model_stats),
                    'total_frameworks_used': len(framework_model_matrix),
                    'most_used_model': max(model_stats.items(), key=lambda x: x[1]['requests'])[0] if model_stats else None
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get model usage breakdown: {e}")
            return {'error': str(e)}

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        try:
            return self.collector.get_prometheus_metrics()
        except Exception as e:
            logger.error(f"❌ Failed to get Prometheus metrics: {e}")
            return ""

    def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up old metrics data"""
        try:
            return self.collector.cleanup_old_metrics(days)
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old data: {e}")
            return 0

    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """Fallback metrics when real-time collection fails"""
        return {
            'summary': {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'success_rate': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'total_input_cost': 0.0,
                'total_output_cost': 0.0,
                'total_cost': 0.0,
                'avg_latency_ms': 0.0
            },
            'time_series': {
                'labels': [],
                'input_tokens': [],
                'output_tokens': [],
                'total_tokens': [],
                'input_costs': [],
                'output_costs': [],
                'total_costs': [],
                'latencies': [],
                'request_counts': []
            },
            'recent_traces': [],
            'framework_analytics': {'error': 'No data available'}
        }

    def get_site24x7_status(self) -> Dict[str, Any]:
        """Get Site24x7 integration status"""
        try:
            api_key = os.getenv('SITE24X7_API_KEY')
            app_key = os.getenv('SITE24X7_APP_KEY')
            
            status = {
                'integration_enabled': bool(api_key and app_key),
                'api_key_configured': bool(api_key),
                'app_key_configured': bool(app_key),
                'reporter_status': 'unknown',
                'last_metrics_sent': None
            }
            
            if self.site24x7:
                status['reporter_status'] = 'active'
                # You could add more detailed status checks here
            
            return status
        except Exception as e:
            logger.error(f"❌ Failed to get Site24x7 status: {e}")
            return {
                'integration_enabled': False,
                'error': str(e)
            }

    def _get_fallback_enhanced_metrics(self, days: int) -> Dict[str, Any]:
        """Fallback enhanced metrics for backward compatibility"""
        return {
            'session_id': 'fallback-session',
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'success_rate': 0,
            'average_duration': 0,
            'total_tokens_used': 0,
            'active_requests': 0,
            'framework_analytics': {'error': 'No data available'},
            'token_usage': {
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'daily_data': {'labels': [], 'input_tokens': [], 'output_tokens': [], 'total_tokens': []}
            },
            'cost_analysis': {
                'total_input_cost': 0.0,
                'total_output_cost': 0.0,
                'total_cost': 0.0,
                'daily_data': {'labels': [], 'input_costs': [], 'output_costs': [], 'total_costs': []}
            },
            'latency_analysis': {
                'avg_latency_ms': 0,
                'max_latency_ms': 0,
                'min_latency_ms': 0,
                'daily_data': {'labels': [], 'latencies': []}
            },
            'time_range_days': days
        }

# Global enhanced metrics service instance
enhanced_metrics_service = EnhancedMetricsService()
