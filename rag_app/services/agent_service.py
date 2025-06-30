"""
Agent Service: RAG Agent Orchestration with Factory-based Providers and Enhanced Observability
Production-grade service with comprehensive framework integration and Site24x7 metrics
"""

from typing import Dict, Any, Optional
import structlog
import time
import hashlib
from datetime import datetime

from config.settings import settings
from core.tracing import tracing_manager

# Provider factories - dynamic framework discovery
from providers.framework_factory import FrameworkFactory
from providers.llm_factory import LLMFactory
from providers.vector_store_factory import VectorStoreFactory

# Enhanced metrics and observability
try:
    from services.enhanced_metrics_service import enhanced_metrics_service
except ImportError:
    enhanced_metrics_service = None

logger = structlog.get_logger()

class AgentService:
    """
    Production AgentService with comprehensive framework integration:
    - Dynamic framework, LLM, and vector store selection via factories
    - Full tracing and enhanced metrics with Site24x7 integration
    - Advanced error handling and escalation
    - Framework-specific performance analytics
    """

    def __init__(self):
        """Initialize AgentService with provider factories"""
        try:
            # Initialize factories for dynamic provider discovery
            self.framework_factory = FrameworkFactory()
            self.llm_factory = LLMFactory()
            self.vector_store_factory = VectorStoreFactory()
            
            # Performance tracking
            self.query_count = 0
            self.success_count = 0
            self.framework_performance = {}
            
            logger.info("✅ AgentService initialized with provider factories")
            
        except Exception as e:
            logger.error(f"❌ AgentService initialization failed: {e}")
            # Create minimal fallback factories
            self._create_fallback_factories()

    def _create_fallback_factories(self):
        """Create minimal fallback factories when initialization fails"""
        class MockFactory:
            def list_supported_frameworks(self):
                return ['mock']
            def list_supported_models(self):
                return ['mock-model']
            def list_supported_vector_stores(self):
                return ['mock-store']
            def create_framework(self, *args, **kwargs):
                return MockFramework()
            def create_llm(self, *args, **kwargs):
                return MockLLM()
            def create_vector_store(self, *args, **kwargs):
                return MockVectorStore()
        
        class MockFramework:
            def query(self, question):
                return f"Mock response for: {question}"
        
        class MockLLM:
            def generate(self, prompt):
                return f"Mock LLM response for: {prompt[:50]}..."
        
        class MockVectorStore:
            def similarity_search(self, query, k=5):
                return [("Mock document content", 0.9)]
        
        self.framework_factory = MockFactory()
        self.llm_factory = MockFactory()
        self.vector_store_factory = MockFactory()
        logger.warning("⚠️ Using fallback mock factories")

    def get_available_configurations(self) -> Dict[str, Any]:
        """
        Get all available frameworks, LLM models, and vector stores for UI/API.
        Enhanced with framework metadata and performance recommendations.
        """
        try:
            base_config = {
                "frameworks": self.framework_factory.list_supported_frameworks(),
                "models": self.llm_factory.list_supported_models(),
                "vector_stores": self.vector_store_factory.list_supported_vector_stores()
            }
            
            # Enhance with framework metadata
            enhanced_config = self._enhance_configurations(base_config)
            
            return enhanced_config
            
        except Exception as e:
            logger.error(f"❌ Failed to get configurations: {e}")
            return {
                "frameworks": ['mock'],
                "models": ['mock-model'],
                "vector_stores": ['mock-store'],
                "error": str(e)
            }

    def _enhance_configurations(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance configurations with framework metadata and recommendations"""
        enhanced = base_config.copy()
        
        # Framework metadata
        framework_metadata = {}
        for framework in base_config.get("frameworks", []):
            framework_metadata[framework] = {
                "type": self._get_framework_type(framework),
                "complexity": self._get_framework_complexity(framework),
                "best_use_cases": self._get_framework_use_cases(framework),
                "performance_tier": self._get_framework_performance_tier(framework)
            }
        
        enhanced["framework_metadata"] = framework_metadata
        
        # Performance recommendations
        if self.framework_performance:
            enhanced["performance_recommendations"] = self._get_performance_recommendations()
        
        # Model categorization
        model_categories = {}
        for model in base_config.get("models", []):
            model_categories[model] = {
                "provider": self._get_model_provider(model),
                "cost_tier": self._get_model_cost_tier(model),
                "performance_tier": self._get_model_performance_tier(model)
            }
        
        enhanced["model_metadata"] = model_categories
        
        return enhanced

    def _get_framework_type(self, framework: str) -> str:
        """Get framework type classification"""
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

    def _get_framework_complexity(self, framework: str) -> str:
        """Get framework complexity level"""
        complexity_levels = {
            'langchain': 'medium',
            'llamaindex': 'high',
            'autogen': 'high',
            'crewai': 'high',
            'cleanlab': 'medium',
            'dspy': 'low'
        }
        return complexity_levels.get(framework.lower(), 'medium')

    def _get_framework_use_cases(self, framework: str) -> list:
        """Get best use cases for framework"""
        use_cases = {
            'langchain': ['general_qa', 'document_processing', 'workflow_automation'],
            'llamaindex': ['complex_indexing', 'multi_document_qa', 'knowledge_management'],
            'autogen': ['multi_agent_collaboration', 'complex_reasoning', 'code_generation'],
            'crewai': ['team_based_tasks', 'role_specific_agents', 'collaborative_problem_solving'],
            'cleanlab': ['data_quality_assessment', 'noise_detection', 'confidence_scoring'],
            'dspy': ['prompt_optimization', 'programmatic_prompting', 'performance_tuning']
        }
        return use_cases.get(framework.lower(), ['general_purpose'])

    def _get_framework_performance_tier(self, framework: str) -> str:
        """Get performance tier based on historical data"""
        if framework in self.framework_performance:
            avg_latency = self.framework_performance[framework].get('avg_latency', 1000)
            success_rate = self.framework_performance[framework].get('success_rate', 100)
            
            if success_rate > 95 and avg_latency < 2000:
                return 'high'
            elif success_rate > 90 and avg_latency < 5000:
                return 'medium'
            else:
                return 'low'
        
        # Default tiers based on framework characteristics
        default_tiers = {
            'langchain': 'high',
            'llamaindex': 'medium',
            'autogen': 'medium',
            'crewai': 'medium',
            'cleanlab': 'high',
            'dspy': 'high'
        }
        return default_tiers.get(framework.lower(), 'medium')

    def _get_model_provider(self, model: str) -> str:
        """Get model provider"""
        if 'gpt' in model.lower():
            return 'openai'
        elif 'gemini' in model.lower():
            return 'google'
        elif 'llama' in model.lower() or 'mixtral' in model.lower():
            return 'groq'
        else:
            return 'unknown'

    def _get_model_cost_tier(self, model: str) -> str:
        """Get model cost tier"""
        high_cost_models = ['gpt-4o', 'gpt-4-turbo', 'gemini-1.5-pro']
        medium_cost_models = ['gpt-3.5-turbo', 'gemini-1.5-flash']
        
        if model in high_cost_models:
            return 'high'
        elif model in medium_cost_models:
            return 'medium'
        else:
            return 'low'

    def _get_model_performance_tier(self, model: str) -> str:
        """Get model performance tier"""
        high_performance = ['gpt-4o', 'gpt-4-turbo', 'gemini-1.5-pro']
        medium_performance = ['gpt-3.5-turbo', 'llama3-70b-8192']
        
        if model in high_performance:
            return 'high'
        elif model in medium_performance:
            return 'medium'
        else:
            return 'standard'

    def _get_performance_recommendations(self) -> list:
        """Generate performance recommendations based on historical data"""
        recommendations = []
        
        for framework, perf in self.framework_performance.items():
            if perf.get('success_rate', 100) < 90:
                recommendations.append(f"Consider investigating {framework} reliability issues")
            
            if perf.get('avg_latency', 0) > 10000:  # 10 seconds
                recommendations.append(f"Optimize {framework} performance for better response times")
        
        if not recommendations:
            recommendations.append("All frameworks performing within acceptable parameters")
        
        return recommendations

    def execute_query(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a RAG query using selected framework, LLM, and vector store.
        Enhanced with comprehensive tracing, metrics, and framework-specific analytics.
        """
        trace_id = tracing_manager.start_trace(request_data)
        start_time = time.time()
        self.query_count += 1
        
        # Extract configuration
        framework_name = request_data.get('framework', 'langchain')
        llm_provider = request_data.get('llm_provider', 'openai')
        model_name = request_data.get('model', 'gpt-4o-mini')
        vector_store_name = request_data.get('vector_store', 'faiss')
        query = request_data.get('query', '')
        
        try:
            # Dynamic instantiation via factories
            tracing_manager.add_step(trace_id, 'provider_selection', {
                'framework': framework_name,
                'llm_provider': llm_provider,
                'model': model_name,
                'vector_store': vector_store_name
            })

            # Create LLM instance
            llm = self.llm_factory.create_llm(
                provider=llm_provider,
                model_name=model_name
            )
            
            # Create Vector Store instance
            vector_store = self.vector_store_factory.create_vector_store(
                provider=vector_store_name,
                embedding_model=getattr(llm, 'embedding_model', None)
            )

            tracing_manager.add_step(trace_id, 'component_initialization', {
                'llm_type': type(llm).__name__,
                'vector_store_type': type(vector_store).__name__
            })

            # Create Framework instance with LLM and Vector Store
            framework = self.framework_factory.create_framework(
                provider=framework_name,
                llm=llm,
                vector_store=vector_store
            )

            tracing_manager.add_step(trace_id, 'framework_ready', {
                'framework_class': type(framework).__name__
            })

            # Execute query with framework-specific handling
            result = self._execute_framework_query(framework, query, framework_name)
            
            end_time = time.time()
            duration = end_time - start_time

            tracing_manager.add_step(trace_id, 'query_execution', {
                'query_length': len(query),
                'response_length': len(result.get('answer', '')) if isinstance(result, dict) else len(str(result)),
                'duration': duration,
                'status': 'completed'
            })

            # Process and clean response
            answer = result.get('answer') if isinstance(result, dict) else str(result)
            cleaned_response = self._clean_response(answer)

            # Update framework performance tracking
            self._update_framework_performance(framework_name, duration, True)

            # Prepare comprehensive metrics data
            metrics_data = {
                'trace_id': trace_id,
                'framework': framework_name,
                'llm_provider': llm_provider,
                'model': model_name,
                'vector_store': vector_store_name,
                'query': query,
                'response': cleaned_response,
                'duration': duration,
                'tokens_used': result.get('tokens_used', 0) if isinstance(result, dict) else 0,
                'input_tokens': result.get('input_tokens') if isinstance(result, dict) else None,
                'output_tokens': result.get('output_tokens') if isinstance(result, dict) else None,
                'status': 'completed',
                'framework_metadata': {
                    'framework_type': self._get_framework_type(framework_name),
                    'complexity_score': self._calculate_query_complexity(query, cleaned_response),
                    'performance_tier': self._get_framework_performance_tier(framework_name)
                }
            }

            # Record enhanced metrics
            if enhanced_metrics_service:
                try:
                    metric_record = enhanced_metrics_service.record_trace_metrics(metrics_data)
                    logger.info(
                        "📊 Enhanced metrics recorded",
                        trace_id=trace_id,
                        framework=framework_name,
                        tokens=getattr(metric_record, 'total_tokens', 0),
                        cost=getattr(metric_record, 'total_cost', 0)
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to record enhanced metrics: {e}")

            # Prepare final response
            final_result = {
                'answer': cleaned_response,
                'trace_id': trace_id,
                'framework': framework_name,
                'model': model_name,
                'vector_store': vector_store_name,
                'duration': duration,
                'tokens_used': metrics_data['tokens_used'],
                'status': 'success',
                'framework_metadata': metrics_data['framework_metadata']
            }

            tracing_manager.end_trace(trace_id, 'completed')
            self.success_count += 1
            
            logger.info(
                "✅ Query executed successfully",
                trace_id=trace_id,
                framework=framework_name,
                duration=duration
            )

            return final_result

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            # Update framework performance with failure
            self._update_framework_performance(framework_name, duration, False)

            # Record failed metrics
            if enhanced_metrics_service:
                metrics_data = {
                    'trace_id': trace_id,
                    'framework': framework_name,
                    'llm_provider': llm_provider,
                    'model': model_name,
                    'vector_store': vector_store_name,
                    'query': query,
                    'response': '',
                    'duration': duration,
                    'tokens_used': 0,
                    'status': 'failed',
                    'error': str(e)
                }

                try:
                    enhanced_metrics_service.record_trace_metrics(metrics_data)
                except Exception as metrics_error:
                    logger.error(f"❌ Failed to record error metrics: {metrics_error}")

            tracing_manager.end_trace(trace_id, 'failed', str(e))
            logger.error(
                "❌ Query execution failed",
                trace_id=trace_id,
                error=str(e),
                framework=framework_name
            )

            return {
                'answer': f"❌ Error: {str(e)}",
                'trace_id': trace_id,
                'framework': framework_name,
                'model': model_name,
                'vector_store': vector_store_name,
                'duration': duration,
                'tokens_used': 0,
                'status': 'error',
                'error': str(e)
            }

    def _execute_framework_query(self, framework, query: str, framework_name: str) -> Any:
        """Execute query with framework-specific optimizations"""
        try:
            # Framework-specific query execution
            if framework_name.lower() == 'cleanlab':
                # For Cleanlab, we might want quality assessment
                return framework.query(query, assess_quality=True)
            elif framework_name.lower() in ['autogen', 'crewai']:
                # For multi-agent frameworks, we might want collaborative mode
                return framework.query(query, collaborative=True)
            else:
                # Standard query execution
                return framework.query(query)
                
        except Exception as e:
            logger.error(f"❌ Framework-specific query failed: {e}")
            # Fallback to standard query
            return framework.query(query)

    def _calculate_query_complexity(self, query: str, response: str) -> float:
        """Calculate query complexity score"""
        try:
            # Factors for complexity calculation
            query_length = len(query.split())
            response_length = len(response.split())
            
            # Complexity indicators
            question_words = len([w for w in query.lower().split() if w in ['how', 'why', 'what', 'when', 'where', 'which']])
            technical_terms = len([w for w in query.lower().split() if w in ['docker', 'container', 'kubernetes', 'api', 'database']])
            
            # Normalize and combine factors
            complexity = (
                min(query_length / 20, 1.0) * 0.3 +
                min(response_length / 100, 1.0) * 0.3 +
                min(question_words / 3, 1.0) * 0.2 +
                min(technical_terms / 5, 1.0) * 0.2
            )
            
            return round(complexity, 3)
            
        except Exception as e:
            logger.warning(f"⚠️ Complexity calculation failed: {e}")
            return 0.5

    def _update_framework_performance(self, framework_name: str, duration: float, success: bool):
        """Update framework performance tracking"""
        if framework_name not in self.framework_performance:
            self.framework_performance[framework_name] = {
                'total_queries': 0,
                'successful_queries': 0,
                'total_duration': 0.0,
                'avg_latency': 0.0,
                'success_rate': 100.0
            }
        
        perf = self.framework_performance[framework_name]
        perf['total_queries'] += 1
        perf['total_duration'] += duration
        
        if success:
            perf['successful_queries'] += 1
        
        # Update averages
        perf['avg_latency'] = (perf['total_duration'] / perf['total_queries']) * 1000  # Convert to ms
        perf['success_rate'] = (perf['successful_queries'] / perf['total_queries']) * 100

    def _clean_response(self, text: str) -> str:
        """Clean and format the response"""
        if not text:
            return "No response generated."

        # Remove query repetition and clean formatting
        cleaned = text.strip()
        
        # Remove common prefixes
        prefixes_to_remove = ["Here is", "Here's", "Based on the context"]
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                if cleaned.startswith(",") or cleaned.startswith(":"):
                    cleaned = cleaned[1:].strip()
        
        return cleaned if cleaned else "Response generated but content was empty."

    def get_framework_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive framework performance metrics"""
        return {
            'framework_performance': self.framework_performance,
            'total_queries': self.query_count,
            'total_successful': self.success_count,
            'overall_success_rate': (self.success_count / max(self.query_count, 1)) * 100,
            'performance_summary': self._get_performance_summary()
        }

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        if not self.framework_performance:
            return {'status': 'No performance data available'}
        
        # Find best and worst performing frameworks
        best_framework = max(
            self.framework_performance.items(),
            key=lambda x: x[1]['success_rate']
        )
        
        fastest_framework = min(
            self.framework_performance.items(),
            key=lambda x: x[1]['avg_latency']
        )
        
        return {
            'best_reliability': {
                'framework': best_framework[0],
                'success_rate': best_framework[1]['success_rate']
            },
            'fastest_response': {
                'framework': fastest_framework[0],
                'avg_latency_ms': fastest_framework[1]['avg_latency']
            },
            'total_frameworks_used': len(self.framework_performance)
        }

# Global service instance
agent_service = AgentService()
