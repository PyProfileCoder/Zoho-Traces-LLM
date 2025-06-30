import logging
import time
import subprocess
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict
import requests

# Core configuration imports
from config.core_config import RAGConfig, config_manager
from providers.llm_factory import LLMFactory
from providers.vector_store_factory import VectorStoreFactory
from providers.framework_factory import FrameworkFactory

# LangChain tools for agent functionality
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import create_react_agent

# Setup logging
logger = logging.getLogger(__name__)

class RAGOrchestrator:
    """
    Main orchestrator for RAG operations with comprehensive error handling and recovery.
    
    This class manages the entire RAG pipeline including:
    - LLM initialization and management
    - Vector store operations
    - Framework coordination
    - Agent-based query processing
    - Error handling and auto-recovery
    - Metrics and health monitoring
    """
    
    def __init__(self, config: RAGConfig):
        """
        Initialize RAG Orchestrator with given configuration.
        
        Args:
            config: RAGConfig object containing all necessary settings
        """
        self.config = config
        
        # Core components
        self.llm = None
        self.vector_store = None
        self.framework = None
        self.agent = None
        self.tools = []
        
        # Error handling configuration
        self.max_retries = 3
        self.error_count = 0
        self.last_error = None
        
        # Performance metrics
        self.query_count = 0
        self.success_count = 0
        self.error_escalations = 0
        self.auto_fix_attempts = 0
        
        # Initialize observability with graceful fallback
        self._initialize_observability()
        
        # Initialize core components
        self._initialize_components()
        self._setup_agent_tools()
    
    def _initialize_observability(self):
        """Initialize observability with graceful error handling."""
        try:
            from observability.trace_setup import ObservabilityManager, RAGObservability
            self.observability_manager = ObservabilityManager(config_manager)
            self.rag_observability = RAGObservability(self.observability_manager)
            self.observability_manager.initialize()
            logger.info("Observability initialized successfully")
        except ImportError as e:
            logger.warning(f"Observability modules not available: {e}")
            # Create mock objects to prevent attribute errors
            self._create_mock_observability()
        except Exception as e:
            logger.warning(f"Observability initialization failed: {e}")
            self._create_mock_observability()
    
    def _create_mock_observability(self):
        """Create mock observability objects to prevent errors."""
        class MockObservability:
            def __init__(self):
                self.is_initialized = False
            
            def initialize(self):
                pass
            
            def trace_rag_query(self, func):
                """Mock decorator that just returns the original function."""
                return func
            
            def get_metrics(self):
                return {"status": "disabled", "reason": "observability_unavailable"}
        
        self.observability_manager = MockObservability()
        self.rag_observability = MockObservability()
    
    def _initialize_components(self):
        """Initialize LLM, Vector Store, and Framework components with error handling."""
        try:
            # Initialize LLM
            logger.info(f"Initializing LLM: {self.config.llm_provider.value} - {self.config.model_name}")
            self.llm = LLMFactory.create_llm(
                provider=self.config.llm_provider,
                model_name=self.config.model_name
            )
            
            # Initialize embedding model for vector store
            embedding_model = self._get_embedding_model()
            
            # Initialize Vector Store
            logger.info(f"Initializing Vector Store: {self.config.vector_store.value}")
            vector_store_kwargs = self._get_vector_store_kwargs()
            self.vector_store = VectorStoreFactory.create_vector_store(
                provider=self.config.vector_store,
                embedding_model=embedding_model,
                collection_name=f"rag_{self.config.vector_store.value}",
                **vector_store_kwargs
            )
            
            # Initialize Framework
            logger.info(f"Initializing Framework: {self.config.framework.value}")
            self.framework = FrameworkFactory.create_framework(
                provider=self.config.framework,
                llm=self.llm,
                vector_store=self.vector_store
            )
            
            # Create RAG chain with system prompt
            system_prompt = self._get_system_prompt()
            self.framework.create_rag_chain(system_prompt)
            
            logger.info("RAG Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            raise RuntimeError(f"RAG initialization failed: {e}")
    
    def _get_embedding_model(self):
        """
        Get embedding model based on LLM provider with fallback options.
        Returns:
            Appropriate embedding model instance
        """
        try:
            if self.config.llm_provider.value == "openai":
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(
                    model=self.config.embedding_model,
                    openai_api_key=config_manager.get_api_key("openai")
                )
            elif self.config.llm_provider.value == "gemini":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                return GoogleGenerativeAIEmbeddings(
                    model=self.config.embedding_model,
                    google_api_key=config_manager.get_api_key("gemini")
                )
            else:
                # HuggingFace fallback for other providers
                from langchain_huggingface import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        except ImportError as e:
            logger.error(f"Failed to import embedding model dependencies: {e}")
            # Use default sentence transformer as ultimate fallback
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    def _get_vector_store_kwargs(self) -> Dict[str, Any]:
        """
        Get vector store specific configuration parameters.
        
        Returns:
            Dictionary of vector store specific arguments
        """
        base_path = config_manager.vector_data_dir
        
        # Default configuration
        kwargs = {
            "dimension": 1536,  # Standard dimension for most embedding models
        }
        
        # Vector store specific settings
        if self.config.vector_store.value == "faiss":
            kwargs.update({
                "index_path": f"{base_path}/faiss_index"
            })
        elif self.config.vector_store.value == "chromadb":
            kwargs.update({
                "persist_directory": f"{base_path}/chroma_db"
            })
        elif self.config.vector_store.value == "milvus":
            kwargs.update({
                "host": "localhost",
                "port": 19530
            })
        elif self.config.vector_store.value == "qdrant":
            kwargs.update({
                "host": "localhost",
                "port": 6333
            })
        elif self.config.vector_store.value == "weaviate":
            kwargs.update({
                "url": "http://localhost:8080"
            })
        
        return kwargs
    
    def _get_system_prompt(self) -> str:
        """
        Get system prompt for RAG chain focused on Docker expertise.
        
        Returns:
            Formatted system prompt string
        """
        return """
        You are an expert Docker assistant with access to comprehensive Docker documentation.
        
        Instructions:
        1. Use the provided context to answer Docker-related questions accurately
        2. If you need to execute Docker commands, clearly state the command and explain what it does
        3. If the context doesn't contain enough information, say so clearly
        4. Provide practical, actionable advice for Docker operations
        5. Always prioritize safety when suggesting Docker commands
        6. Include relevant examples and best practices
        
        Context: {context}
        Question: {input}
        
        Provide a comprehensive and accurate answer:
        """
    
    def _setup_agent_tools(self):
        """Setup agent tools for enhanced Docker assistance capabilities."""
        
        @tool
        def doc_qa(query: str) -> str:
            """Answer questions based on Docker documentation context."""
            try:
                # Use framework's query method with vector search
                result = self.framework.query(query)
                return result
            except Exception as e:
                logger.error(f"Doc QA error: {e}")
                return f"Error searching documentation: {str(e)}"
        
        @tool
        def execute_docker_command(command: str) -> str:
            """
            Execute Docker CLI commands safely with proper validation.
            
            Args:
                command: Docker command to execute
                
            Returns:
                Command output or error message
            """
            # Security validation - only allow docker commands
            if not command.strip().startswith('docker'):
                return "Error: Only Docker commands are allowed for security reasons"
            
            # Additional safety checks for destructive commands
            dangerous_patterns = ['rm -rf', 'delete', 'prune -a', 'system prune']
            if any(pattern in command.lower() for pattern in dangerous_patterns):
                return f"Warning: '{command}' is potentially destructive. Please confirm if you want to proceed with this command."
            
            try:
                logger.info(f"Executing Docker command: {command}")
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30  # 30 second timeout
                )
                
                if result.returncode == 0:
                    output = f"Command executed successfully:\n{result.stdout}"
                    logger.info(f"Command success: {command}")
                    return output
                else:
                    error_output = f"Command failed (exit code {result.returncode}):\n{result.stderr}"
                    logger.warning(f"Command failed: {command} - {result.stderr}")
                    return error_output
                    
            except subprocess.TimeoutExpired:
                error_msg = "Command timed out after 30 seconds"
                logger.error(f"Command timeout: {command}")
                return error_msg
            except Exception as e:
                error_msg = f"Execution error: {str(e)}"
                logger.error(f"Command execution error: {command} - {e}")
                return error_msg
        
        @tool
        def search_web(query: str) -> str:
            """Search web for additional Docker information when documentation is insufficient."""
            try:
                search = DuckDuckGoSearchResults(num_results=3)
                results = search.run(f"Docker {query}")
                return f"Web search results for '{query}':\n{results}"
            except Exception as e:
                logger.error(f"Web search error: {e}")
                return f"Web search error: {str(e)}"
        
        # Store tools for agent use
        self.tools = [doc_qa, execute_docker_command, search_web]
        
        # Create agent with comprehensive prompt
        try:
            agent_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a Docker expert assistant with access to:
                1. Docker documentation (doc_qa tool)
                2. Docker CLI execution capabilities (execute_docker_command tool) 
                3. Web search for additional information (search_web tool)

                For Docker questions:
                1. First consult the documentation using doc_qa
                2. If the user needs command execution or demonstration, use execute_docker_command
                3. If you need additional current information, use search_web
                4. Always explain commands before executing them
                5. Prioritize safety and best practices
                6. Provide clear, step-by-step instructions

                Be helpful, accurate, and educational in your responses."""),
                ("user", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            
            self.agent = create_react_agent(
                model=self.llm,
                tools=self.tools,
                prompt=agent_prompt
            )
            logger.info("Agent created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            self.agent = None
    
    def handle_query(self, query: str, use_agent: bool = True) -> Dict[str, Any]:
        """
        Main query handling method with comprehensive error handling and observability.
        
        Args:
            query: User query string
            use_agent: Whether to use agent capabilities (default: True)
            
        Returns:
            Dict containing answer, metadata, and performance information
        """
        start_time = time.time()
        self.query_count += 1
        
        # Initialize response structure
        response_data = {
            "answer": "",
            "metadata": {
                "query": query,
                "config": asdict(self.config),
                "response_time": 0,
                "success": False,
                "error_count": self.error_count,
                "escalated": False,
                "agent_used": use_agent
            }
        }
        
        # Apply observability tracing if available
        if (hasattr(self.rag_observability, 'trace_rag_query') and 
            hasattr(self.observability_manager, 'is_initialized') and 
            self.observability_manager.is_initialized):
            try:
                return self._handle_query_with_tracing(query, use_agent, response_data, start_time)
            except Exception as trace_error:
                logger.warning(f"Tracing failed, continuing without: {trace_error}")
        
        # Continue without tracing
        return self._handle_query_core(query, use_agent, response_data, start_time)
    
    def _handle_query_with_tracing(self, query: str, use_agent: bool, response_data: Dict, start_time: float) -> Dict[str, Any]:
        """Handle query with observability tracing."""
        @self.rag_observability.trace_rag_query
        def traced_query():
            return self._handle_query_core(query, use_agent, response_data, start_time)
        
        return traced_query()
    
    def _handle_query_core(self, query: str, use_agent: bool, response_data: Dict, start_time: float) -> Dict[str, Any]:
        """
        Core query handling logic with error recovery.
        
        Args:
            query: User query
            use_agent: Whether to use agent
            response_data: Response data structure
            start_time: Query start time
            
        Returns:
            Processed response data
        """
        try:
            # Choose query method based on agent availability and preference
            if use_agent and self.agent:
                answer = self._query_with_agent(query)
            else:
                answer = self.framework.query(query)
            
            # Record success metrics
            response_time = time.time() - start_time
            self.success_count += 1
            self.error_count = 0  # Reset error count on success
            
            response_data.update({
                "answer": answer,
                "metadata": {
                    **response_data["metadata"],
                    "response_time": response_time,
                    "success": True
                }
            })
            
            logger.info(f"Query successful in {response_time:.2f}s")
            return response_data
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            self.error_count += 1
            self.last_error = str(e)
            
            # Handle error based on retry count
            if self.error_count >= self.max_retries:
                # Escalate to LLM for analysis
                escalated_response = self._escalate_error(query, str(e))
                response_data.update({
                    "answer": escalated_response,
                    "metadata": {
                        **response_data["metadata"],
                        "response_time": time.time() - start_time,
                        "escalated": True,
                        "error": str(e)
                    }
                })
            else:
                # Attempt auto-fix and retry
                if self._attempt_auto_fix():
                    logger.info(f"Auto-fix successful, retrying query: {query}")
                    return self.handle_query(query, use_agent)  # Recursive retry
                else:
                    response_data.update({
                        "answer": f"Error processing query: {str(e)}",
                        "metadata": {
                            **response_data["metadata"],
                            "response_time": time.time() - start_time,
                            "error": str(e)
                        }
                    })
            
            return response_data
    
    def _query_with_agent(self, query: str) -> str:
        """
        Query using agent with tools and streaming support.
        
        Args:
            query: User query
            
        Returns:
            Agent response string
        """
        inputs = {"messages": [("user", query)]}
        
        final_response = ""
        try:
            # Stream through agent execution for better user experience
            for step in self.agent.stream(inputs, stream_mode="values"):
                if step.get('messages'):
                    final_response = step['messages'][-1].content
                    
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            # Fallback to direct framework query
            logger.info("Falling back to direct framework query")
            final_response = self.framework.query(query)
        
        return final_response
    
    def _attempt_auto_fix(self) -> bool:
        """
        Attempt to automatically fix common issues based on error patterns.
        
        Returns:
            True if auto-fix was attempted and might have succeeded
        """
        self.auto_fix_attempts += 1
        
        if not self.last_error:
            return False
        
        error_lower = self.last_error.lower()
        
        try:
            # Connection/Network issues - reinitialize components
            if any(keyword in error_lower for keyword in ['connection', 'timeout', 'network', 'refused']):
                logger.info("Auto-fix: Detected connection issue, reinitializing components")
                self._initialize_components()
                return True
            
            # API rate limit - implement backoff
            elif any(keyword in error_lower for keyword in ['rate limit', 'quota', 'limit exceeded', 'too many requests']):
                logger.info("Auto-fix: Rate limit detected, implementing backoff")
                time.sleep(min(2 ** self.auto_fix_attempts, 10))  # Exponential backoff, max 10s
                return True
            
            # Vector store issues - reinitialize vector store
            elif any(keyword in error_lower for keyword in ['vector', 'index', 'embedding', 'similarity']):
                logger.info("Auto-fix: Vector store issue detected, reinitializing")
                embedding_model = self._get_embedding_model()
                vector_store_kwargs = self._get_vector_store_kwargs()
                self.vector_store = VectorStoreFactory.create_vector_store(
                    provider=self.config.vector_store,
                    embedding_model=embedding_model,
                    collection_name=f"rag_{self.config.vector_store.value}",
                    **vector_store_kwargs
                )
                return True
            
            # Memory issues - force garbage collection
            elif any(keyword in error_lower for keyword in ['memory', 'out of memory', 'allocation']):
                logger.info("Auto-fix: Memory issue detected, running garbage collection")
                import gc
                gc.collect()
                return True
                
        except Exception as fix_error:
            logger.error(f"Auto-fix attempt failed: {fix_error}")
        
        return False
    
    def _escalate_error(self, query: str, error: str) -> str:
        """
        Escalate persistent errors to LLM for analysis and user guidance.
        
        Args:
            query: Original user query
            error: Error string
            
        Returns:
            LLM-generated analysis and suggestions
        """
        self.error_escalations += 1
        
        escalation_prompt = f"""
        The RAG system encountered persistent errors processing a user query.
        Please analyze the situation and provide helpful suggestions.
        
        User Query: {query}
        System Error: {error}
        Configuration: {asdict(self.config)}
        Error Count: {self.error_count}
        
        Please provide:
        1. Possible causes of this error
        2. Suggested solutions for the user
        3. Alternative approaches to get the information
        4. Any relevant Docker troubleshooting steps
        
        Be helpful and actionable in your response.
        """
        
        try:
            # Use LLM to generate helpful error analysis
            if hasattr(self.llm, 'generate'):
                suggestions = self.llm.generate(escalation_prompt)
            else:
                # Fallback if generate method not available
                suggestions = "Please check your Docker installation and try rephrasing your question."
                
            return f"""I apologize, but I encountered technical difficulties processing your query. Here's what happened and how to proceed:

**Your Query:** {query}

**Analysis and Suggestions:**
{suggestions}

**Next Steps:**
- Try rephrasing your question
- Check if you need specific Docker setup steps
- Use the web interface to try different LLM/framework combinations
- Verify your Docker installation is working: `docker --version`

The system administrators have been notified of this issue."""

        except Exception as e:
            logger.error(f"Error escalation failed: {e}")
            return f"""I apologize, but I'm experiencing technical difficulties and cannot process your query at this time.

**Your Query:** {query}
**Error:** {error}

**Suggested Actions:**
1. Try again in a few moments
2. Rephrase your question
3. Check Docker documentation directly at https://docs.docker.com
4. Verify Docker is installed: `docker --version`
5. Contact system administrator if the issue persists

Error ID: {int(time.time())}"""
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None) -> bool:
        """
        Add documents to the vector store for enhanced context.
        
        Args:
            documents: List of document strings
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            self.vector_store.add_documents(documents, metadatas)
            logger.info("Documents added successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive system metrics and performance data.
        
        Returns:
            Dictionary containing all system metrics
        """
        success_rate = (self.success_count / max(self.query_count, 1)) * 100
        
        metrics = {
            "orchestrator_metrics": {
                "total_queries": self.query_count,
                "successful_queries": self.success_count,
                "failed_queries": self.query_count - self.success_count,
                "success_rate": round(success_rate, 2),
                "current_error_count": self.error_count,
                "error_escalations": self.error_escalations,
                "auto_fix_attempts": self.auto_fix_attempts,
                "last_error": self.last_error
            },
            "configuration": asdict(self.config),
            "component_status": self._get_component_status()
        }
        
        # Add observability metrics if available
        try:
            if hasattr(self.rag_observability, 'get_metrics'):
                metrics["observability_metrics"] = self.rag_observability.get_metrics()
        except Exception as e:
            logger.warning(f"Could not get observability metrics: {e}")
            metrics["observability_metrics"] = {
                "status": "unavailable", 
                "error": str(e)
            }
        
        return metrics
    
    def _get_component_status(self) -> Dict[str, str]:
        """
        Check the health status of all system components.
        
        Returns:
            Dictionary with component names and their status
        """
        status = {}
        
        # Check LLM health
        try:
            if hasattr(self.llm, 'generate'):
                # Test with simple query
                test_response = self.llm.generate("Hello")
                status["llm"] = "healthy" if test_response else "unhealthy"
            else:
                status["llm"] = "healthy"  # Assume healthy if no generate method
        except Exception as e:
            status["llm"] = f"error: {str(e)[:100]}"
        
        # Check Vector Store health
        try:
            # Test similarity search
            test_search = self.vector_store.similarity_search("test", k=1)
            status["vector_store"] = "healthy"
        except Exception as e:
            status["vector_store"] = f"error: {str(e)[:100]}"
        
        # Check Framework health
        try:
            # Test framework query
            test_query = self.framework.query("test")
            status["framework"] = "healthy" if test_query else "unhealthy"
        except Exception as e:
            status["framework"] = f"error: {str(e)[:100]}"
        
        # Check Agent status
        status["agent"] = "healthy" if self.agent else "not_initialized"
        
        return status
    
    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive system health check.
        
        Returns:
            Dictionary containing complete health status
        """
        overall_status = "healthy" if self.error_count < self.max_retries else "degraded"
        
        health_data = {
            "status": overall_status,
            "timestamp": time.time(),
            "components": self._get_component_status(),
            "metrics": self.get_metrics()
        }
        
        # Add observability status if available
        try:
            health_data["observability"] = {
                "initialized": getattr(self.observability_manager, 'is_initialized', False),
                "tracer_active": (hasattr(self.rag_observability, 'obs_manager') and 
                                hasattr(self.rag_observability.obs_manager, 'tracer') and 
                                self.rag_observability.obs_manager.tracer is not None)
            }
        except Exception as e:
            health_data["observability"] = {
                "status": "error", 
                "message": str(e)
            }
        
        return health_data
    
    def reset_error_state(self):
        """Reset error state for recovery after manual intervention."""
        self.error_count = 0
        self.last_error = None
        logger.info("Error state reset successfully")
    
    def switch_configuration(self, new_config: RAGConfig):
        """
        Switch to a new configuration and reinitialize components.
        
        Args:
            new_config: New RAGConfig to switch to
        """
        logger.info(f"Switching configuration from {self.config} to {new_config}")
        self.config = new_config
        self.reset_error_state()
        
        try:
            self._initialize_components()
            self._setup_agent_tools()
            logger.info("Configuration switched successfully")
        except Exception as e:
            logger.error(f"Configuration switch failed: {e}")
            raise


def create_rag_orchestrator(
    llm_provider: str = "openai",
    framework: str = "langchain",
    vector_store: str = "faiss",
    model_name: str = "gpt-4o",
    embedding_model: str = None,
    **kwargs
) -> RAGOrchestrator:
    """
    Factory function to create RAG orchestrator with specified configuration.
    
    Args:
        llm_provider: LLM provider name
        framework: Framework to use
        vector_store: Vector store type
        model_name: Specific model name
        embedding_model: Embedding model name
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured RAGOrchestrator instance
    """
    config = config_manager.create_rag_config(
        llm_provider=llm_provider,
        framework=framework,
        vector_store=vector_store,
        model_name=model_name,
        embedding_model=embedding_model
    )
    
    return RAGOrchestrator(config)


# Safe execution for testing
if __name__ == "__main__":
    try:
        logger.info("Starting RAG Orchestrator test...")
        
        # Test with default configuration
        orchestrator = create_rag_orchestrator(
            llm_provider="openai",
            framework="langchain",
            vector_store="faiss",
            model_name="gpt-4o"
        )
        
        # Test basic functionality
        test_query = "How do I run a Docker container?"
        logger.info(f"Testing query: {test_query}")
        
        result = orchestrator.handle_query(test_query)
        
        # Display results
        answer = result['answer']
        if len(answer) > 200:
            print(f"Result: {answer[:200]}...")
        else:
            print(f"Result: {answer}")
        
        # Display health status
        health = orchestrator.health_check()
        print(f"Health Status: {health['status']}")
        print(f"Components Status: {health['components']}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"Error during execution: {e}")
        print("Please check your configuration and dependencies.")
        print("Common issues:")
        print("1. Missing API keys in environment variables")
        print("2. Required packages not installed")
        print("3. Observability modules not properly configured")