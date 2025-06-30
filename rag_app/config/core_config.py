import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from enum import Enum
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"


class Framework(Enum):
    LANGCHAIN = "langchain"
    LLAMAINDEX = "llamaindex"
    AUTOGEN = "autogen"
    CREWAI = "crewai"
    NEO4J = "neo4j"
    GRAPHLIT = "graphlit"
    AWS_BEDROCK = "aws_bedrock"
    LITELLM = "litellm"
    VERCEL = "vercel"
    CLEANLAB = "cleanlab"


class VectorStore(Enum):
    FAISS = "faiss"
    CHROMADB = "chromadb"
    MILVUS = "milvus"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    PINECONE = "pinecone"


@dataclass
class ObservabilityConfig:
    """Configuration for observability tools"""
    trace_enabled: bool = True
    service_name: str = "rag-observability-tool"
    environment: str = "development"
    grafana_tempo_endpoint: Optional[str] = None
    
    def __post_init__(self):
        """Set default values after initialization"""
        if self.grafana_tempo_endpoint is None:
            self.grafana_tempo_endpoint = os.getenv("GRAFANA_TEMPO_ENDPOINT", "http://localhost:3200")


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    llm_provider: LLMProvider
    framework: Framework
    vector_store: VectorStore
    model_name: str
    embedding_model: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.observability = ObservabilityConfig()
        self.data_dir = os.getenv("DATA_DIR", "data")
        self.vector_data_dir = os.getenv("VECTOR_DATA_DIR", "vector_data")
        self._validate_required_keys()
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment - Enhanced version"""
        keys = {}
        
        # LLM Provider Keys
        env_mappings = {
            # OpenAI
            "openai": "OPENAI_API_KEY",
            "openai_org": "OPENAI_ORG_ID",
            
            # Google
            "gemini": "GEMINI_API_KEY",
            "google_credentials": "GOOGLE_APPLICATION_CREDENTIALS",
            
            # Groq
            "groq": "GROQ_API_KEY",
            
            # Anthropic
            "anthropic": "ANTHROPIC_API_KEY",
            
            # Cohere
            "cohere": "COHERE_API_KEY",
            
            # Vector Stores
            "qdrant": "QDRANT_API_KEY",
            "qdrant_url": "QDRANT_URL",
            "weaviate": "WEAVIATE_API_KEY", 
            "weaviate_url": "WEAVIATE_URL",
            "pinecone": "PINECONE_API_KEY",
            "pinecone_env": "PINECONE_ENVIRONMENT",
            
            # Graph Databases
            "neo4j_uri": "NEO4J_URI",
            "neo4j_user": "NEO4J_USER", 
            "neo4j_password": "NEO4J_PASSWORD",
            
            # Cloud Providers
            "aws_access_key": "AWS_ACCESS_KEY_ID",
            "aws_secret_key": "AWS_SECRET_ACCESS_KEY",
            "aws_region": "AWS_REGION",
            
            # Framework Specific
            "vercel_endpoint": "VERCEL_ENDPOINT",
            "vercel_token": "VERCEL_TOKEN",
            
            # Observability
            "traceloop": "TRACELOOP_API_KEY",
            "grafana_tempo": "GRAFANA_TEMPO_ENDPOINT",
            
            # Additional Services
            "huggingface": "HUGGINGFACE_API_KEY",
        }
        
        # Load all environment variables
        for key, env_var in env_mappings.items():
            if value := os.getenv(env_var):
                keys[key] = value
        
        return keys
    
    def _validate_required_keys(self):
        """Validate that required API keys are present"""
        required_keys = {
            'openai': 'OPENAI_API_KEY',
            'gemini': 'GEMINI_API_KEY',
            'groq': 'GROQ_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY'
        }
        
        # Check if at least one LLM provider key is available
        available_providers = [provider for provider in required_keys.keys() 
                             if self.get_api_key(provider)]
        
        if not available_providers:
            logger.warning("No LLM provider API keys found. Please set at least one of: %s", 
                          list(required_keys.values()))
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specific provider"""
        key = self.api_keys.get(provider.lower())
        if not key:
            logger.warning(f"API key not found for provider: {provider}")
        return key
    
    def has_api_key(self, provider: str) -> bool:
        """Check if API key exists for provider"""
        return provider.lower() in self.api_keys
    
    def create_rag_config(
        self,
        llm_provider: str,
        framework: str,
        vector_store: str,
        model_name: str,
        embedding_model: Optional[str] = None,
        **kwargs
    ) -> RAGConfig:
        """Create RAG configuration from string inputs"""
        
        # Validate provider has API key
        if not self.has_api_key(llm_provider):
            raise ValueError(f"No API key found for provider: {llm_provider}")
        
        # Default embedding models
        if not embedding_model:
            embedding_defaults = {
                "openai": "text-embedding-3-large",
                "gemini": "models/embedding-001",
                "anthropic": "sentence-transformers/all-MiniLM-L6-v2",
                "cohere": "embed-english-v3.0",
                "groq": "sentence-transformers/all-MiniLM-L6-v2"
            }
            embedding_model = embedding_defaults.get(llm_provider, "sentence-transformers/all-MiniLM-L6-v2")
        
        try:
            return RAGConfig(
                llm_provider=LLMProvider(llm_provider),
                framework=Framework(framework),
                vector_store=VectorStore(vector_store),
                model_name=model_name,
                embedding_model=embedding_model,
                **kwargs
            )
        except ValueError as e:
            logger.error(f"Invalid configuration value: {e}")
            raise
    
    def get_available_providers(self) -> list[str]:
        """Get list of available LLM providers with valid API keys"""
        return [provider for provider in LLMProvider if self.has_api_key(provider.value)]
    
    def ensure_directories(self):
        """Ensure data directories exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.vector_data_dir, exist_ok=True)
        logger.info(f"Data directories ensured: {self.data_dir}, {self.vector_data_dir}")


@dataclass
class SystemConfig:
    """Overall system configuration"""
    rag_config: RAGConfig
    flask_config: Dict[str, Any]
    observability_config: ObservabilityConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "rag": asdict(self.rag_config),
            "flask": self.flask_config,
            "observability": asdict(self.observability_config)
        }
    
    def validate(self) -> bool:
        """Validate the complete system configuration"""
        try:
            # Validate RAG config
            if not isinstance(self.rag_config, RAGConfig):
                raise ValueError("Invalid RAG configuration")
            
            # Validate Flask config
            required_flask_keys = ['DEBUG', 'HOST', 'PORT']
            if not all(key in self.flask_config for key in required_flask_keys):
                raise ValueError("Missing required Flask configuration keys")
            
            # Validate observability config
            if not isinstance(self.observability_config, ObservabilityConfig):
                raise ValueError("Invalid observability configuration")
            
            logger.info("System configuration validated successfully")
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


def create_system_config() -> SystemConfig:
    """Create complete system configuration"""
    try:
        # Ensure config manager is initialized
        config_manager.ensure_directories()
        
        # Get environment variables with defaults
        llm_provider = os.getenv('LLM_PROVIDER', 'openai')
        framework = os.getenv('FRAMEWORK', 'langchain')
        vector_store = os.getenv('VECTOR_STORE', 'faiss')
        model_name = os.getenv('MODEL_NAME', 'gpt-4o')
        
        # Create system configuration
        system_config = SystemConfig(
            rag_config=config_manager.create_rag_config(
                llm_provider=llm_provider,
                framework=framework,
                vector_store=vector_store,
                model_name=model_name
            ),
            flask_config={
                'DEBUG': os.getenv('DEBUG', 'False').lower() == 'true',
                'HOST': os.getenv('HOST', '0.0.0.0'),
                'PORT': int(os.getenv('PORT', 5000)),
                'SECRET_KEY': os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
            },
            observability_config=ObservabilityConfig()
        )
        
        # Validate configuration
        if not system_config.validate():
            raise ValueError("System configuration validation failed")
        
        logger.info("System configuration created successfully")
        return system_config
        
    except Exception as e:
        logger.error(f"Failed to create system configuration: {e}")
        raise


# Global configuration instance
config_manager = ConfigManager()

# Export commonly used functions
__all__ = [
    'LLMProvider',
    'Framework', 
    'VectorStore',
    'RAGConfig',
    'ObservabilityConfig',
    'SystemConfig',
    'ConfigManager',
    'config_manager',
    'create_system_config'
]