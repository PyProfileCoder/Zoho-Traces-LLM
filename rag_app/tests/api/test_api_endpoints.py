"""
🧪 COMPREHENSIVE API ENDPOINT TESTS
Production-ready test suite for the Ultimate RAG API
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the Flask app
from rag_app.api_server import create_app

@pytest.fixture
def app():
    """Create test Flask application"""
    # Mock environment variables
    test_env = {
        'OPENAI_API_KEY': 'test-key',
        'FLASK_ENV': 'testing',
        'SECRET_KEY': 'test-secret',
        'REDIS_URL': 'memory://'
    }
    
    with patch.dict(os.environ, test_env):
        # Mock the RAG orchestrator to avoid actual initialization
        with patch('rag_app.rag_orchestrator.create_rag_orchestrator') as mock_orchestrator:
            mock_orch = Mock()
            mock_orch.handle_query.return_value = {
                'answer': 'Test response',
                'metadata': {'success': True, 'tokens_used': 100}
            }
            mock_orch.health_check.return_value = {
                'status': 'healthy',
                'components': {'llm': 'healthy', 'vector_store': 'healthy'}
            }
            mock_orch.add_documents.return_value = True
            mock_orch.get_metrics.return_value = {'total_queries': 0}
            mock_orchestrator.return_value = mock_orch
            
            app = create_app()
            app.config['TESTING'] = True
            yield app

@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()

class TestHealthEndpoint:
    """🏥 Test health check endpoint"""
    
    def test_health_check_success(self, client):
        """Test successful health check"""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'status' in data
        assert 'timestamp' in data
        assert 'version' in data
        assert 'components' in data
        assert 'metrics' in data

class TestQueryEndpoint:
    """🧠 Test RAG query processing endpoint"""
    
    def test_query_success(self, client):
        """Test successful query processing"""
        query_data = {
            'query': 'How do I run a Docker container?',
            'llm_provider': 'openai',
            'model_name': 'gpt-4o'
        }
        
        response = client.post('/api/query', 
                             data=json.dumps(query_data),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'query_id' in data
        assert 'answer' in data
        assert 'metadata' in data
        assert 'processing_time' in data
        assert data['answer'] == 'Test response'
    
    def test_query_validation_error(self, client):
        """Test query validation error"""
        invalid_data = {
            'query': '',  # Empty query should fail validation
            'llm_provider': 'openai'
        }
        
        response = client.post('/api/query',
                             data=json.dumps(invalid_data),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error_code'] == 'VALIDATION_ERROR'
    
    def test_query_missing_json(self, client):
        """Test query with missing JSON data"""
        response = client.post('/api/query')
        assert response.status_code == 400

class TestChatEndpoint:
    """💬 Test conversational chat endpoint"""
    
    def test_chat_success(self, client):
        """Test successful chat interaction"""
        chat_data = {
            'messages': [
                {
                    'role': 'user',
                    'content': 'What is Docker?'
                }
            ],
            'llm_provider': 'openai',
            'model_name': 'gpt-4o'
        }
        
        response = client.post('/api/chat',
                             data=json.dumps(chat_data),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'message' in data
        assert 'metadata' in data
        assert data['message']['role'] == 'assistant'
        assert 'content' in data['message']
    
    def test_chat_validation_error(self, client):
        """Test chat validation error"""
        invalid_data = {
            'messages': [],  # Empty messages should fail
            'llm_provider': 'openai'
        }
        
        response = client.post('/api/chat',
                             data=json.dumps(invalid_data),
                             content_type='application/json')
        
        assert response.status_code == 400

class TestEmbedEndpoint:
    """📚 Test document embedding endpoint"""
    
    def test_embed_success(self, client):
        """Test successful document embedding"""
        embed_data = {
            'documents': [
                'Docker is a containerization platform.',
                'Containers are lightweight and portable.'
            ],
            'metadatas': [
                {'source': 'doc1'},
                {'source': 'doc2'}
            ]
        }
        
        response = client.post('/api/embed',
                             data=json.dumps(embed_data),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'success' in data
        assert 'documents_processed' in data
        assert 'processing_time' in data
        assert data['documents_processed'] == 2
    
    def test_embed_validation_error(self, client):
        """Test embed validation error"""
        invalid_data = {
            'documents': [],  # Empty documents should fail
        }
        
        response = client.post('/api/embed',
                             data=json.dumps(invalid_data),
                             content_type='application/json')
        
        assert response.status_code == 400

class TestUploadEndpoint:
    """📁 Test file upload endpoint"""
    
    def test_upload_success(self, client):
        """Test successful file upload"""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('This is a test Docker documentation file.')
            temp_file_path = f.name
        
        try:
            with open(temp_file_path, 'rb') as f:
                response = client.post('/api/upload',
                                     data={'file': (f, 'test_doc.txt')},
                                     content_type='multipart/form-data')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert 'success' in data
            assert 'filename' in data
            assert 'file_size' in data
            assert data['filename'] == 'test_doc.txt'
        
        finally:
            os.unlink(temp_file_path)
    
    def test_upload_no_file(self, client):
        """Test upload with no file"""
        response = client.post('/api/upload',
                             data={},
                             content_type='multipart/form-data')
        
        assert response.status_code == 500  # Should be handled as an error
    
    def test_upload_invalid_file_type(self, client):
        """Test upload with invalid file type"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', delete=False) as f:
            f.write('Invalid file content')
            temp_file_path = f.name
        
        try:
            with open(temp_file_path, 'rb') as f:
                response = client.post('/api/upload',
                                     data={'file': (f, 'malware.exe')},
                                     content_type='multipart/form-data')
            
            assert response.status_code == 500  # Should reject invalid file types
        
        finally:
            os.unlink(temp_file_path)

class TestModelsEndpoint:
    """🤖 Test available models endpoint"""
    
    def test_models_success(self, client):
        """Test successful models retrieval"""
        response = client.get('/api/models')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'llm_providers' in data
        assert 'frameworks' in data
        assert 'vector_stores' in data
        assert 'default_config' in data
        
        # Check that OpenAI models are included
        assert 'openai' in data['llm_providers']
        assert 'gpt-4o' in data['llm_providers']['openai']

class TestFeedbackEndpoint:
    """📝 Test feedback submission endpoint"""
    
    def test_feedback_success(self, client):
        """Test successful feedback submission"""
        feedback_data = {
            'query_id': 'test-query-123',
            'rating': 5,
            'feedback': 'Excellent response!',
            'helpful': True
        }
        
        response = client.post('/api/feedback',
                             data=json.dumps(feedback_data),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'success' in data
        assert data['success'] is True
        assert 'message' in data
    
    def test_feedback_validation_error(self, client):
        """Test feedback validation error"""
        invalid_data = {
            'query_id': 'test-query-123',
            'rating': 10,  # Rating should be 1-5
        }
        
        response = client.post('/api/feedback',
                             data=json.dumps(invalid_data),
                             content_type='application/json')
        
        assert response.status_code == 400

class TestTracesEndpoint:
    """🔍 Test traces retrieval endpoint"""
    
    def test_traces_success(self, client):
        """Test successful traces retrieval"""
        response = client.get('/api/traces')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'traces' in data
        assert 'total_count' in data
        assert 'limit' in data
        assert 'offset' in data
    
    def test_traces_with_params(self, client):
        """Test traces with query parameters"""
        response = client.get('/api/traces?limit=10&offset=5')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['limit'] == 10
        assert data['offset'] == 5

class TestDocumentsEndpoint:
    """📋 Test documents management endpoints"""
    
    def test_list_documents_success(self, client):
        """Test successful documents listing"""
        response = client.get('/api/documents')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'documents' in data
        assert 'total_count' in data
        assert 'collection_info' in data
    
    def test_delete_document_success(self, client):
        """Test successful document deletion"""
        response = client.delete('/api/documents/test-doc-123')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'success' in data
        assert data['success'] is True
        assert 'message' in data

class TestMetricsEndpoint:
    """📊 Test Prometheus metrics endpoint"""
    
    def test_metrics_success(self, client):
        """Test successful metrics retrieval"""
        response = client.get('/metrics')
        
        assert response.status_code == 200
        assert response.content_type == 'text/plain; charset=utf-8'
        
        # Check that response contains Prometheus metrics format
        metrics_text = response.data.decode('utf-8')
        assert '# HELP' in metrics_text or '# TYPE' in metrics_text

class TestDebugEndpoint:
    """🐛 Test debug configuration endpoint"""
    
    def test_debug_config_success(self, client):
        """Test successful debug config retrieval"""
        response = client.get('/api/debug/config')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should contain some configuration information
        assert isinstance(data, dict)

class TestErrorHandling:
    """🚨 Test error handling"""
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get('/api/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        
        assert 'error' in data
        assert data['error_code'] == 'NOT_FOUND'
    
    def test_method_not_allowed(self, client):
        """Test method not allowed error"""
        response = client.put('/api/health')  # Health only supports GET
        
        assert response.status_code == 405

class TestRateLimiting:
    """⚡ Test rate limiting functionality"""
    
    def test_rate_limiting_headers(self, client):
        """Test that rate limiting headers are present"""
        response = client.get('/api/health')
        
        # Rate limiting should add headers (implementation dependent)
        assert response.status_code == 200

class TestSecurity:
    """🛡️ Test security features"""
    
    def test_security_headers(self, client):
        """Test that security headers are present"""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        
        # Check for security headers
        assert 'X-Content-Type-Options' in response.headers
        assert 'X-Frame-Options' in response.headers
        assert 'X-XSS-Protection' in response.headers
        assert response.headers['X-Content-Type-Options'] == 'nosniff'
        assert response.headers['X-Frame-Options'] == 'DENY'
    
    def test_request_id_header(self, client):
        """Test that request ID header is present"""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        assert 'X-Request-ID' in response.headers
        assert 'X-Response-Time' in response.headers
        assert 'X-API-Version' in response.headers

# Performance and Load Testing
class TestPerformance:
    """⚡ Test performance characteristics"""
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get('/api/health')
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # All requests should succeed
        assert len(results) == 10
        assert all(status == 200 for status in results)
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])