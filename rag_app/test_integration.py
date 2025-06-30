#!/usr/bin/env python3
"""
Integration Test Script for Docker Agent with Site24x7
"""

import requests
import time
import json
import sys

def test_flask_api():
    """Test Flask API endpoints"""
    try:
        response = requests.get('http://localhost:5001/api/health', timeout=10)
        if response.status_code == 200:
            print("✅ Flask API is running")
            return True
        else:
            print(f"❌ Flask API returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Flask API test failed: {e}")
        return False

def test_metrics_endpoint():
    """Test metrics endpoint"""
    try:
        response = requests.get('http://localhost:5001/metrics', timeout=10)
        if response.status_code == 200:
            print("✅ Metrics endpoint is working")
            return True
        else:
            print(f"❌ Metrics endpoint returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Metrics endpoint test failed: {e}")
        return False

def test_query_execution():
    """Test query execution"""
    try:
        test_query = {
            "query": "How do I run a Docker container?",
            "framework": "langchain",
            "model": "gpt-4o-mini",
            "llm_provider": "openai",
            "vector_store": "faiss"
        }
        
        response = requests.post(
            'http://localhost:5001/api/query',
            json=test_query,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Query execution successful")
            print(f"📊 Trace ID: {result.get('trace_id', 'N/A')}")
            return True
        else:
            print(f"❌ Query execution failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Query execution test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🧪 Running Integration Tests...")
    print("=" * 50)
    
    tests = [
        ("Flask API", test_flask_api),
        ("Metrics Endpoint", test_metrics_endpoint),
        ("Query Execution", test_query_execution)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        result = test_func()
        results.append(result)
        time.sleep(2)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} tests passed!")
        print("\n✅ Integration successful! Your Docker Agent is ready.")
        return 0
    else:
        print(f"❌ {total - passed} out of {total} tests failed.")
        print("\n🔧 Please check the logs and fix the issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
