#!/usr/bin/env python3
"""
Comprehensive test execution with detailed reporting
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_tests():
    """🚀 Run comprehensive test suite"""
    
    print("🧪 ULTIMATE RAG API TEST SUITE")
    print("=" * 50)
    
    # Set test environment
    test_env = os.environ.copy()
    test_env.update({
        'FLASK_ENV': 'testing',
        'OPENAI_API_KEY': 'test-key',
        'SECRET_KEY': 'test-secret',
        'REDIS_URL': 'memory://'
    })
    
    # Test commands to run
    test_commands = [
        {
            'name': '🔍 Unit Tests',
            'cmd': ['python', '-m', 'pytest', 'tests/', '-v', '--tb=short'],
            'description': 'Run all unit tests with verbose output'
        },
        {
            'name': '📊 Coverage Report',
            'cmd': ['python', '-m', 'pytest', 'tests/', '--cov=api_server', '--cov-report=html', '--cov-report=term'],
            'description': 'Generate test coverage report'
        },
        {
            'name': '🔧 Code Quality',
            'cmd': ['python', '-m', 'flake8', 'api_server.py', '--max-line-length=120'],
            'description': 'Check code quality with flake8'
        },
        {
            'name': '🛡️ Security Check',
            'cmd': ['python', '-m', 'bandit', '-r', '.', '-f', 'json'],
            'description': 'Security vulnerability scan'
        }
    ]
    
    results = []
    
    for test in test_commands:
        print(f"\n{test['name']}")
        print(f"📝 {test['description']}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                test['cmd'],
                env=test_env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ PASSED ({duration:.2f}s)")
                results.append(('PASS', test['name'], duration))
            else:
                print(f"❌ FAILED ({duration:.2f}s)")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                results.append(('FAIL', test['name'], duration))
                
        except subprocess.TimeoutExpired:
            print("⏰ TIMEOUT (5 minutes)")
            results.append(('TIMEOUT', test['name'], 300))
        except FileNotFoundError:
            print("⚠️ SKIPPED (command not found)")
            results.append(('SKIP', test['name'], 0))
        except Exception as e:
            print(f"💥 ERROR: {e}")
            results.append(('ERROR', test['name'], 0))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    total_tests = len(results)
    passed = len([r for r in results if r[0] == 'PASS'])
    failed = len([r for r in results if r[0] == 'FAIL'])
    skipped = len([r for r in results if r[0] == 'SKIP'])
    errors = len([r for r in results if r[0] in ['ERROR', 'TIMEOUT']])
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️ Skipped: {skipped}")
    print(f"💥 Errors: {errors}")
    
    total_time = sum(r[2] for r in results)
    print(f"⏱️ Total Time: {total_time:.2f}s")
    
    # Detailed results
    print("\n📋 DETAILED RESULTS:")
    for status, name, duration in results:
        status_emoji = {
            'PASS': '✅',
            'FAIL': '❌', 
            'SKIP': '⚠️',
            'ERROR': '💥',
            'TIMEOUT': '⏰'
        }
        print(f"{status_emoji[status]} {name} ({duration:.2f}s)")
    
    # Success criteria
    if failed == 0 and errors == 0:
        print("\n🎉 ALL TESTS PASSED! API IS READY FOR PRODUCTION!")
        return True
    else:
        print(f"\n💔 {failed + errors} TESTS FAILED. PLEASE FIX ISSUES BEFORE DEPLOYMENT.")
        return False

def install_test_dependencies():
    """📦 Install test dependencies"""
    print("📦 Installing test dependencies...")
    
    test_deps = [
        'pytest>=7.4.3',
        'pytest-cov>=4.1.0',
        'pytest-asyncio>=0.21.1',
        'flake8>=6.0.0',
        'bandit>=1.7.5',
        'httpx>=0.25.2'
    ]
    
    for dep in test_deps:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                         check=True, capture_output=True)
            print(f"✅ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install {dep}")

def main():
    """🎯 Main test runner"""
    
    if len(sys.argv) > 1 and sys.argv[1] == '--install-deps':
        install_test_dependencies()
        return
    
    # Check if we're in the right directory
    if not Path('api_server.py').exists():
        print("❌ Please run this script from the Flask-app directory")
        sys.exit(1)
    
    # Run tests
    success = run_tests()
    
    if success:
        print("\n🚀 READY FOR DEPLOYMENT!")
        print("Next steps:")
        print("1. Set production environment variables")
        print("2. Run: python run_api_server.py prod")
        print("3. Monitor metrics at /metrics")
        print("4. Check health at /api/health")
        sys.exit(0)
    else:
        print("\n🔧 PLEASE FIX FAILING TESTS BEFORE DEPLOYMENT")
        sys.exit(1)

if __name__ == '__main__':
    main()