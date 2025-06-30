#!/usr/bin/env python3
"""
Enhanced Flask API Server with Site24x7 Integration
"""

import os
import sys
from api_server import create_app

def main():
    """Main entry point with proper configuration"""
    # Set environment variables if not already set
    if not os.getenv('FLASK_ENV'):
        os.environ['FLASK_ENV'] = 'production'
    
    if not os.getenv('FLASK_RUN_PORT'):
        os.environ['FLASK_RUN_PORT'] = '5001'
    
    # Create the Flask app
    app = create_app()
    
    # Get configuration
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 5001))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    print(f"🚀 Starting Flask API Server on {host}:{port}")
    print(f"📊 Site24x7 integration: {'Enabled' if os.getenv('SITE24X7_API_KEY') else 'Disabled'}")
    
    # Run the application
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )

if __name__ == '__main__':
    main()
