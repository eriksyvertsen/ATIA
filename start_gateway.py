#!/usr/bin/env python
"""
Run the ATIA API Gateway for Replit environment.
"""

import uvicorn
import os
from atia.config import settings

# Ensure we're using the right settings for Replit
os.environ["API_GATEWAY_HOST"] = "0.0.0.0"
os.environ["API_GATEWAY_PORT"] = "8080"

# Optional: set a default admin password if not configured
if not os.environ.get("ATIA_ADMIN_PASSWORD"):
    os.environ["ATIA_ADMIN_PASSWORD"] = "atia_admin"

# Import the app after setting environment variables
from atia.api_gateway.gateway import app

if __name__ == "__main__":
    print(f"Starting API Gateway on 0.0.0.0:8080")
    print(f"Access using the 'Web' tab in Replit UI or your Replit URL")

    # Run the gateway
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        log_level="info"
    )