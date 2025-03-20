#!/usr/bin/env python

"""
Fix Pinecone connectivity issues in Replit environment.
"""

import os
import re

# Files to modify
TOOL_REGISTRY_FILE = "atia/tool_registry/registry.py"
CONFIG_FILE = "atia/config.py"

def update_tool_registry():
    """Add Replit environment detection to the tool registry."""
    with open(TOOL_REGISTRY_FILE, 'r') as f:
        content = f.read()

    # Find the __init__ method
    init_pattern = r'def __init__\(self\):'
    new_init = """def __init__(self):
        \"\"\"Initialize the Tool Registry.\"\"\"
        self._tools = {}  # In-memory storage for Phase 2
        self._pinecone_initialized = False
        self._pinecone_index = None

        # Check if running in Replit environment
        in_replit = 'REPL_ID' in os.environ or 'REPL_OWNER' in os.environ
        if in_replit:
            self.logger.warning("Running in Replit environment - disabling Pinecone integration")
            return

        # Initialize Pinecone if API key is available
        if hasattr(settings, 'pinecone_api_key') and settings.pinecone_api_key:
            self._init_pinecone()"""

    # Replace the method
    if re.search(init_pattern, content):
        content = re.sub(r'def __init__\(self\):.*?self\._init_pinecone\(\)', new_init, content, flags=re.DOTALL)

        # Write updated content
        with open(TOOL_REGISTRY_FILE, 'w') as f:
            f.write(content)

        print(f"✅ Updated {TOOL_REGISTRY_FILE}")
    else:
        print(f"❌ Could not find __init__ method in {TOOL_REGISTRY_FILE}")

def update_config():
    """Update config to disable vector DB by default."""
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()

    # Update the vector_db setting
    vector_pattern = r'enable_vector_db: bool = os.getenv\("ENABLE_VECTOR_DB",[^)]*\)'

    if re.search(vector_pattern, content):
        content = re.sub(
            vector_pattern,
            'enable_vector_db: bool = os.getenv("ENABLE_VECTOR_DB", "False").lower() == "true"',
            content
        )

        # Write updated content
        with open(CONFIG_FILE, 'w') as f:
            f.write(content)

        print(f"✅ Updated {CONFIG_FILE}")
    else:
        print(f"❌ Could not find vector_db setting in {CONFIG_FILE}")

def main():
    """Apply all fixes."""
    print("🔧 Applying Replit-specific fixes for Pinecone...")
    update_tool_registry()
    update_config()
    print("\n✅ ATIA has been configured to work optimally in Replit!")
    print("   Pinecone vector storage will be automatically disabled when running in Replit.")

if __name__ == "__main__":
    main()
    