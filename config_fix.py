"""
Script to update the config.py file with new settings for Phase 5.
"""

import os
import re

# Path to the config.py file
CONFIG_FILE = 'atia/config.py'

# New settings to add
NEW_SETTINGS = """
    # Authentication settings
    atia_admin_password: str = os.getenv("ATIA_ADMIN_PASSWORD", "atia_admin")

    # Feedback system settings
    feedback_storage_dir: str = os.getenv("FEEDBACK_STORAGE_DIR", "data/feedback")
    feedback_min_samples: int = int(os.getenv("FEEDBACK_MIN_SAMPLES", "3"))
    feedback_analysis_ttl_hours: int = int(os.getenv("FEEDBACK_ANALYSIS_TTL_HOURS", "24"))
    feedback_enable_auto_analysis: bool = os.getenv("FEEDBACK_ENABLE_AUTO_ANALYSIS", "True").lower() == "true"
"""

def update_config_file():
    """Update the config.py file with new settings."""
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: {CONFIG_FILE} not found!")
        return False

    # Read the current file
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()

    # Check where to insert the new settings - look for the existing API Gateway settings
    settings_end = content.find("    # API Gateway settings")
    if settings_end == -1:
        # Look for another section to insert before
        settings_end = content.find("    # Logging settings")

    if settings_end == -1:
        # If still not found, look for tool registry settings
        settings_end = content.find("    # Tool Registry settings")

    if settings_end == -1:
        print("Could not find a suitable insertion point in the config file!")
        return False

    # Insert new settings before the found section
    new_content = content[:settings_end] + NEW_SETTINGS + content[settings_end:]

    # Write back to the file
    with open(CONFIG_FILE, 'w') as f:
        f.write(new_content)

    print(f"Successfully updated {CONFIG_FILE} with new settings!")
    return True

def manual_fix_instructions():
    """Print instructions for manual fixing."""
    print("\nTo manually fix the issue, add these lines to your Settings class in atia/config.py:")
    print(NEW_SETTINGS)
    print("\nAdd them before the API Gateway settings or another appropriate section.")

if __name__ == "__main__":
    success = update_config_file()
    if not success:
        manual_fix_instructions()