#!/usr/bin/env python3
"""
Update Label Studio Token in config.yaml
"""

import yaml
import sys

def update_token(new_token):
    """Update the token in config.yaml"""
    
    try:
        # Read current config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Update token
        config['labelstudio']['auth']['token'] = new_token
        
        # Write back to file
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print("✅ Token updated successfully in config.yaml")
        return True
        
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 update_token.py <new_token>")
        sys.exit(1)
    
    new_token = sys.argv[1]
    update_token(new_token) 