#!/usr/bin/env python3
"""
Generate Label Studio Access Token from Personal Access Token
Based on: https://labelstud.io/guide/access_tokens
"""

import requests
import json
import yaml
from datetime import datetime, timezone
import jwt

def generate_access_token(personal_access_token):
    """Generate access token from personal access token"""
    
    print("🔄 Generating access token from personal access token...")
    
    try:
        # Step 1: Generate access token using refresh endpoint
        response = requests.post(
            "http://localhost:8080/api/token/refresh",
            headers={"Content-Type": "application/json"},
            json={"refresh": personal_access_token},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            access_token = data.get("access")
            
            if access_token:
                print("✅ Access token generated successfully!")
                
                # Decode and show token info
                try:
                    decoded = jwt.decode(access_token, options={"verify_signature": False})
                    exp = decoded.get("exp")
                    if exp:
                        exp_time = datetime.fromtimestamp(exp, tz=timezone.utc)
                        print(f"⏰ Token expires at: {exp_time}")
                        
                        # Check if expired
                        now = datetime.now(timezone.utc)
                        if exp_time > now:
                            print(f"✅ Token is valid for {exp_time - now}")
                        else:
                            print("❌ Token has expired")
                            return None
                except Exception as e:
                    print(f"⚠️ Could not decode token: {e}")
                
                return access_token
            else:
                print("❌ No access token in response")
                return None
        else:
            print(f"❌ Failed to generate access token: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error generating access token: {e}")
        return None

def test_access_token(access_token):
    """Test the generated access token"""
    
    print("\n🧪 Testing generated access token...")
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get("http://localhost:8080/api/projects/", 
                              headers=headers, timeout=5)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS! Access token works!")
            projects = response.json()
            print(f"📊 Found {len(projects)} projects")
            if isinstance(projects, list):
                for project in projects:
                    if isinstance(project, dict):
                        title = project.get('title', 'Unknown')
                        project_id = project.get('id', 'Unknown')
                        print(f"   • {title} (ID: {project_id})")
            return True
        else:
            print(f"❌ Access token test failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing access token: {e}")
        return False

def update_config_with_personal_token(personal_token):
    """Update config.yaml with personal access token"""
    
    try:
        # Read current config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Update with personal access token
        config['labelstudio']['auth']['token'] = personal_token
        
        # Write back to file
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print("✅ Updated config.yaml with personal access token")
        return True
        
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False

def main():
    """Main function"""
    
    print("🔑 Label Studio Personal Access Token Handler")
    print("=" * 50)
    print("Based on: https://labelstud.io/guide/access_tokens")
    print()
    
    # Get personal access token from user
    print("Enter your personal access token (JWT refresh token):")
    personal_token = input().strip()
    
    if not personal_token:
        print("❌ No personal access token provided")
        return
    
    # Generate access token
    access_token = generate_access_token(personal_token)
    
    if access_token:
        # Test the access token
        if test_access_token(access_token):
            print("\n🎉 Everything works!")
            print("🔧 Your personal access token is valid and can be used with the SDK")
            
            # Update config
            if update_config_with_personal_token(personal_token):
                print("✅ Config updated successfully!")
                print("\n🚀 You can now use the Label Studio integration!")
        else:
            print("\n❌ Access token test failed")
    else:
        print("\n❌ Could not generate access token")
        print("   Make sure your personal access token is valid")

if __name__ == "__main__":
    main() 