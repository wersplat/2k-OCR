#!/usr/bin/env python3
"""
Test Label Studio Token
"""

import requests
import json

def test_token():
    """Test the Label Studio token"""
    
    # Get token from user
    print("🔑 Please enter your Label Studio token:")
    token = input().strip()
    
    if not token:
        print("❌ No token provided")
        return
    
    headers = {
        'Authorization': f'Token {token}',
        'Content-Type': 'application/json'
    }
    
    print("🧪 Testing Label Studio Token")
    print("=" * 40)
    
    # Test connection to Label Studio
    try:
        response = requests.get("http://localhost:8080/api/projects/", headers=headers, timeout=5)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Token is valid!")
            projects = response.json()
            print(f"📊 Found {len(projects)} projects")
            for project in projects:
                print(f"   • {project['title']} (ID: {project['id']})")
            
            print("\n🔧 To update config.yaml:")
            print(f"   Replace the token in config.yaml with: {token}")
            
        elif response.status_code == 401:
            print("❌ Token is invalid or expired")
            print("\n🔧 To get a new token:")
            print("1. Open Label Studio at http://localhost:8080")
            print("2. Go to Account & Settings")
            print("3. Click on 'Access Token'")
            print("4. Create a new token")
            print("5. Update the token in config.yaml")
        elif response.status_code == 404:
            print("⚠️ Label Studio API not found")
            print("   Make sure Label Studio is running at http://localhost:8080")
        else:
            print(f"⚠️ Unexpected response: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Label Studio")
        print("   Make sure Label Studio is running at http://localhost:8080")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_token() 