#!/usr/bin/env python3
"""
Test Label Studio Bearer Token Authentication
Based on: https://labelstud.io/guide/api.html
"""

import requests
import json

def test_bearer_token():
    """Test Bearer token authentication"""
    
    print("🔑 Label Studio Bearer Token Tester")
    print("=" * 40)
    print("Based on: https://labelstud.io/guide/api.html")
    print()
    
    # Get access token from user
    print("Enter your access token from Account & Settings:")
    print("(Should be shorter, not a long JWT token)")
    access_token = input().strip()
    
    if not access_token:
        print("❌ No access token provided")
        return
    
    print(f"\n🧪 Testing Bearer token: {access_token[:10]}...")
    
    # Test Bearer authentication (as per Label Studio docs)
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get("http://localhost:8080/api/projects/", 
                              headers=headers, timeout=5)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS! Bearer authentication works!")
            projects = response.json()
            print(f"📊 Found {len(projects)} projects")
            for project in projects:
                print(f"   • {project['title']} (ID: {project['id']})")
            
            print("\n🎉 Your access token is valid!")
            print("🔧 Update your config.yaml with this token")
            return True
            
        elif response.status_code == 401:
            print("❌ Authentication failed")
            print("   Make sure you're using the access token from Account & Settings")
            print("   Not a refresh token (those are longer JWT tokens)")
        else:
            print(f"⚠️ Unexpected response: {response.text[:100]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n📋 To get the correct access token:")
    print("1. Open Label Studio: http://localhost:8080")
    print("2. Click user icon in upper right")
    print("3. Click 'Account & Settings'")
    print("4. Look for 'Access Token' section")
    print("5. Copy the access token (shorter, not JWT format)")
    
    return False

if __name__ == "__main__":
    test_bearer_token() 