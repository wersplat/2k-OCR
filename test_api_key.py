#!/usr/bin/env python3
"""
Test Label Studio API Key
"""

import requests
import json

def test_api_key():
    """Test different API key formats"""
    
    print("🔑 Label Studio API Key Tester")
    print("=" * 40)
    
    # Get API key from user
    print("Enter your API key (should be shorter, like: d6f8a2622d39e9d89ff0dfef1a80ad877f4ee9e3):")
    api_key = input().strip()
    
    if not api_key:
        print("❌ No API key provided")
        return
    
    print(f"\n🧪 Testing API key: {api_key[:10]}...")
    
    # Test different authentication methods
    methods = [
        {
            "name": "Token Authorization",
            "headers": {'Authorization': f'Token {api_key}', 'Content-Type': 'application/json'}
        },
        {
            "name": "X-API-Key Header", 
            "headers": {'X-API-Key': api_key, 'Content-Type': 'application/json'}
        },
        {
            "name": "Bearer Token",
            "headers": {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
        }
    ]
    
    for method in methods:
        print(f"\n🔍 Testing: {method['name']}")
        try:
            response = requests.get("http://localhost:8080/api/projects/", 
                                  headers=method['headers'], timeout=5)
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print("   ✅ SUCCESS!")
                projects = response.json()
                print(f"   📊 Found {len(projects)} projects")
                for project in projects:
                    print(f"      • {project['title']} (ID: {project['id']})")
                
                print(f"\n🎉 {method['name']} works!")
                print("🔧 Update your config.yaml with this API key")
                return True
                
            elif response.status_code == 401:
                print("   ❌ Authentication failed")
            else:
                print(f"   ⚠️ Unexpected response: {response.text[:100]}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n❌ None of the authentication methods worked")
    print("\n📋 Make sure you have the correct API key:")
    print("1. Go to Label Studio: http://localhost:8080")
    print("2. Account & Settings > Access Tokens")
    print("3. Generate a new API key")
    print("4. The key should be shorter (not a long JWT token)")
    
    return False

if __name__ == "__main__":
    test_api_key() 