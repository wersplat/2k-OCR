#!/usr/bin/env python3
"""
Get Label Studio Token Helper
"""

import requests
import json

def test_token_format():
    """Test different token formats"""
    
    print("🔍 Label Studio Token Helper")
    print("=" * 40)
    
    # Test 1: No token (should fail)
    print("\n1️⃣ Testing without token...")
    try:
        response = requests.get("http://localhost:8080/api/projects/")
        print(f"   Status: {response.status_code}")
        if response.status_code == 401:
            print("   ✅ Expected: Authentication required")
        else:
            print("   ❌ Unexpected response")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Empty token
    print("\n2️⃣ Testing with empty token...")
    try:
        headers = {'Authorization': 'Token '}
        response = requests.get("http://localhost:8080/api/projects/", headers=headers)
        print(f"   Status: {response.status_code}")
        if response.status_code == 401:
            print("   ✅ Expected: Invalid token")
        else:
            print("   ❌ Unexpected response")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n📋 Instructions to get Access Token:")
    print("1. Open Label Studio: http://localhost:8080")
    print("2. Log in with your credentials")
    print("3. Go to Account & Settings (click username in top right)")
    print("4. Look for 'Access Token' or 'API Token' section")
    print("5. Click 'Create Token' or 'Generate Token'")
    print("6. Make sure it says 'Access Token' (not 'Refresh Token')")
    print("7. Copy the token (should be shorter than refresh tokens)")
    print("\n💡 Tip: Access tokens are usually shorter and don't have 'refresh' in the payload")

if __name__ == "__main__":
    test_token_format() 