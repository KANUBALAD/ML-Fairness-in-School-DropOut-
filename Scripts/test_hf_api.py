#!/usr/bin/env python3
"""
Test the Google Gemini API with model listing capability
"""

import os
import argparse
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

def list_available_models(api_key):
    """List all available Gemini models"""
    genai.configure(api_key=api_key)
    
    print("\n🔍 Listing available models...")
    print("-" * 50)
    
    try:
        models = genai.list_models()
        available_models = []
        
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                print(f"✅ {model.name} (Supports generation)")
                available_models.append(model.name.split('/')[-1])
            else:
                print(f"❌ {model.name} (Doesn't support generation)")
        
        print("-" * 50)
        return available_models
        
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []

def test_gemini_models(api_key, models_to_test=None):
    """Test Gemini models"""
    
    genai.configure(api_key=api_key)
    
    if models_to_test is None:
        models_to_test = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.0-pro"
        ]
    
    prompt = "Generate a JSON object with name and age fields: "
    
    working_models = []
    
    for model in models_to_test:
        print(f"\n🧪 Testing model: {model}")
        print("-" * 50)
        
        try:
            # Initialize the model
            generative_model = genai.GenerativeModel(model)
            
            # Generate content
            response = generative_model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=100
                )
            )
            
            print(f"✅ Success!")
            print(f"Response: {response.text}")
            working_models.append(model)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)
    
    print(f"\n🎯 Working models: {working_models}")
    return working_models

def test_specific_model(api_key, model_name, prompt):
    """Test a specific Gemini model"""
    
    genai.configure(api_key=api_key)
    
    print(f"🧪 Testing model: {model_name}")
    print(f"💬 Prompt: {prompt}")
    print("=" * 50)
    
    try:
        # Initialize the model
        generative_model = genai.GenerativeModel(model_name)
        
        # Generate content
        response = generative_model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.7,
                max_output_tokens=200
            )
        )
        
        print(f"✅ Success!")
        print(f"Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Google Gemini API")
    parser.add_argument("--api_key", required=True, help="Google AI Studio API key")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Model to test")
    parser.add_argument("--prompt", default="Generate a JSON object:", help="Test prompt")
    parser.add_argument("--test_all", action="store_true", help="Test multiple models")
    parser.add_argument("--list_models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        available_models = list_available_models(args.api_key)
        if available_models and input("\nTest these models? (y/n): ").lower() == 'y':
            test_gemini_models(args.api_key, available_models)
    elif args.test_all:
        test_gemini_models(args.api_key)
    else:
        test_specific_model(args.api_key, args.model, args.prompt)