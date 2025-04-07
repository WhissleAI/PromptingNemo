import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_gemini_api():
    # Load environment variables
    load_dotenv()
    
    # Configure Gemini API
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    
    try:
        # Initialize the model
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Test prompt
        prompt = """
        Write a short essay about the importance of artificial intelligence in modern society. 
        Include three main points and a conclusion. Keep it under 300 words.
        """
        
        # Generate content
        print("Generating essay...\n")
        response = model.generate_content(prompt)
        
        print("API Response Status:", "Success" if response else "Failed")
        print("\nGenerated Essay:\n")
        print(response.text)
        
        return True
        
    except Exception as e:
        print("Error testing Gemini API:", str(e))
        return False

if __name__ == "__main__":
    print("Testing Gemini API Connection...")
    test_gemini_api()