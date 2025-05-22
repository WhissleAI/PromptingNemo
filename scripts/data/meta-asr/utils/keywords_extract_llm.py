import google.generativeai as genai
import os
from dotenv import load_dotenv
import nltk
from nltk.corpus import words
from nltk.probability import FreqDist
import re
import random

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

try:
    nltk.data.find('corpora/words')
    nltk.data.find('corpora/brown')  
except LookupError:
    nltk.download('words')
    nltk.download('brown')

english_vocab = set(w.lower() for w in words.words())
word_freq = FreqDist(w.lower() for w in nltk.corpus.brown.words())


def is_super_rare_english_word(word, threshold=10):
    """
    Checks if a word is extremely rare in English based on its frequency.
    Uses a much lower threshold to focus on super rare words.
    """
    word = word.lower()
    if word not in english_vocab:
        return False 

    return word_freq[word] < threshold


def extract_rare_keywords_gemini(text, max_keywords=300):
    """
    Extracts rare keywords from the given text using the Gemini API.
    """
    try:
        prompt = f"""You are a keyword extraction expert. From the following text, extract ONLY the EXTREMELY uncommon,
        very rarely used, or exceptionally difficult English words. Focus on the RAREST words possible.
        
        Do NOT include common words or even moderately uncommon words. Prioritize words that are:
        - Extremely rare in everyday language
        - Highly specialized technical terms
        - Archaic or obsolete vocabulary
        - Words with very low frequency in modern usage
        - Words that would be most difficult for speech recognition systems
        
        Output a maximum of {max_keywords} keywords. Provide them as a comma-separated list:

        {text}"""
        response = model.generate_content(prompt)
        response.resolve()

        if response.text:
            keywords_str = response.text.strip()
           
            keywords = []
            seen = set()
            for keyword in keywords_str.split(','):
                keyword = keyword.strip()
                if (keyword.lower() not in seen and
                        len(keyword) >= 3 and 
                        is_super_rare_english_word(keyword)):
                    keywords.append(keyword)
                    seen.add(keyword.lower())
            return keywords
        else:
            print(f"Gemini API returned empty response for text: '{text[:50]}...'")
            return []

    except Exception as e:
        print(f"Error extracting keywords with Gemini API: {e}")
        return []

def process_text_file(input_filepath, output_filepath, target_word_count=300, sample_size=500):
    """
    Reads the input text file, samples transcriptions in larger batches, extracts super rare
    keywords using Gemini, and writes a list of approximately target_word_count
    unique rare words to the output file.
    """
 
    all_rare_words = set()
    
    
    with open(input_filepath, 'r', encoding='utf-8') as infile:
        all_lines = infile.readlines()
    

    all_lines = all_lines[1:]
 
    valid_lines = []
    for line in all_lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('|')
        if len(parts) == 2:
            valid_lines.append(parts[1].strip())  
        else:
            print(f"Warning: Skipping line due to unexpected format: {line}")
    
    print(f"Total valid lines: {len(valid_lines)}")
    
  
    batch_count = 0
    total_lines_processed = 0
    
    while len(all_rare_words) < target_word_count and total_lines_processed < len(valid_lines):
        
        remaining_lines = valid_lines[total_lines_processed:]
        batch_size = min(sample_size, len(remaining_lines))
        
        if batch_size == 0:
            break
            
        batch_lines = remaining_lines[:batch_size]
        total_lines_processed += batch_size
        batch_count += 1
        
        
        combined_text = "\n".join(batch_lines)
        rare_keywords = extract_rare_keywords_gemini(combined_text)
        
        if rare_keywords:
         
            previous_count = len(all_rare_words)
            for word in rare_keywords:
                all_rare_words.add(word)
            new_words = len(all_rare_words) - previous_count
            print(f"Batch {batch_count}: Added {new_words} new rare words (total: {len(all_rare_words)})")
        else:
            print(f"Batch {batch_count}: No rare keywords found")
        
        print(f"Processed {total_lines_processed}/{len(valid_lines)} lines ({(total_lines_processed/len(valid_lines)*100):.1f}%)")

    
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        
        rare_words_list = sorted(list(all_rare_words))
        outfile.write(str(rare_words_list))
        
    print(f"Extracted {len(all_rare_words)} unique super rare words and saved to '{output_filepath}'")
    return all_rare_words


if __name__ == "__main__":
    input_file = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/transcriptions_hf_data.txt"
    output_file = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/rare_keyword_data.txt"
    
  
    rare_words = process_text_file(input_file, output_file, target_word_count=300, sample_size=500)
    
    print(f"Final count of unique super rare words: {len(rare_words)}")