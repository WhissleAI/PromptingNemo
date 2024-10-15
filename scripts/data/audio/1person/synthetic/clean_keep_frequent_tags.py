import os
import re
from collections import Counter

def read_tagged_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def count_tags(lines):
    tag_pattern = re.compile(r'T\d+')
    tags = []
    for line in lines:
        tags.extend(tag_pattern.findall(line))
    tag_counts = Counter(tags)
    return tag_counts

def filter_tags(tag_counts, threshold):
    filtered_tags = {tag for tag, count in tag_counts.items() if count > threshold}
    return filtered_tags

def filter_lines(lines, clean_lines, filtered_tags):
    filtered_lines = []
    filtered_clean_lines = []
        
    for num, line in enumerate(lines):
        if all(tag in filtered_tags for tag in re.findall(r'T\d+', line)):
            filtered_lines.append(line)
            filtered_clean_lines.append(clean_lines[num])
            
    return filtered_lines, filtered_clean_lines

def filter_alltags(alltags_path, filtered_tags):
    with open(alltags_path, 'r') as file:
        lines = file.readlines()
    
    filtered_alltags = []
    for line in lines:
        tag = line.split()[1]  # Ensure we are looking at the second column
        if tag in filtered_tags:
            filtered_alltags.append(line)
    
    # Debug: Print filtered_alltags
    print("Filtered alltags lines:", filtered_alltags)
    
    return filtered_alltags

def write_filtered_files(filtered_lines, filtered_alltags, output_tagged_path, output_alltags_path):
    with open(output_tagged_path, 'w') as file:
        file.writelines(filtered_lines)
    
    with open(output_alltags_path, 'w') as file:
        file.writelines(filtered_alltags)

def process_folder(tagged_file_folder, alltags_file, threshold):
    # Step 1: Count tags across all tagged_*_clean.txt files
    combined_tag_counts = Counter()
    file_paths = []
    
    for filename in os.listdir(tagged_file_folder):
        if filename.startswith('tagged_') and filename.endswith('_clean.txt'):
            file_path = os.path.join(tagged_file_folder, filename)
            file_paths.append(file_path)
            tagged_lines = read_tagged_file(file_path)
            combined_tag_counts.update(count_tags(tagged_lines))
    
    # Debug: Print combined tag counts
    print("Combined tag counts:", combined_tag_counts)
    
    # Step 2: Filter tags based on the combined frequency
    filtered_tags = filter_tags(combined_tag_counts, threshold)
    
    # Debug: Print filtered tags
    print("Filtered tags:", filtered_tags)
    
    # Step 3: Create filtered_alltags.txt
    filtered_alltags_lines = filter_alltags(alltags_file, filtered_tags)
    filtered_alltags_path = os.path.join(tagged_file_folder, 'filtered_alltags.txt')
    
    with open(filtered_alltags_path, 'w') as file:
        file.writelines(filtered_alltags_lines)
    
    print(f"Filtered alltags file has been written to {filtered_alltags_path}")

    # Step 4: Create filtered versions of tagged_*_clean.txt files
    for file_path in file_paths:
        tagged_lines = read_tagged_file(file_path)
        
        clean_file_path = file_path.replace('_clean', '_notag')
        clean_lines = read_tagged_file(clean_file_path)
        
        filtered_tagged_lines, filtered_clean_lines = filter_lines(tagged_lines, clean_lines, filtered_tags)
        
        output_tagged_path = os.path.join(tagged_file_folder, 'filtered_' + os.path.basename(file_path))
        with open(output_tagged_path, 'w') as file:
            file.writelines(filtered_tagged_lines)
        
        output_clean_path = os.path.join(tagged_file_folder, 'filtered_' + os.path.basename(clean_file_path))
        with open(output_clean_path, 'w') as file:
            file.writelines(filtered_clean_lines)
        
        
        print(f"Filtered file has been written to {output_tagged_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python script.py <tagged_file_folder> <alltags_file> <threshold>")
        sys.exit(1)
    
    tagged_file_folder = sys.argv[1]
    alltags_file = sys.argv[2]
    threshold = int(sys.argv[3])

    process_folder(tagged_file_folder, alltags_file, threshold)

