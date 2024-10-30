import csv
import os
import re
import subprocess
import json

# Path to the CSV file
filepath = "/root/workspace/ted-talks-download/data/TED_Talks_by_ID_plus-transcripts-and-LIWC-and-MFT-plus-views.csv"

# Output folder for cropped videos and manifest file
output_folder = "/external2/datasets/visual"  # Change this to your actual output folder path
# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)


# Function to convert time string to seconds
def time_to_seconds(time_str):
    try:
        minutes, seconds = map(int, time_str.strip().split(':'))
        return minutes * 60 + seconds
    except ValueError as e:
        print(f"Error parsing time: {time_str}")
        raise e

# Function to parse transcript into segments
def parse_transcript(transcript):
    segments = []
    transcript = transcript.replace("\n\n", "\n")
    parts = transcript.split("\n\n")
    print(parts[:10])
    for i in range(0, len(parts) - 1, 2):
        start_time = parts[i].strip()
        text = parts[i + 1].strip()
        
        if i + 2 < len(parts):
            end_time = parts[i + 2].strip()
            # Check if the end_time is actually a time or part of the text
            if re.match(r'^\d+:\d+$', end_time):
                next_start_time = end_time
                end_time = time_to_seconds(next_start_time)
            else:
                end_time = None
        else:
            end_time = None
        
        text = text.replace("\n"," ")
        
        segments.append({
            'text': text,
            'start_time': time_to_seconds(start_time),
            'end_time': end_time
        })
    
    return segments

def list_formats(url):
    result = subprocess.run(f"yt-dlp -F {url}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error listing formats: {result.stderr.decode()}")
        return None
    formats = result.stdout.decode()
    print(formats)
    return formats

def download_combined_video(url, video_output_path):
    result = subprocess.run(f"yt-dlp -f 'best' -o '{video_output_path}' {url}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error downloading combined video: {result.stderr.decode()}")

def crop_segment(input_path, output_path, start_time, end_time):
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ss', str(start_time), '-to', str(end_time),
        '-c', 'copy', output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error cropping segment: {result.stderr.decode()}")
    else:
        print(f"Cropped segment: {output_path}")

def verify_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    duration = result.stdout.decode().strip()
    if not duration:
        print(f"Error verifying file: {file_path}")
        print(result.stderr.decode())
        return False
    print(f"Verified file: {file_path}, duration: {duration} seconds")
    return True

# Process the CSV file
with open(filepath, mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        
        rowid = row['\ufeffid']
        print(rowid)
        url = row['URL']
        transcript = row['transcript']
        
        print(row.keys())
        
        transcript_url = row['transcript_URL']
        print("transcript_url:", transcript_url)
        #print(transcript)
        
        
        # Path for the downloaded combined video file
        combined_video_path = os.path.join(output_folder, str(rowid) + ".mp4")
        
        # List available formats
        list_formats(url)
        
        # Download the combined video
        download_combined_video(url, combined_video_path)



