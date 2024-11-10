import os
import sys
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set the path to your Google Cloud service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/external2/workspace/google_service_account_files/stream2action-773bd306c0f0.json"

try:
    from youtubesearchpython import VideosSearch
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "youtube-search-python"])
    from youtubesearchpython import VideosSearch

try:
    import yt_dlp
except ImportError:
    print("Installing yt-dlp...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
    import yt_dlp

try:
    from google.cloud import storage
except ImportError:
    print("Installing google-cloud-storage...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-cloud-storage"])
    from google.cloud import storage

def sanitize_filename(filename):
    """Convert spaces and special characters to underscores in a filename"""
    sanitized = re.sub(r'[^\w\-_.]', '_', str(filename).replace(' ', '_'))
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized.strip('_')

def upload_to_gcs(bucket_name, source_file_path, destination_blob_name, output_folder):
    """Uploads a file to the specified Google Cloud Storage bucket and saves the URI."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_path)
    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
    print(f"Uploaded to {gcs_uri}")

    # Write the GCS URI to a file in the output folder
    with open(os.path.join(output_folder, 'uploaded_files_uris.txt'), 'a') as uri_file:
        uri_file.write(gcs_uri + '\n')

def download_content(url, download_folder, bucket_name, format_type='mp4'):
    """
    Download either video or audio content from YouTube and upload to GCS bucket.
    format_type: 'mp4' for video or 'mp3' for audio
    """
    def hook(d):
        if d['status'] == 'finished':
            print(f"Downloaded: {d['filename']}")
            # Upload the downloaded file to GCS bucket
            upload_to_gcs(bucket_name, d['filename'], os.path.basename(d['filename']), download_folder)
        elif d['status'] == 'downloading':
            print(f"Downloading {d['_percent_str']} of {d['filename']} at {d['_speed_str']} ETA: {d['_eta_str']}")

    if format_type == 'mp4':
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'merge_output_format': 'mp4',
            'outtmpl': os.path.join(download_folder, '%(title)s.%(ext)s'),
            'quiet': True,
            'no-warnings': True,
            'noplaylist': True,
            'progress_hooks': [hook],
            'restrictfilenames': True,
            'windowsfilenames': True,
            'overwrites': False
        }
    else:  # mp3
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(download_folder, '%(title)s.%(ext)s'),
            'quiet': True,
            'no-warnings': True,
            'noplaylist': True,
            'progress_hooks': [hook],
            'restrictfilenames': True,
            'windowsfilenames': True,
            'overwrites': False
        }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get('title'):
                info['title'] = sanitize_filename(info['title'])
            ydl.download([url])
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def search_and_download(query, num_results, download_folder, bucket_name, format_type='mp4', max_workers=5):
    """Search for and download videos/audio from YouTube, then upload to GCS."""
    query_folder_name = sanitize_filename(query)
    final_download_folder = os.path.join(download_folder, query_folder_name)
    os.makedirs(final_download_folder, exist_ok=True)
    
    try:
        search = VideosSearch(query, limit=num_results)
        results = search.result()['result']
        video_urls = [result['link'] for result in results]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(download_content, url, final_download_folder, bucket_name, format_type) 
                for url in video_urls
            ]
            for future in as_completed(futures):
                future.result()
    except Exception as e:
        print(f"Error processing query '{query}': {e}")

def main():
    parser = argparse.ArgumentParser(description='Download YouTube videos or audio and upload to Google Cloud Storage')
    parser.add_argument('format', choices=['mp4', 'mp3'],
                      help='Download format: mp4 for video, mp3 for audio')
    parser.add_argument('--results', type=int, default=5,
                      help='Number of results to download per query (default: 5)')
    parser.add_argument('--workers', type=int, default=5,
                      help='Number of parallel downloads (default: 5)')
    parser.add_argument('--output', type=str, default='downloaded_content',
                      help='Base output directory (default: downloaded_content)')
    parser.add_argument('--query', type=str,
                      help='Single search query. If not provided, will use predefined queries list')
    parser.add_argument('--bucket_name', type=str, required=True,
                      help='Google Cloud Storage bucket name to upload files')

    args = parser.parse_args()

    # Predefined queries if no single query is provided
    queries = [args.query] if args.query else [
        "Hollywood movie scenes with intense family dinner discussions",
        "Classic courtroom scenes with prosecutor, defense, and witness dialogue",
        "Hollywood therapy session scenes with therapist and patient talking",
        "Iconic detective interview scenes with multiple suspects",
        "Hollywood family reunion scenes with siblings and relatives catching up",
        "Police interrogation scenes with detectives questioning suspects",
        "Team meeting scenes in action movies with characters planning a heist",
        "Romantic dinner scenes with couples having heartfelt conversations",
        "Classroom scenes with teachers and students discussing a topic",
        "Boardroom negotiation scenes with business executives debating",
        "Road trip scenes with friends sharing life experiences",
        "High-stakes poker game scenes with players exchanging banter",
        "Couples therapy scenes with counselor and couple in heated dialogue",
        "War room briefings with commanders and soldiers strategizing",
        "Mentor-mentee scenes with characters discussing life lessons",
        "Campfire scenes with friends revealing secrets and personal stories",
        "Hospital emergency room scenes with doctors and patients talking",
        "Detective partner discussions while examining clues",
        "Parent-teacher conference scenes with discussions about a student",
        "Classic diner scenes with friends chatting about relationships",
        "Celebrity interview scenes with journalists asking tough questions",
        "Family intervention scenes with relatives expressing concerns",
        "Teacher-principal discussions about student issues",
        "Sports locker room scenes with coaches motivating players",
        "Restaurant scenes with servers and customers discussing orders",
        "Family vacation scenes with parents and kids sharing experiences",
        "Town hall meetings with community members discussing issues",
        "Party scenes with friends discussing their personal lives",
        "Office scenes with boss and employees discussing project updates",
        "Detective scenes with suspects talking about alibis",
        "Spy movie briefings with agents planning a mission",
        "Group therapy scenes with participants sharing stories",
        "Reunion scenes with ex-partners discussing their past",
        "Hotel concierge scenes with staff assisting guests",
        "Courtroom settlement discussions with lawyers and clients",
        "Car ride scenes with characters having deep conversations",
        "Support group meetings with members sharing personal experiences",
        "Royal court scenes with king, queen, and advisors deliberating",
        "Prison yard scenes with inmates discussing life beyond bars",
        "Student-teacher conference scenes discussing academic progress",
        "Talk show interviews with hosts and guests exchanging insights",
        "Coffee shop scenes with strangers meeting for the first time",
        "Military command briefings with generals and soldiers",
        "Planning scenes for surprise parties with friends",
        "Hospital waiting room scenes with family discussing treatments",
        "Airplane scenes with flight attendants and passengers interacting",
        "Charity gala scenes with philanthropists discussing causes",
        "Museum scenes with curators explaining exhibits to visitors",
        "Political debate scenes with candidates discussing policies",
        "Group dinner party scenes with multiple guests talking",
    ]

    base_output_dir = args.output
    os.makedirs(base_output_dir, exist_ok=True)

    # Clear or create a file to store URIs of uploaded files
    with open(os.path.join(base_output_dir, 'uploaded_files_uris.txt'), 'w') as uri_file:
        uri_file.write("")

    for query in queries:
        print(f"\nProcessing query: {query}")
        search_and_download(
            query=query,
            num_results=args.results,
            download_folder=base_output_dir,
            bucket_name=args.bucket_name,
            format_type=args.format,
            max_workers=args.workers
        )
        print(f"Completed downloads and uploads for query: {query}")

if __name__ == "__main__":
    main()
