import os
import sys
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# Set the path to your Google Cloud service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/external2/workspace/google_service_account_files/stream2action-773bd306c0f0.json"

try:
    from youtubesearchpython import VideosSearch
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "youtube-search-python"])
    from youtubesearchpython import VideosSearch

try:
    import yt_dlp
except ImportError:
    print("Installing yt-dlp...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
    import yt_dlp

try:
    from google.cloud import storage
except ImportError:
    print("Installing google-cloud-storage...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-cloud-storage"])
    from google.cloud import storage

def sanitize_filename(filename):
    """Convert spaces and special characters to underscores in a filename"""
    sanitized = re.sub(r'[^\w\-_.]', '_', str(filename).replace(' ', '_'))
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized.strip('_')

def get_video_duration(filepath):
    """Returns the duration of a video in seconds."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        duration = float(result.stdout)
        return duration
    except Exception as e:
        print(f"Error getting duration for {filepath}: {e}")
        return None

def trim_video(input_path, output_path, max_duration=1800):
    """Trim video to max_duration in seconds (default 30 minutes)."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-t", str(max_duration), "-c", "copy", output_path],
            check=True
        )
        print(f"Trimmed video saved to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error trimming video {input_path}: {e}")
        return None

def convert_to_wav(input_path, output_path):
    """Convert audio (e.g., WebM) to 16kHz mono WAV PCM audio."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", output_path],
            check=True
        )
        print(f"Converted audio saved to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path} to WAV: {e}")
        return None

def upload_to_gcs(bucket_name, source_file_path, destination_blob_name, output_folder, dataname="soccer_data"):
    """Uploads a file to the specified Google Cloud Storage bucket under a folder and saves the URI."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    folder = dataname + "/"
    full_destination_blob_name = f"{folder}{destination_blob_name}"
    
    blob = bucket.blob(full_destination_blob_name)
    blob.upload_from_filename(source_file_path)
    
    gcs_uri = f"gs://{bucket_name}/{full_destination_blob_name}"
    print(f"Uploaded to {gcs_uri}")

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'uploaded_files_uris.txt'), 'a') as uri_file:
        uri_file.write(gcs_uri + '\n')
    
    #os.remove(source_file_path)

def download_content(url, download_folder, bucket_name, format_type='mp4'):
    """
    Download either video or audio content from YouTube and upload to GCS bucket.
    format_type: 'mp4' for video or 'mp3' for audio
    """
    def hook(d):
        if d['status'] == 'finished':
            print(f"Downloaded: {d['filename']}")
            # Check video duration
            duration = get_video_duration(d['filename'])
            if duration and duration > 1800:  # 30 minutes in seconds
                # Trim the video
                trimmed_path = os.path.join(download_folder, "trimmed_" + os.path.basename(d['filename']))
                trimmed_path = trim_video(d['filename'], trimmed_path)
            else:
                trimmed_path = d['filename']



            # Convert video to 16kHz WAV PCM audio
            print("\n\n Trimmed path: ", trimmed_path, "\n\n\n")
            
            audio_path = trimmed_path.replace(".webm", "_16k.wav")
            trimmer_mp4_path = trimmed_path.replace(".webm", ".mp4")
            #audio_path = os.path.join(download_folder, sanitize_filename(os.path.splitext(trimmed_path)[0]) + "_16k.wav")
            audio_path = convert_to_wav(trimmed_path, audio_path)
            print("Wave file path: ", audio_path)
            if audio_path:
                # Upload the WAV audio file to GCS
                # Upload the video file to GCS
                upload_to_gcs(bucket_name, trimmed_path, os.path.basename(trimmed_path), download_folder)
                upload_to_gcs(bucket_name, audio_path, os.path.basename(audio_path), download_folder)
                upload_to_gcs(bucket_name, trimmer_mp4_path, os.path.basename(audio_path), download_folder)
                
            #os.remove(trimmed_path)
            #os.remove(audio_path)
            #os.remove(d['filename'])
        #elif d['status'] == 'downloading':
        #    continue
            #print(f"Downloading {d['_percent_str']} of {d['filename']} at {d['_speed_str']} ETA: {d['_eta_str']}")

    ydl_opts = {
        'format': 'bestvideo[height<=480]+bestaudio/best',  # Limits to best video at 480p or lower with best audio
        'merge_output_format': 'mp4',
        'outtmpl': os.path.join(download_folder, '%(title)s.%(ext)s'),
        'quiet': True,
        'no-warnings': True,
        'noplaylist': True,
        'progress_hooks': [hook],
        'restrictfilenames': True,
        'windowsfilenames': True,
        'overwrites': False,
        'cookiefile': '/external2/workspace/youtube.com_cookies.txt'
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
        
    #os.remove(os.path.join(final_download_folder, '*.mp4'))
    #os.remove(os.path.join(final_download_folder, '*.wav'))

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

    queries = [args.query] if args.query else [
        "Full soccer game Barcelona vs Real Madrid",
        "Full soccer match Manchester United vs Liverpool",
        "Full match UEFA Champions League Final",
        "Full soccer game World Cup Final",
        "Full soccer game Brazil vs Argentina",
        "Full Premier League game Manchester City vs Chelsea",
        "Full match FC Barcelona Champions League",
        "Full soccer game Chelsea vs Manchester City",
        "La Liga full game highlights",
        "Full soccer match Arsenal vs Tottenham",
        "Bundesliga full game Bayern Munich vs Borussia Dortmund",
        "Full game highlights France vs Germany Euro",
        "Copa America full match Brazil vs Chile",
        "Serie A full match Juventus vs AC Milan",
        "Full game MLS Cup Final",
        "World Cup qualifying match full replay",
        "International friendly full game USA vs Mexico",
        "Africa Cup of Nations full match",
        "Full game PSG vs Marseille",
        "EPL full game Chelsea vs Arsenal",
        "Full soccer match Spain vs Portugal",
        "Asian Cup full game Japan vs South Korea",
        "FIFA World Cup full game replay",
        "Full match Barcelona Champions League",
        "Euro Cup full game Italy vs England",
        "Premier League full game Manchester City vs Chelsea",
        "CONCACAF Gold Cup full match USA vs Canada",
        "Serie A full match Inter Milan vs Juventus",
        "El Clasico full match Real Madrid vs Barcelona",
        "Full game Argentina vs Uruguay Copa America",
        "Full match English Championship playoff final",
        "Full game Ligue 1 PSG vs Lyon",
        "FA Cup final full game Arsenal vs Manchester United",
        "Women's World Cup full match USA vs Netherlands",
        "Full soccer game Spain vs Italy",
        "Full match Manchester United Champions League",
        "Copa Libertadores full game Boca Juniors vs River Plate",
        "World Cup group stage full match",
        "Olympic soccer final full game",
        "Full soccer game Netherlands vs Germany",
        "FIFA Club World Cup full match",
        "Full match Liverpool Champions League",
        "Full game Premier League Liverpool vs Manchester United",
        "Full match Real Madrid Champions League",
        "Women's Euro full game England vs Germany",
        "Nations League full game Portugal vs France",
        "Classic World Cup full match Italy vs Brazil",
        "Full game Barcelona vs Atletico Madrid La Liga",
        "Full game African Cup of Nations",
        "Full soccer game Copa America Brazil vs Argentina",
        "Full soccer match World Cup Final 2022",
    ]

    base_output_dir = args.output
    os.makedirs(base_output_dir, exist_ok=True)

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
