import subprocess
import os
import pandas as pd
import argparse
from tqdm import tqdm
import socks
import socket

def download_clip(ytid, start_seconds, end_seconds, output_path, video_frame_rate=5, audio_sample_rate=16000, proxy_url=None):
    """
    Downloads a clip from a YouTube video within a specified time range.

    Parameters:
    ytid (str): YouTube video ID.
    start_seconds (float): Start time of the clip in seconds.
    end_seconds (float): End time of the clip in seconds.
    output_path (str): Directory where the output video will be saved.
    video_frame_rate (int): Frame rate of the output video in frames per second.
    audio_sample_rate (int): Sample rate of the output audio in Hz.
    proxy_url (str): URL of the proxy to use for fetching the video.

    Returns:
    None
    """
    # Sanitize the YouTube ID to remove any unsafe characters for filenames
    ytid_strip = ''.join(e for e in ytid if e.isalnum())
    if os.path.exists(f"{output_path}/{ytid_strip}.mp4"):
        return
    # Convert start and end times to HH:MM:SS format
    start_time = f"{int(start_seconds // 3600):02d}:{int((start_seconds % 3600) // 60):02d}:{int(start_seconds % 60):02d}"
    duration = end_seconds - start_seconds

    # Construct the yt-dlp command to get the video and audio URLs
    command = ["yt-dlp","-i", "--youtube-skip-dash-manifest", "-g", f"https://youtu.be/{ytid}"]
    if proxy_url:
        command += ["--proxy", proxy_url]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if error:
        raise Exception(f"Error fetching URLs for {ytid}: {error.decode()}")

    # Decode the output and split by newlines to get separate URLs
    urls = output.decode("utf-8").strip().split("\n")
    if len(urls) != 2:
        # raise Exception(f"Error fetching URLs for {ytid}: {urls}")
        urls = [urls[0], urls[0]]
    
    # Construct the ffmpeg command to download the clip
    ffmpeg_command = [
        "ffmpeg",
        "-ss", start_time,
        "-i", urls[0],  # Video URL
        "-ss", start_time,
        "-i", urls[1],  # Audio URL
        "-ar", str(audio_sample_rate),  # Set audio sample rate
        "-ac", "1",  # Set audio channels to mono
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-c:a", "aac",
        "-t", str(duration),
        f"{output_path}/{ytid_strip}.mp4"
    ]

    process = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if process.returncode != 0:
        raise Exception(f"Error downloading {ytid}: {process.stderr.decode()}")

def read_csv(csv_path, l_idx, u_idx):
    df = pd.read_csv(csv_path)
    df = df.iloc[l_idx:u_idx]
    return df


def setup_socks_proxy(proxy_host='localhost', proxy_port=1080):
    # Set up a local SOCKS proxy
    socks.set_default_proxy(socks.SOCKS5, proxy_host, proxy_port)
    socket.socket = socks.socksocket

def download_clips(df, output_path, error_path):
    # Wrap the dataframe iteration with tqdm to add a progress bar
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Downloading Clips"):
        try:
            download_clip(row['YTID'], row['start_seconds'], row['end_seconds'], output_path)
        except Exception as e:
            with open(error_path, 'a') as f:
                f.write(f"ERROR: {e}\n")

def main(l_idx, u_idx,split):
    # csv_path = f"/disk1/audioset/annotations/audioset_{split}_strong.csv"
    csv_path = f"/disk1/audioset/annotations/missing_files_{split}.csv"
    output_path = f"/disk1/audioset/{split}_missing/videos"
    if split == 'eval' or True:
        error_path = f"/disk1/audioset/{split}_missing/download_logs/error.log"
    else:
        error_path = f"/disk1/audioset/{split}/download_logs/error_{l_idx}_{u_idx}.log"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.dirname(error_path), exist_ok=True)
    df = read_csv(csv_path, l_idx, u_idx)
    download_clips(df, output_path, error_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download clips from YouTube videos.')
    parser.add_argument('--l_idx', type=int, help='lower index', default=0)
    parser.add_argument('--u_idx', type=int, help='upper index', default=10000)
    parser.add_argument('--split', type=str, help='split', default='eval')
    args = parser.parse_args()
    # Set up the local SOCKS proxy
    # setup_socks_proxy()
    main(args.l_idx, args.u_idx, args.split)