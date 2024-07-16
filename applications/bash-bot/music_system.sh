#!/bin/bash

# Declare API keys for Spotify and YouTube
SPOTIFY_CLIENT_ID="your_spotify_client_id"
SPOTIFY_CLIENT_SECRET="your_spotify_client_secret"
YOUTUBE_API_KEY="your_youtube_api_key"

# Function to get Spotify access token
get_spotify_access_token() {
  local access_token=$(curl -s -X POST "https://accounts.spotify.com/api/token" \
    -H "Authorization: Basic $(echo -n "$SPOTIFY_CLIENT_ID:$SPOTIFY_CLIENT_SECRET" | base64)" \
    -d grant_type=client_credentials | jq -r '.access_token')
  echo "$access_token"
}

# Function to search and play music on Spotify
play_spotify() {
  local song_name="$1"
  local access_token=$(get_spotify_access_token)

  local track_id=$(curl -s -X GET "https://api.spotify.com/v1/search?q=${song_name// /%20}&type=track&limit=1" \
    -H "Authorization: Bearer $access_token" | jq -r '.tracks.items[0].id')

  if [ -n "$track_id" ]; then
    echo "Playing $song_name on Spotify..."
    # Command to play the track on Spotify
    spotify play --uri="spotify:track:$track_id"
  else
    echo "Song not found on Spotify."
  fi
}

# Function to search and play music on YouTube
play_youtube() {
  local song_name="$1"
  local video_id=$(curl -s "https://www.googleapis.com/youtube/v3/search?part=snippet&q=${song_name// /%20}&key=$YOUTUBE_API_KEY&type=video&maxResults=1" \
    | jq -r '.items[0].id.videoId')

  if [ -n "$video_id" ]; then
    echo "Playing $song_name on YouTube..."
    # Command to play the video on YouTube
    youtube-dl -o - "https://www.youtube.com/watch?v=$video_id" | mpv -
  else
    echo "Video not found on YouTube."
  fi
}

# Function to pause Spotify
pause_spotify() {
  echo "Pausing Spotify..."
  # Command to pause Spotify
  spotify pause
}

# Function to pause YouTube
pause_youtube() {
  echo "Pausing YouTube..."
  # Command to pause YouTube
  pkill -STOP mpv
}

# Function to skip Spotify
skip_spotify() {
  echo "Skipping current song on Spotify..."
  # Command to skip Spotify
  spotify next
}

# Function to skip YouTube
skip_youtube() {
  echo "Skipping current song on YouTube..."
  # Command to skip YouTube
  pkill -CONT mpv
}

# Main function to handle input
main() {
  local action="$1"
  local platform="$2"
  local media="$3"

  case "$platform" in
    "Spotify")
      case "$action" in
        "play")
          play_spotify "$media"
          ;;
        "pause")
          pause_spotify
          ;;
        "skip")
          skip_spotify
          ;;
        *)
          echo "Unsupported action: $action"
          exit 1
          ;;
      esac
      ;;
    "YouTube")
      case "$action" in
        "play")
          play_youtube "$media"
          ;;
        "pause")
          pause_youtube
          ;;
        "skip")
          skip_youtube
          ;;
        *)
          echo "Unsupported action: $action"
          exit 1
          ;;
      esac
      ;;
    *)
      echo "Unsupported platform: $platform"
      exit 1
      ;;
  esac
}

# Parse command-line arguments
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <action> <platform> [<media>]"
  exit 1
fi

main "$1" "$2" "$3"

