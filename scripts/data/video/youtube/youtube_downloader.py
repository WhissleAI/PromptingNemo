try:
    from youtubesearchpython import VideosSearch
except ImportError:
    print("Installing required packages...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "youtube-search-python"])
    from youtubesearchpython import VideosSearch

try:
    import yt_dlp
except ImportError:
    print("Installing yt-dlp...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
    import yt_dlp

import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

def sanitize_filename(filename):
    """Convert spaces and special characters to underscores in a filename"""
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r'[^\w\-_.]', '_', str(filename).replace(' ', '_'))
    # Replace multiple underscores with a single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    return sanitized.strip('_')

def download_content(url, download_folder, format_type='mp4'):
    """
    Download either video or audio content from YouTube
    format_type: 'mp4' for video or 'mp3' for audio
    """
    def hook(d):
        if d['status'] == 'finished':
            print(f"Downloaded: {d['filename']}")
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
            # Download info first to get the title
            info = ydl.extract_info(url, download=False)
            # Sanitize the title
            if info.get('title'):
                info['title'] = sanitize_filename(info['title'])
            # Download with sanitized title
            ydl.download([url])
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def search_and_download(query, num_results, download_folder, format_type='mp4', max_workers=5):
    """Search for and download videos/audio from YouTube"""
    # Create sanitized folder name for the query
    query_folder_name = sanitize_filename(query)
    final_download_folder = os.path.join(download_folder, query_folder_name)
    os.makedirs(final_download_folder, exist_ok=True)
    
    try:
        search = VideosSearch(query, limit=num_results)
        results = search.result()['result']
        video_urls = [result['link'] for result in results]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(download_content, url, final_download_folder, format_type) 
                for url in video_urls
            ]
            for future in as_completed(futures):
                future.result()
    except Exception as e:
        print(f"Error processing query '{query}': {e}")

def main():
    parser = argparse.ArgumentParser(description='Download YouTube videos or audio')
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

    args = parser.parse_args()

    # Predefined queries list
    queries = [
        "Hollywood movie scenes with intense family dinner discussions",
        "Classic courtroom scenes with prosecutor, defense, and witness dialogue",
        "Hollywood therapy session scenes with therapist and patient talking",
        "Iconic movie scenes with detective and suspect interview exchanges",
        "Hollywood family reunion scenes with multiple characters catching up",
        "Police interrogation scenes in Hollywood movies with back-and-forth dialogue",
        "Team meeting scenes from Hollywood action movies planning a heist",
        "Romantic dinner scenes in Hollywood movies with heartfelt conversations",
        "Hollywood classroom scenes with teacher and students discussing lessons",
        "Boardroom negotiation scenes from Hollywood movies with group discussions",
        "Hollywood road trip scenes with friends talking about life experiences",
        "High-stakes gambling scenes with players and dealers exchanging words",
        "Hollywood couples therapy scenes with counselor and couple discussing issues",
        "Army briefing scenes in Hollywood war movies with soldiers and commanders",
        "Classic movie scenes with mentor and mentee discussing life lessons",
        "Hollywood scenes of friends sharing secrets around a campfire",
        "Intense hospital scenes in Hollywood with doctor and patient dialogues",
        "Legendary Hollywood detective scenes with partner discussing clues",
        "Hollywood movie scenes with parents discussing their child's future",
        "Iconic Hollywood diner scenes with friends chatting about relationships",
        "High-profile interview scenes from Hollywood with journalist and celebrity",
        "Hollywood family intervention scenes with siblings talking about concerns",
        "Classic scenes from Hollywood with teacher and principal discussing students",
        "Hollywood sports locker room scenes with coach and players talking strategy",
        "Hollywood restaurant scenes with server and customers discussing orders",
        "Family vacation scenes in Hollywood movies with parents and kids talking",
        "Hollywood town hall scenes with community members discussing issues",
        "Hollywood party scenes with friends having personal conversations",
        "Old Hollywood office scenes with boss and employees discussing work",
        "Hollywood detective scenes with suspects talking about alibis",
        "Hollywood spy movie scenes with briefings and tactical discussions",
        "Therapy group scenes from Hollywood movies with counselor and attendees",
        "Hollywood reunion scenes with exes discussing their relationship",
        "Classic hotel scenes in Hollywood movies with concierge and guests",
        "Hollywood courtroom settlement discussions with lawyer and clients",
        "Classic car ride scenes from Hollywood movies with partners talking",
        "Hollywood support group scenes with members sharing personal stories",
        "Hollywood royalty scenes with king, queen, and advisor talking",
        "Hollywood prison scenes with inmates talking about life outside",
        "Hollywood student-teacher conference scenes discussing academic progress",
        "Hollywood talk show scenes with host and guest exchanging stories",
        "Hollywood coffee shop scenes with strangers meeting for the first time",
        "Hollywood military command scenes with generals discussing strategy",
        "Hollywood planning scenes with friends organizing a surprise party",
        "Hollywood hospital waiting room scenes with family discussing treatment",
        "Iconic Hollywood airplane scenes with passengers talking to flight attendants",
        "Hollywood charity gala scenes with socialites discussing causes",
        "Hollywood museum scenes with curator explaining art to visitors",
        "Hollywood political debate scenes with candidates discussing issues",
        "Hollywood dinner party scenes with friends and couples talking",
        "Hospital room scenes in Hollywood movies with doctors and family discussing treatment",
        "Hollywood office politics scenes with coworkers discussing promotions",
        "Hollywood courtroom defense strategy scenes with lawyers discussing case plans",
        "Hollywood movie scenes with family talking about financial issues",
        "Hollywood crime scene investigation with detectives discussing clues",
        "Hollywood debate scenes with politicians arguing policy",
        "Police and witness questioning scenes from Hollywood films",
        "Hollywood travel scenes with tourists talking to locals about destinations",
        "Hollywood graduation party scenes with students and parents discussing future plans",
        "Hollywood bar conversations between bartender and regular patrons",
        "Hollywood scenes of colleagues discussing business proposals in a coffee shop",
        "Hollywood apartment complex scenes with neighbors chatting in the hallway",
        "Hollywood reunion scenes with childhood friends reminiscing about the past",
        "Hollywood award ceremony acceptance speeches and backstage interviews",
        "Hollywood movie scenes with priests and congregation members discussing faith",
        "Hollywood romantic balcony scenes with couples talking about their relationship",
        "Hollywood wedding preparation scenes with bride, groom, and family talking",
        "Hollywood law firm scenes with partners discussing a merger",
        "Hollywood college dorm scenes with roommates discussing their lives",
        "Hollywood travel agency scenes with clients discussing vacation plans",
        "Hollywood shopping mall scenes with friends talking while window shopping",
        "Hollywood father-son conversations about career paths",
        "Hollywood babysitting scenes with parents giving instructions to the sitter",
        "Hollywood diner scenes with truckers and waitresses talking about life on the road",
        "Hollywood airplane terminal scenes with passengers discussing flight delays",
        "Hollywood garage scenes with car mechanics discussing vehicle repairs",
        "Hollywood parent-teacher conference scenes with discussions about student progress",
        "Hollywood jewelry store scenes with customers discussing engagement ring options",
        "Hollywood boat tour scenes with tourists discussing local history",
        "Hollywood scenes with lawyers discussing contract negotiations over dinner",
        "Hollywood charity event scenes with organizers discussing fundraising goals",
        "Hollywood architect-client meetings discussing house renovation plans",
        "Hollywood board game night scenes with friends discussing rules and strategies",
        "Hollywood family holiday dinner scenes with relatives discussing old memories",
        "Hollywood mentoring sessions between senior executives and interns",
        "Hollywood courtroom witness preparation scenes with lawyers practicing testimonies",
        "Hollywood pet adoption scenes with shelter workers discussing animal care",
        "Hollywood therapy group scenes with participants sharing stories",
        "Hollywood rooftop garden scenes with neighbors discussing city life",
        "Hollywood library study group scenes with students discussing assignments",
        "Hollywood farm family scenes with parents discussing crop seasons",
        "Hollywood school assembly scenes with principal addressing students",
        "Hollywood cafeteria scenes with students gossiping about school events",
        "Hollywood landlord-tenant discussions about rent and repairs",
        "Hollywood tech startup pitch meetings with investors discussing funding",
        "Hollywood mystery dinner party scenes with guests discussing clues",
        "Hollywood elevator scenes with strangers making small talk",
        "Hollywood fashion show backstage scenes with designers and models discussing outfits",
        "Hollywood university lecture scenes with students and professors debating topics",
        "Hollywood movie reunion scenes with friends reminiscing",
        "Iconic dinner party scenes in Hollywood films with multiple guests",
        "Hollywood board game night scenes with friends strategizing",
        "Classic Hollywood family therapy sessions with parents and children",
        "Hollywood high school prom scenes with couples dancing and talking",
        "Hollywood movie launch party scenes with celebrities mingling",
        "Hollywood office party scenes with colleagues socializing",
        "Hollywood coffee break scenes with coworkers discussing projects",
        "Hollywood movie family road trip conversations with parents and kids",
        "Hollywood pub scenes with friends having casual conversations",
        "Hollywood holiday gathering scenes with extended family dialogues",
        "Hollywood movie award ceremony scenes with presenters and nominees",
        "Hollywood movie housewarming party scenes with new homeowners talking",
        "Hollywood movie book club scenes with members discussing a book",
        "Hollywood movie car dealership scenes with salespeople and customers",
        "Hollywood movie charity event scenes with organizers and donors",
        "Hollywood movie game show scenes with host and contestants",
        "Hollywood movie school assembly scenes with teachers and students",
        "Hollywood movie startup pitch scenes with entrepreneurs and investors",
        "Hollywood movie therapy group discussions with multiple participants",
        "Hollywood movie library scenes with librarians and visitors talking",
        "Hollywood movie neighborhood meeting scenes with residents discussing issues",
        "Hollywood movie art class scenes with instructor and students",
        "Hollywood movie cooking class scenes with chef and participants",
        "Hollywood movie business conference scenes with keynote speakers and attendees",
        "Hollywood movie travel agency scenes with agents and clients planning trips",
        "Hollywood movie courtroom trial scenes with judges, lawyers, and witnesses",
        "Hollywood movie newsroom scenes with journalists reporting and discussing",
        "Hollywood movie hotel lobby scenes with guests and staff conversing",
        "Hollywood movie airport terminal scenes with travelers and airline staff",
        "Hollywood movie wedding planning scenes with planners and couples",
        "Hollywood movie tech startup scenes with developers and managers brainstorming",
        "Hollywood movie political strategy meeting scenes with campaign team",
        "Hollywood movie startup office scenes with team members collaborating",
        "Hollywood movie fashion show backstage scenes with designers and models",
        "Hollywood movie sports team locker room scenes with coach and players",
        "Hollywood movie space mission briefing scenes with astronauts and controllers",
        "Hollywood movie detective office scenes with partners solving cases",
        "Hollywood movie family game night scenes with parents and children",
        "Hollywood movie neighborhood block party scenes with neighbors interacting",
        "Hollywood movie emergency room scenes with doctors and nurses communicating",
        "Hollywood movie high-stakes poker game scenes with players strategizing",
        "Hollywood movie space station scenes with crew members discussing missions",
        "Hollywood movie jazz club scenes with musicians and patrons conversing",
        "Hollywood movie startup brainstorming sessions with creative teams",
        "Hollywood movie family picnic scenes with relatives chatting",
        "Hollywood movie courtroom deliberation scenes with jurors discussing",
        "Hollywood movie newsroom deadline scenes with reporters collaborating",
        "Hollywood movie restaurant kitchen scenes with chefs coordinating",
        "Hollywood movie charity auction scenes with bidders and organizers",
        "Hollywood movie corporate board meeting scenes with executives debating",
        "Hollywood movie school project presentation scenes with students presenting",
        "Hollywood movie travel group planning scenes with tourists and guides",
        "Hollywood movie tech conference panels with experts discussing innovations",
        "Hollywood movie family therapy group scenes with multiple family members",
        "Hollywood movie home renovation scenes with contractors and homeowners",
        "Hollywood movie startup product launch scenes with team members celebrating",
        "Hollywood movie high school reunion scenes with alumni reconnecting",
        "Hollywood movie political debate scenes with candidates and moderators",
        "Hollywood movie art gallery opening scenes with artists and guests mingling",
        "Hollywood movie suburban neighborhood scenes with families interacting",
        "Hollywood movie startup office brainstorming scenes with diverse team",
        "Hollywood movie business lunch scenes with executives discussing deals",
        "Hollywood movie family emergency scenes with relatives coordinating actions",
        "Hollywood movie beach party scenes with friends socializing and talking",
        "Hollywood movie office workspace scenes with employees collaborating",
        "Hollywood movie fitness class scenes with instructor and participants",
        "Hollywood movie startup hackathon scenes with developers coding and communicating"
    ]

    # If a single query is provided, use only that
    if args.query:
        queries = [args.query]

    # Use the output directory path as-is
    base_output_dir = args.output
    os.makedirs(base_output_dir, exist_ok=True)

    # Process each query
    for query in queries:
        print(f"\nProcessing query: {query}")
        print(f"Downloading {args.format.upper()} content to: {base_output_dir}")
        search_and_download(
            query=query,
            num_results=args.results,
            download_folder=base_output_dir,
            format_type=args.format,
            max_workers=args.workers
        )
        print(f"Completed downloads for query: {query}")

if __name__ == "__main__":
    main()