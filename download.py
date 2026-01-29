from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import csv
import requests
import yt_dlp
import statistics
from tqdm import tqdm


class DataDownloader:
    def __init__(self, api_key, niches, OUTPUT_FOLDER="data/raw/"):
        self.api_key = api_key
        self.niches = niches
        self.OUTPUT_FOLDER = OUTPUT_FOLDER
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.makedirs(self.OUTPUT_FOLDER)

    def channels_by_niche(self, niches, max_searches=10):
        """
        Searches YouTube for channels in provided niches.
        """
        youtube = build("youtube", "v3", developerKey=self.api_key)

        # We use a dictionary to automatically handle duplicates
        # Key = Channel ID, Value = Channel Title
        unique_channels = {}

        print(f"Starting search for {len(niches)} niches...\n")

        for niche in niches:
            print(f"🔎 Searching for: '{niche}'...")

            try:
                # The search().list() endpoint is the standard way to find channels
                request = youtube.search().list(
                    part="snippet",
                    type="channel",  # strictly search for channels
                    q=niche,  # the search term
                    maxResults=max_searches,
                    order="relevance"  # usually finds the most "popular" for that term
                )
                response = request.execute()

                for item in response.get('items', []):
                    channel_id = item['snippet']['channelId']
                    channel_title = item['snippet']['title']

                    # LOGIC: Check for duplicates
                    if channel_id not in unique_channels:
                        unique_channels[channel_id] = channel_title
                        print(f"   ✅ Added: {channel_title}")
                    else:
                        print(f"   ⚠️ Duplicate ignored: {channel_title}")

            except HttpError as e:
                print(f"   ❌ API Error: {e}")

        return unique_channels

    def get_long_form_data(self, channel_id):
        """Fetch all video metadata."""
        channel_url = f"https://www.youtube.com/channel/{channel_id}/videos"
        print(f"Fetching metadata from: {channel_url}...")

        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'sleep_interval': 1,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(channel_url, download=False)
                if 'entries' in info:
                    return list(info['entries'])
            except Exception as e:
                print(f"Error fetching data: {e}")
                return []
        return []

    def calculate_elastic_average(self, all_videos, current_index, neighbor_target=10):
        """
        Calculates average of 'neighbor_target' surrounding videos.
        Elastic Logic: Tries to get 5 before and 5 after.
        If not enough on one side, takes more from the other.
        """
        total_videos = len(all_videos)
        if total_videos <= 1:
            return None

        # 1. Define the ideal 'span' (Current video + neighbors)
        # If we want 10 neighbors, we need a span of 11 videos total.
        span_size = neighbor_target + 1

        # 2. Calculate ideal start point (centered)
        # For 10 neighbors, we want to start 5 slots back.
        half_window = neighbor_target // 2
        start_idx = current_index - half_window

        # 3. Elastic Adjustments
        # A. If start is negative (too new), shift window right to start at 0
        if start_idx < 0:
            start_idx = 0

        # B. Calculate end index based on start
        end_idx = start_idx + span_size

        # C. If end exceeds list (too old), shift window left to fit
        if end_idx > total_videos:
            end_idx = total_videos
            # Try to push start back to maintain span size
            start_idx = max(0, end_idx - span_size)

        # 4. Collect views from the window, SKIPPING the current video
        neighbor_views = []

        # Loop from start to end (exclusive)
        for i in range(start_idx, end_idx):
            if i == current_index:
                continue

            v = all_videos[i].get('view_count')
            if v is not None:
                neighbor_views.append(v)

        if not neighbor_views:
            return None

        return statistics.mean(neighbor_views)

    def download_thumbnail(self, video_id, url):
        try:
            if not url:
                url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(os.path.join(f'{self.OUTPUT_FOLDER}/thumbnails', f"{video_id}.jpg"), "wb") as f:
                    f.write(response.content)
        except:
            pass

    def process_channel(self, channel_id, neighbor_count=10):
        """
        Main processing function for a single channel.
        """
        # 1. Fetching the metadata
        videos = self.get_long_form_data(channel_id)
        if not videos:
            print(f"⚠️ No videos found or error for channel: {channel_id}")
            return

        print(f"✅ Found {len(videos)} videos. Processing...")

        # 2. Prepraring CSV file
        file_path = os.path.join(self.OUTPUT_FOLDER, f'{channel_id}.csv')

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Headers
            writer.writerow(['Video ID', 'Title', 'Current Views', 'Elastic Avg', 'V-Score'])

            # 3. Processing each video
            for i, video in tqdm(enumerate(videos), total=len(videos), desc=f"Processing {channel_id[:10]}..."):
                vid_id = video.get('id')
                title = video.get('title')
                current_views = video.get('view_count')

                # Fetching thumbnails
                thumbnails = video.get('thumbnails', [])
                thumb_url = thumbnails[-1].get('url') if thumbnails else None

                if not vid_id or current_views is None:
                    continue

                # 4. Calculating Elastic Average
                elastic_avg = self.calculate_elastic_average(videos, i, neighbor_count)

                # 5. Calculating V-Score (Virality Score)
                if elastic_avg and elastic_avg > 0:
                    v_score = current_views / elastic_avg
                    avg_str = f"{elastic_avg:.0f}"
                    score_str = f"{v_score:.2f}"
                else:
                    avg_str = "N/A"
                    score_str = "N/A"

                # 6. Downloading thumbnails
                self.download_thumbnail(vid_id, thumb_url)

                # 7. Writing to CSV
                writer.writerow([
                    vid_id,
                    title,
                    current_views,
                    avg_str,
                    score_str
                ])

        print(f"🏁 Finished! Data saved to {file_path}")
