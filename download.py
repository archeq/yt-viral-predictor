from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import requests
import yt_dlp
import pandas as pd
import numpy as np
from tqdm import tqdm


class DataDownloader:
    def __init__(self, api_key, niches, OUTPUT_FOLDER="data/raw/", WINDOW_SIZE=30, MIN_PERIODS=5):
        self.api_key = api_key
        self.niches = niches
        self.OUTPUT_FOLDER = OUTPUT_FOLDER
        self.WINDOW_SIZE = WINDOW_SIZE
        self.MIN_PERIODS = MIN_PERIODS
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.makedirs(self.OUTPUT_FOLDER)
        # Create thumbnails folder
        self.thumbnails_folder = os.path.join(self.OUTPUT_FOLDER, 'thumbnails')
        if not os.path.exists(self.thumbnails_folder):
            os.makedirs(self.thumbnails_folder)

    def calculate_log_v_score(self, df):
        """
        Calculates Logarithmic V-Score for a DataFrame with video data.
        Assumes input data is sorted: Row 0 = Newest video.
        """
        # Make a copy to avoid modifying the original
        df_calc = df.copy()

        # 1. Convert to logarithms (log1p is log(x + 1), safe for zeros)
        # Also ensure the column is numeric
        df_calc['Current Views'] = pd.to_numeric(df_calc['Current Views'], errors='coerce').fillna(0)
        df_calc['log_views'] = np.log1p(df_calc['Current Views'])

        # 2. Reverse the order (now index 0 is the OLDEST video)
        # This is necessary for .rolling() to "see" the past
        df_reversed = df_calc.iloc[::-1].copy()

        # 3. Calculate rolling statistics
        # .shift(1) is CRUCIAL - means "don't include current video in the average",
        # only those that came before it.
        indexer = df_reversed['log_views'].shift(1).rolling(
            window=self.WINDOW_SIZE,
            min_periods=self.MIN_PERIODS
        )

        df_reversed['baseline_median'] = indexer.median()
        df_reversed['baseline_std'] = indexer.std()

        # 4. Handle division by zero (when std = 0, i.e., perfectly constant views)
        # Substitute 1.0 as a safe divisor
        df_reversed['baseline_std'] = df_reversed['baseline_std'].replace(0, 1.0)

        # 5. V-Score formula
        # (Log(Target) - Log(Baseline_Median)) / Log(Baseline_Std)
        df_reversed['V-Score'] = (
            (df_reversed['log_views'] - df_reversed['baseline_median']) /
            df_reversed['baseline_std']
        )

        # 6. Clean up and return to original order
        df_final = df_reversed.iloc[::-1]

        # Round for aesthetics
        df_final['V-Score'] = df_final['V-Score'].round(2)

        return df_final

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


    def download_thumbnail(self, video_id, url):
        try:
            if not url:
                url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(os.path.join(self.thumbnails_folder, f"{video_id}.jpg"), "wb") as f:
                    f.write(response.content)
        except:
            pass

    def process_channel(self, channel_id):
        """
        Main processing function for a single channel.
        Uses new logarithmic V-Score calculation.
        """
        # 1. Fetching the metadata
        videos = self.get_long_form_data(channel_id)
        if not videos:
            print(f"⚠️ No videos found or error for channel: {channel_id}")
            return

        print(f"✅ Found {len(videos)} videos. Processing...")

        # 2. Collect video data into a list
        video_data = []
        for video in tqdm(videos, desc=f"Processing {channel_id[:10]}..."):
            vid_id = video.get('id')
            title = video.get('title')
            current_views = video.get('view_count')

            # Fetching thumbnails
            thumbnails = video.get('thumbnails', [])
            thumb_url = thumbnails[-1].get('url') if thumbnails else None

            if not vid_id or current_views is None:
                continue

            # Downloading thumbnails
            self.download_thumbnail(vid_id, thumb_url)

            video_data.append({
                'Video ID': vid_id,
                'Title': title,
                'Current Views': current_views
            })

        if not video_data:
            print(f"⚠️ No valid video data for channel: {channel_id}")
            return

        # 3. Create DataFrame and calculate V-Score
        df = pd.DataFrame(video_data)
        df_processed = self.calculate_log_v_score(df)

        # 4. Save to CSV (keeping only relevant columns)
        file_path = os.path.join(self.OUTPUT_FOLDER, f'{channel_id}.csv')
        output_columns = ['Video ID', 'Title', 'Current Views', 'V-Score']
        df_processed[output_columns].to_csv(file_path, index=False)

        print(f"🏁 Finished! Data saved to {file_path}")
