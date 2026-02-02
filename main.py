import torch
import sys
from download import DataDownloader


def system_check():
    print(f"--- Python Version: {sys.version.split()[0]} ---")
    print(f"--- PyTorch Version: {torch.__version__} ---")

    if torch.cuda.is_available():
        print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available (Running on CPU)")


def main():
    # --- consts ---
    with open("api.txt", "r") as f:
        API_KEY = f.read().strip()

    NICHES = ["programming", "software engineering", "vibe coding", "counter strike", "study"]

    # Downloader initialization
    downloader = DataDownloader(
        api_key=API_KEY,
        niches=NICHES,
        OUTPUT_FOLDER="data/raw/"
    )

    discovered_channels = downloader.channels_by_niche(NICHES, max_searches=3)

    if not discovered_channels:
        print("❌ ")
        return

    chosen_channels = [
        'UCcgNiiErDGHICbNYtuyxW4A',
        'UCV0qA-eDDICsRR9rPcnG7tw',
        'UCPhyA52nHlW3L6r-7sDfcyg',
        'UC7zZ2-Q_oxbUaoMVL0z51wg',
        'UCG4pyCrLYfDsSaRUrBH1MKA',
        'UCvYUyKg7wDj760PippmWhig',
        'UCzSc8bhRKEKe7xEGp-5LWAg',
        'UCZ59iKBmGRfQlnl73sOX0Lw',
        'UCsguRUxh-0DiryIdPPWFeCQ',
        'UCDlqbnftd2Z_Ysh8itMvhwg',
        'UCMUsC4hvDRNefChiOJflyRg'
                       ]
    # If you want to include specific channels, add their IDs here

    print(f"\n📺 Final Collection: {len(discovered_channels)}  Unique Channels")
    print("-" * 30)

    # 2. Processing each found channel
    for channel_id, channel_title in discovered_channels.items():
        print(f"\n--- Processing channel: {channel_title} ({channel_id}) ---")

        try:
            downloader.process_channel(channel_id)
        except Exception as e:
            print(f"❌ Error occurred while processing {channel_title}: {e}")

    for channel_id in chosen_channels:
        print(f"\n--- Processing channel: ({channel_id}) ---")

        try:
            downloader.process_channel(channel_id)
        except Exception as e:
            print(f"❌ Error occurred while processing {channel_id}: {e}")

    print("\n✅ Job finished successfully!")


if __name__ == "__main__":
    main()
