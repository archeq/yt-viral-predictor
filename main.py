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

    NICHES = ["programming", "software engineering", "vibe coding"]

    # Downloader initialization
    downloader = DataDownloader(
        api_key=API_KEY,
        niches=NICHES,
        OUTPUT_FOLDER="data/raw/"
    )

    discovered_channels = downloader.channels_by_niche(NICHES, max_searches=10)

    if not discovered_channels:
        print("❌ ")
        return

    chosen_channels = []  # If you want to include specific channels, add their IDs here

    print(f"\n📺 Final Collection: {len(discovered_channels)}  Unique Channels")
    print("-" * 30)

    # 2. Processing each found channel
    for channel_id, channel_title in discovered_channels.items():
        print(f"\n--- Processing channel: {channel_title} ({channel_id}) ---")

        try:
            # neighbor_count=10 is default value for Elastic Average
            downloader.process_channel(channel_id, neighbor_count=10)
        except Exception as e:
            print(f"❌ Error occurred while processing {channel_title}: {e}")

    print("\n✅ Job finished successfully!")


if __name__ == "__main__":
    main()
