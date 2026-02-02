# V-Score Algorithm Documentation

## Overview

The V-Score (Virality Score) is a logarithmic metric that measures how well a video performs compared to the channel's historical baseline. It uses rolling statistics to create a fair comparison that accounts for channel growth over time.

## How It Works

### 1. Data Collection (`process_channel`)

For each channel, the system:
1. Fetches all video metadata using `yt-dlp`
2. Downloads thumbnails for each video
3. Collects: `Video ID`, `Title`, `Current Views`
4. Data is ordered from **newest to oldest** (Row 0 = newest video)

### 2. V-Score Calculation (`calculate_log_v_score`)

#### Step 1: Logarithmic Transformation
```python
log_views = log(Current Views + 1)
```
- Uses `log1p` (log(x+1)) to safely handle zeros
- Logarithms normalize the huge differences between view counts

#### Step 2: Rolling Baseline Statistics
- **Window Size**: Last 30 videos (configurable via `WINDOW_SIZE`)
- **Minimum Periods**: At least 5 videos required (configurable via `MIN_PERIODS`)
- Uses `.shift(1)` to exclude the current video from its own baseline

Calculated metrics:
- `baseline_median` - median of log views from previous videos
- `baseline_std` - standard deviation of log views from previous videos

#### Step 3: V-Score Formula
```
V-Score = (log_views - baseline_median) / baseline_std
```

This is essentially a **z-score** in log space, showing how many standard deviations a video is from the channel's typical performance.

### 3. Interpretation

| V-Score | Meaning |
|---------|---------|
| > 2.0 | 🔥 Viral - Exceptional performance |
| 1.0 to 2.0 | 📈 Above average |
| -1.0 to 1.0 | 📊 Normal performance |
| -1.0 to -2.0 | 📉 Below average |
| < -2.0 | ❌ Significantly underperformed |

## Output Format

### CSV Structure
Data is saved to `data/raw/{channel_id}.csv` with columns:

| Column | Description |
|--------|-------------|
| `Video ID` | YouTube video ID |
| `Title` | Video title |
| `Current Views` | Total view count |
| `V-Score` | Calculated virality score (rounded to 2 decimals) |

### Example Output
```csv
Video ID,Title,Current Views,V-Score
dQw4w9WgXcQ,Amazing Video Title,1500000,2.34
abc123xyz,Another Great Video,500000,0.87
def456uvw,Regular Content,250000,-0.12
```

### Thumbnails
Downloaded to `data/raw/thumbnails/{video_id}.jpg`

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WINDOW_SIZE` | 30 | Number of previous videos to consider for baseline |
| `MIN_PERIODS` | 5 | Minimum videos needed to calculate V-Score |
| `OUTPUT_FOLDER` | `data/raw/` | Where CSV files and thumbnails are saved |

## Edge Cases

- **New channels** (< 5 videos): V-Score will be `NaN` until enough history exists
- **Zero views**: Handled safely by `log1p`
- **Zero standard deviation**: Replaced with 1.0 to avoid division by zero

## Why Logarithmic?

1. **Handles scale**: A video with 10M views vs 100K views is treated proportionally
2. **Fairer comparison**: Small channels and large channels can be compared
3. **Statistical validity**: View counts follow a log-normal distribution
