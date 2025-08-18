# Domain Sticks 🎬

An automated pipeline for creating short-form vertical videos from public domain content. The system takes any URL, extracts content, generates scripts using AI, finds relevant public domain media, creates videos, and can automatically upload to YouTube.

## Features ✨

- **Intelligent Web Scraping**: Extracts metadata, content, and media from any URL
- **AI Script Generation**: Uses DeepSeek v3 to create engaging 45-second scripts
- **Media Search & Caching**: Finds and caches public domain media from OpenVerse
- **Video Processing**: Creates vertical videos (9:16) with text overlays and effects
- **YouTube Integration**: Automatic metadata generation and upload capabilities
- **Deduplication System**: Prevents reprocessing of same URLs
- **Workflow Management**: Organized pipeline with distinct processing stages

## Architecture 🏗️

The system follows a modular architecture with distinct stages:

```
URL → Scraping → Script Generation → Media Search → Video Processing → Upload
         ↓              ↓                 ↓              ↓              ↓
    [Database]    [DeepSeek v3]    [OpenVerse]      [FFmpeg]      [YouTube]
```

### Workflow Stages

1. **Ingestion**: URL scraping and metadata extraction
2. **Analysis**: Content analysis and categorization
3. **Script Generation**: AI-powered script creation
4. **Media Search**: Finding relevant public domain media
5. **Video Processing**: Creating the vertical video
6. **Metadata**: Generating YouTube metadata
7. **Approval**: Manual review stage
8. **Upload**: Publishing to YouTube
9. **Published**: Completed projects

## Installation 🚀

### Prerequisites

- Python 3.9+
- PostgreSQL
- FFmpeg
- YouTube API credentials
- DeepSeek API key
- OpenVerse API credentials (optional)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/domain-sticks.git
cd domain-sticks
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Set up the database:
```bash
# Create PostgreSQL database
createdb domain_sticks

# Update DATABASE_URL in .env
```

6. Initialize the system:
```bash
python -m src.driver status
# This will create necessary tables and directories
```

## Configuration 🔧

### Environment Variables

Key configuration options in `.env`:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/domain_sticks

# DeepSeek API
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com/v1

# OpenVerse API (optional for authenticated access)
OPENVERSE_CLIENT_ID=your_client_id
OPENVERSE_CLIENT_SECRET=your_client_secret

# YouTube API
YOUTUBE_CLIENT_SECRETS_FILE=client_secrets.json
YOUTUBE_CREDENTIALS_FILE=youtube_credentials.json

# Video Settings
VIDEO_WIDTH=1080
VIDEO_HEIGHT=1920
VIDEO_FPS=30
VIDEO_BITRATE=8000k
MAX_VIDEO_DURATION=45
```

### YouTube Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable YouTube Data API v3
4. Create OAuth 2.0 credentials
5. Download credentials as `client_secrets.json`
6. Place in project root directory

## Usage 📖

### Command Line Interface

Process a single URL:
```bash
python -m src.driver process "https://example.com/article"

# With options
python -m src.driver process "https://example.com/article" \
    --name "my_project" \
    --focus creator \
    --auto-upload
```

Batch processing:
```bash
# Create a file with URLs (one per line)
echo "https://example1.com\nhttps://example2.com" > urls.txt

python -m src.driver batch urls.txt --auto-upload
```

Process workflow stage:
```bash
# Process all projects in approval stage
python -m src.driver process-stage approval
```

Check project status:
```bash
python -m src.driver status "project_name"
```

### Python API

```python
from src.driver import DomainSticksDriver

# Initialize driver
driver = DomainSticksDriver()

# Process a URL
result = await driver.process_url(
    url="https://example.com/article",
    project_name="my_video",
    auto_upload=True,
    focus="creator"  # or "subject", "work", "auto"
)

# Check result
if result["status"] == "success":
    print(f"Video created: {result['video_path']}")
    print(f"YouTube URL: {result['youtube_url']}")
```

## Project Structure 📁

```
domain-sticks/
├── src/
│   ├── config/          # Configuration management
│   ├── models/          # Database models
│   ├── modules/         # Core processing modules
│   │   ├── scraper.py
│   │   ├── script_generator.py
│   │   ├── media_search.py
│   │   ├── video_processor.py
│   │   └── youtube_uploader.py
│   ├── utils/           # Utility functions
│   └── driver.py        # Main orchestration
├── workflow/            # Processing stages
│   ├── 01_ingestion/
│   ├── 02_analysis/
│   ├── 03_script_generation/
│   ├── 04_media_search/
│   ├── 05_video_processing/
│   ├── 06_metadata/
│   ├── 07_approval/
│   ├── 08_upload/
│   └── 09_published/
├── media_storage/       # Cached media files
├── logs/               # Application logs
└── tests/              # Test suite
```

## Features in Detail 🔍

### Deduplication System

- URLs are hashed and stored to prevent reprocessing
- Media files are cached with keyword associations
- Cached media is reused when similar keywords appear

### Script Generation

The AI script generator:
- Analyzes content to determine focus (creator, subject, work)
- Creates engaging hooks for the first 3 seconds
- Generates visual cues for media selection
- Ensures scripts fit within 45-second duration
- Can split long content into multiple segments

### Video Processing

- Supports multiple input formats (images, videos, audio)
- Creates slideshows from images with transitions
- Adds text overlays with proper timing
- Generates gradient backgrounds for text-only videos
- Includes fade effects and professional styling

### YouTube Integration

Automatic generation of:
- SEO-optimized titles with emojis
- Comprehensive descriptions with timestamps
- Relevant tags and hashtags
- Appropriate category selection
- Thumbnail extraction

## Troubleshooting 🔧

### Common Issues

1. **FFmpeg not found**: Ensure FFmpeg is installed and in PATH
2. **Database connection errors**: Check PostgreSQL is running and credentials are correct
3. **API rate limits**: Add delays between batch processing
4. **Memory issues with large videos**: Adjust chunk sizes in video processing

### Logs

Check logs for detailed error information:
```bash
tail -f logs/domain_sticks.log
```

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License 📄

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments 🙏

- DeepSeek for AI capabilities
- OpenVerse for public domain media
- FFmpeg for video processing
- All contributors and open source projects used