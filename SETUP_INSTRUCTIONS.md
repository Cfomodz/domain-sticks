# Domain Sticks Setup Instructions

## Current Status ✅
- ✅ Python 3.12 virtual environment created
- ✅ All Python dependencies installed
- ✅ SQLite database working
- ✅ All core modules implemented and tested
- ✅ PostgreSQL installed
- ⏳ PostgreSQL setup needed

## Complete PostgreSQL Setup

### Option 1: Run the Setup Script (Recommended)
```bash
sudo ./setup_postgresql.sh
```

### Option 2: Manual Setup
If you prefer to run commands manually:

```bash
# 1. Initialize PostgreSQL
sudo -u postgres initdb --locale=C.UTF-8 --encoding=UTF8 -D '/var/lib/postgres/data'

# 2. Start PostgreSQL
sudo systemctl enable postgresql
sudo systemctl start postgresql

# 3. Create database and user
sudo -u postgres psql -c "CREATE DATABASE domain_sticks;"
sudo -u postgres psql -c "CREATE USER domain_user WITH PASSWORD 'domain_sticks_2024';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE domain_sticks TO domain_user;"
```

## Test Database Connection
After PostgreSQL setup:

```bash
source .venv/bin/activate
python -c "from src.models.database import DatabaseManager; db = DatabaseManager(); db.create_tables(); print('✅ PostgreSQL working!')"
```

## Next Steps

### 1. API Keys Setup
Edit `.env` file and add your API keys:

```bash
# For AI script generation (required)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# For media search (optional)
OPENVERSE_CLIENT_ID=your_openverse_client_id
OPENVERSE_CLIENT_SECRET=your_openverse_client_secret

# For YouTube upload (optional)
# Download credentials from Google Cloud Console
YOUTUBE_CLIENT_SECRETS_FILE=client_secrets.json
```

### 2. Install FFmpeg (for video processing)
```bash
sudo pacman -S ffmpeg
```

### 3. Test Complete System
```bash
source .venv/bin/activate
python test_setup.py
```

### 4. Process Your First URL
```bash
source .venv/bin/activate
python -m src.driver process "https://example.com/article"
```

## Troubleshooting

### PostgreSQL Connection Issues
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Restart if needed
sudo systemctl restart postgresql
```

### SQLite Fallback
If PostgreSQL has issues, you can always fall back to SQLite by changing `.env`:
```bash
DATABASE_URL=sqlite:///./domain_sticks.db
```

### Video Processing Issues
Video processing requires moviepy and ffmpeg. If not installed, the system will work but won't create actual videos.

## Project Structure
```
domain-sticks/
├── src/
│   ├── modules/          # Core processing modules
│   ├── models/           # Database models
│   ├── config/           # Configuration management
│   └── utils/            # Utilities (logging, etc.)
├── workflow/             # Processing stages
├── media_storage/        # Downloaded media files
├── logs/                 # Application logs
├── .env                  # Configuration file
└── test_setup.py         # Setup verification script
```

## Support
The system is modular and fault-tolerant. Each component can work independently, so you can test and develop incrementally.
