#!/usr/bin/env python3
"""
Test script to verify Domain Sticks setup.
"""

import asyncio
from src.config.settings import settings
from src.models.database import DatabaseManager
from src.utils.logger import log


async def test_setup():
    """Test basic functionality of Domain Sticks."""
    
    print("🚀 Testing Domain Sticks Setup...")
    
    # Test 1: Settings
    print("\n📍 Test 1: Settings configuration")
    try:
        settings.ensure_directories()
        print(f"✅ Settings loaded: Database URL = {settings.database_url}")
        print(f"✅ Directories created successfully")
    except Exception as e:
        print(f"❌ Settings error: {e}")
        return
    
    # Test 2: Database
    print("\n📍 Test 2: Database setup")
    try:
        db_manager = DatabaseManager(settings.database_url)
        db_manager.create_tables()
        print("✅ Database tables created successfully")
        
        # Test database connection
        with db_manager as session:
            print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database error: {e}")
        return
    
    # Test 3: Logging
    print("\n📍 Test 3: Logging system")
    try:
        log.info("Test log message")
        print("✅ Logging system working")
    except Exception as e:
        print(f"❌ Logging error: {e}")
        return
    
    # Test 4: Web scraper (basic import)
    print("\n📍 Test 4: Module imports")
    try:
        from src.modules.scraper import WebScraper
        print("✅ Web scraper module imported")
        
        from src.modules.script_generator import ScriptGenerator
        print("✅ Script generator module imported")
        
        from src.modules.media_search import MediaSearcher
        print("✅ Media search module imported")
        
        from src.modules.youtube_uploader import YouTubeUploader
        print("✅ YouTube uploader module imported")
        
        try:
            from src.modules.video_processor import VideoProcessor
            print("✅ Video processor module imported")
        except ImportError as e:
            print(f"⚠️  Video processor not available: {e}")
            print("   This is normal if moviepy/ffmpeg are not installed")
        
    except Exception as e:
        print(f"❌ Module import error: {e}")
        return
    
    print("\n✨ All basic tests passed! Domain Sticks is ready to use.")
    print("\n📋 Next steps:")
    print("1. Set up your API keys in the .env file:")
    print("   - DEEPSEEK_API_KEY for script generation")
    print("   - OPENVERSE_CLIENT_ID/SECRET for media search (optional)")
    print("   - YouTube API credentials for video upload")
    print("2. Install ffmpeg for video processing:")
    print("   - Manjaro: sudo pacman -S ffmpeg")
    print("   - Ubuntu: sudo apt install ffmpeg")
    print("3. For PostgreSQL (optional, currently using SQLite):")
    print("   - Install: sudo pacman -S postgresql")
    print("   - Setup: sudo -u postgres createdb domain_sticks")
    print("   - Update DATABASE_URL in .env")


if __name__ == "__main__":
    asyncio.run(test_setup())
