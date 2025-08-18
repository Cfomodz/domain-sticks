#!/usr/bin/env python3
"""
Example script demonstrating audio-to-video processing with shortform content generation.

This script shows how to use the new AudioVideoProcessor to:
1. Process audio files from the ingestion directory
2. Generate full videos with transcription and relevant images
3. Create shortform clips based on AI analysis
4. Manage the generated content through the database

Usage:
    python audio_example.py
"""
import asyncio
from pathlib import Path
from src.config.settings import settings
from src.models.database import DatabaseManager
from src.modules.audio_video_processor import AudioVideoProcessor, AudioBatchProcessor
from src.utils.logger import log


async def demo_single_audio_processing():
    """Demonstrate processing a single audio file."""
    print("🎵 Audio-to-Video Processing Demo")
    print("=" * 50)
    
    # Initialize database and processor
    db_manager = DatabaseManager(settings.database_url)
    db_manager.create_tables()
    
    try:
        processor = AudioVideoProcessor(db_manager)
    except ImportError as e:
        print(f"❌ Error: Audio processing dependencies not available: {e}")
        print("Please install: pip install openai-whisper")
        return
    
    # Look for audio files in the ingestion directory
    ingestion_path = settings.workflow_paths["ingestion"]
    audio_extensions = ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg']
    
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(ingestion_path.glob(f"*{ext}"))
    
    if not audio_files:
        print(f"🔍 No audio files found in: {ingestion_path}")
        print("Please add audio files to the ingestion directory and try again.")
        return
    
    # Process the first audio file found
    audio_file = audio_files[0]
    print(f"📁 Processing: {audio_file.name}")
    
    try:
        # Process the audio file with shortform generation
        result = await processor.process_audio_file(
            audio_file,
            project_name=f"demo_{audio_file.stem}",
            generate_shorts=True
        )
        
        # Display results
        if result["status"] == "success":
            print(f"\n✅ Successfully processed: {result['project_name']}")
            print(f"📹 Main video: {result['main_video']['video_path']}")
            print(f"⏱️ Duration: {result['main_video']['duration']:.1f} seconds")
            print(f"🔍 Keywords extracted: {', '.join(result['keywords'][:10])}")
            print(f"🎯 Media items used: {result['media_items_count']}")
            
            # Show transcription sample
            transcript = result['transcription']['text']
            print(f"\n📝 Transcription sample (first 200 chars):")
            print(f"   {transcript[:200]}{'...' if len(transcript) > 200 else ''}")
            
            # Show shortform clips
            if result.get("shorts"):
                print(f"\n📱 Generated {result['shorts_count']} shortform clips:")
                for i, short in enumerate(result["shorts"], 1):
                    metadata = short["metadata"]
                    print(f"   {i}. {metadata['title']}")
                    print(f"      ⏱️ {metadata['duration']:.1f}s ({metadata['start_time']:.1f}s - {metadata['end_time']:.1f}s)")
                    print(f"      🎯 Hook: {metadata['hook']}")
                    print(f"      📹 File: {short['clip_path']}")
                    print()
        else:
            print(f"❌ Processing failed: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error processing audio: {str(e)}")


async def demo_batch_processing():
    """Demonstrate batch processing of all audio files."""
    print("\n🎵 Batch Audio Processing Demo")
    print("=" * 50)
    
    # Initialize database and batch processor
    db_manager = DatabaseManager(settings.database_url)
    db_manager.create_tables()
    
    try:
        batch_processor = AudioBatchProcessor(db_manager)
    except ImportError as e:
        print(f"❌ Error: Audio processing dependencies not available: {e}")
        return
    
    try:
        # Process all audio files in the ingestion directory
        results = await batch_processor.process_ingestion_directory(
            generate_shorts=True
        )
        
        # Display batch results
        success_count = sum(1 for r in results if r.get("status") == "success")
        total_shorts = sum(r.get("shorts_count", 0) for r in results if r.get("status") == "success")
        
        print(f"\n📊 Batch processing complete:")
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {len(results) - success_count}")
        print(f"📱 Total shorts generated: {total_shorts}")
        
        # Show details for each processed file
        for result in results:
            if result.get("status") == "success":
                print(f"\n🎵 {result['project_name']}:")
                print(f"   📹 Video: {Path(result['main_video']['video_path']).name}")
                print(f"   ⏱️ Duration: {result['main_video']['duration']:.1f}s")
                if result.get("shorts"):
                    print(f"   📱 Shorts: {result['shorts_count']}")
            elif result.get("status") == "failed":
                print(f"\n❌ Failed: {result.get('file', 'Unknown')}")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"❌ Error in batch processing: {str(e)}")


async def demo_database_queries():
    """Demonstrate querying the database for audio projects and shorts."""
    print("\n🗄️ Database Query Demo")
    print("=" * 50)
    
    # Initialize database
    db_manager = DatabaseManager(settings.database_url)
    
    try:
        with db_manager as session:
            from src.models.database import Project, AudioProject, ShortformClip
            
            # Query audio projects
            audio_projects = session.query(Project).join(AudioProject).all()
            
            if not audio_projects:
                print("📭 No audio projects found in database.")
                return
            
            print(f"📊 Found {len(audio_projects)} audio projects:")
            
            for project in audio_projects:
                print(f"\n🎵 {project.name}")
                print(f"   📍 Stage: {project.current_stage}")
                print(f"   ✅ Status: {project.status}")
                
                if project.audio_project:
                    ap = project.audio_project
                    print(f"   🎧 Audio: {Path(ap.audio_file_path).name}")
                    print(f"   🔤 Language: {ap.transcription_language}")
                    print(f"   📱 Shorts: {ap.shortform_clips_count}")
                
                # Show shortform clips
                if project.shortform_clips:
                    print(f"   📱 Shortform clips:")
                    for clip in project.shortform_clips[:3]:  # Show first 3
                        print(f"      • {clip.title} ({clip.duration}s)")
                    if len(project.shortform_clips) > 3:
                        print(f"      ... and {len(project.shortform_clips) - 3} more")
                        
    except Exception as e:
        print(f"❌ Error querying database: {str(e)}")


def demo_cli_usage():
    """Show CLI usage examples."""
    print("\n🖥️ CLI Usage Examples")
    print("=" * 50)
    
    print("Process a single audio file:")
    print("  python -m src.driver process-audio path/to/audio.mp3")
    print("  python -m src.driver process-audio path/to/audio.mp3 --name my_project")
    print("  python -m src.driver process-audio path/to/audio.mp3 --no-shorts")
    
    print("\nProcess all audio files in ingestion directory:")
    print("  python -m src.driver process-all-audio")
    print("  python -m src.driver process-all-audio --no-shorts")
    
    print("\nCheck project status:")
    print("  python -m src.driver status audio_my_project_20240101_120000")
    
    print("\nShow shortform clips for a project:")
    print("  python -m src.driver show-shorts audio_my_project_20240101_120000")


async def main():
    """Run all demos."""
    print("🚀 Domain Sticks Audio-to-Video Demo")
    print("=" * 50)
    
    # Run demos
    await demo_single_audio_processing()
    await demo_batch_processing()
    await demo_database_queries()
    demo_cli_usage()
    
    print("\n✨ Demo complete!")
    print("\nNext steps:")
    print("1. Add audio files to the ingestion directory")
    print("2. Install dependencies: pip install openai-whisper")
    print("3. Configure your DeepSeek API key in .env")
    print("4. Run: python -m src.driver process-all-audio")


if __name__ == "__main__":
    asyncio.run(main())