#!/usr/bin/env python3
"""
Example script demonstrating Domain Sticks usage.
"""

import asyncio
from src.driver import DomainSticksDriver


async def main():
    """Example usage of Domain Sticks."""
    
    # Initialize the driver
    print("🚀 Initializing Domain Sticks...")
    driver = DomainSticksDriver()
    
    # Example 1: Process a single URL
    print("\n📍 Example 1: Processing a single URL")
    result = await driver.process_url(
        url="https://en.wikipedia.org/wiki/Public_domain",
        project_name="public_domain_intro",
        auto_upload=False,  # Set to True to auto-upload to YouTube
        focus="subject"     # Focus on the subject matter
    )
    
    if result["status"] == "success":
        print(f"✅ Video created successfully!")
        print(f"📹 Video path: {result['video_path']}")
        print(f"📝 Script preview: {result['script'][:100]}...")
        print(f"🎬 YouTube metadata generated:")
        print(f"   Title: {result['youtube_metadata']['title']}")
        print(f"   Tags: {', '.join(result['youtube_metadata']['tags'][:5])}...")
    
    # Example 2: Process multiple URLs
    print("\n📍 Example 2: Batch processing")
    urls = [
        "https://en.wikipedia.org/wiki/Leonardo_da_Vinci",
        "https://en.wikipedia.org/wiki/Vincent_van_Gogh"
    ]
    
    # Uncomment to run batch processing
    # results = await driver.process_batch(urls, auto_upload=False)
    # print(f"✅ Processed {len(results)} URLs")
    
    # Example 3: Check project status
    print("\n📍 Example 3: Checking project status")
    with driver.db_manager as session:
        from src.models.database import Project
        projects = session.query(Project).limit(5).all()
        
        if projects:
            print("Recent projects:")
            for project in projects:
                print(f"  - {project.name}: {project.current_stage} ({project.status})")
        else:
            print("No projects found yet.")
    
    # Example 4: Search for cached media
    print("\n📍 Example 4: Searching for media")
    media_items = await driver.media_searcher.search_media(
        keywords=["art", "painting", "public domain"],
        media_type="image",
        limit=5
    )
    
    print(f"Found {len(media_items)} media items")
    for item in media_items[:3]:
        print(f"  - {item.get('title', 'Untitled')} ({item['type']})")
    
    print("\n✨ Example completed!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
