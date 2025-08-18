"""
Media Search Module for finding public domain content.
"""
import os
import hashlib
import aiohttp
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import quote
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.logger import log
from src.config.settings import settings
from src.models.database import DatabaseManager, Media, Keyword, Project


class MediaSearcher:
    """Search and download public domain media from various sources."""
    
    OPENVERSE_API_URL = "https://api.openverse.org/v1"
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.session = requests.Session()
        self._setup_auth()
    
    def _setup_auth(self):
        """Set up authentication for APIs."""
        if settings.openverse_client_id and settings.openverse_client_secret:
            # Get access token for OpenVerse
            self._get_openverse_token()
    
    def _get_openverse_token(self):
        """Get access token for OpenVerse API."""
        try:
            response = self.session.post(
                f"{self.OPENVERSE_API_URL}/auth_tokens/token/",
                data={
                    "client_id": settings.openverse_client_id,
                    "client_secret": settings.openverse_client_secret,
                    "grant_type": "client_credentials"
                }
            )
            if response.status_code == 200:
                token = response.json().get("access_token")
                self.session.headers.update({"Authorization": f"Bearer {token}"})
                log.info("OpenVerse authentication successful")
            else:
                log.warning(f"OpenVerse auth failed: {response.status_code}")
        except Exception as e:
            log.error(f"Error getting OpenVerse token: {str(e)}")
    
    async def search_media(
        self, 
        keywords: List[str], 
        media_type: str = "all",
        project_name: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for public domain media based on keywords.
        
        Args:
            keywords: List of keywords to search
            media_type: Type of media (image, video, audio, all)
            project_name: Name of the project to associate media with
            limit: Maximum number of results
            
        Returns:
            List of media items with metadata
        """
        # Check cache first
        cached_media = await self._check_cache(keywords, media_type)
        if cached_media:
            log.info(f"Found {len(cached_media)} cached media items")
            if project_name:
                await self._associate_with_project(cached_media, project_name)
            return cached_media
        
        # Search OpenVerse
        results = []
        
        if media_type in ["image", "all"]:
            image_results = await self._search_openverse_images(keywords, limit)
            results.extend(image_results)
        
        if media_type in ["audio", "all"]:
            audio_results = await self._search_openverse_audio(keywords, limit)
            results.extend(audio_results)
        
        # Note: OpenVerse doesn't have video search yet, so we'll need alternative sources
        if media_type in ["video", "all"]:
            # Could add Pond5 or other sources here
            pass
        
        # Download and cache the media
        downloaded_media = await self._download_and_cache(results, keywords)
        
        # Associate with project if specified
        if project_name and downloaded_media:
            await self._associate_with_project(downloaded_media, project_name)
        
        return downloaded_media
    
    async def _check_cache(
        self, 
        keywords: List[str], 
        media_type: str
    ) -> List[Dict[str, Any]]:
        """Check if we have cached media for these keywords."""
        with self.db_manager as session:
            # Get keywords from database
            keyword_objects = []
            for keyword in keywords:
                kw = session.query(Keyword).filter_by(keyword=keyword.lower()).first()
                if kw:
                    keyword_objects.append(kw)
            
            if not keyword_objects:
                return []
            
            # Find media associated with ALL keywords
            media_query = session.query(Media)
            for kw in keyword_objects:
                media_query = media_query.filter(Media.keywords.contains(kw))
            
            if media_type != "all":
                media_query = media_query.filter_by(media_type=media_type)
            
            cached_media = media_query.all()
            
            return [
                {
                    "id": str(media.id),
                    "url": media.source_url,
                    "file_path": media.file_path,
                    "type": media.media_type,
                    "metadata": media.file_metadata,
                    "cached": True
                }
                for media in cached_media
            ]
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _search_openverse_images(
        self, 
        keywords: List[str], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search OpenVerse for images."""
        query = quote(" ".join(keywords))
        url = f"{self.OPENVERSE_API_URL}/images/?q={query}&license=cc0&page_size={limit}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("results", []):
                results.append({
                    "id": item.get("id"),
                    "url": item.get("url"),
                    "thumbnail": item.get("thumbnail"),
                    "title": item.get("title"),
                    "creator": item.get("creator"),
                    "type": "image",
                    "width": item.get("width"),
                    "height": item.get("height"),
                    "license": item.get("license"),
                    "source": "openverse"
                })
            
            return results
            
        except Exception as e:
            log.error(f"Error searching OpenVerse images: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _search_openverse_audio(
        self, 
        keywords: List[str], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search OpenVerse for audio."""
        query = quote(" ".join(keywords))
        url = f"{self.OPENVERSE_API_URL}/audio/?q={query}&license=cc0&page_size={limit}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("results", []):
                results.append({
                    "id": item.get("id"),
                    "url": item.get("url"),
                    "title": item.get("title"),
                    "creator": item.get("creator"),
                    "type": "audio",
                    "duration": item.get("duration"),
                    "license": item.get("license"),
                    "source": "openverse"
                })
            
            return results
            
        except Exception as e:
            log.error(f"Error searching OpenVerse audio: {str(e)}")
            return []
    
    async def _download_and_cache(
        self, 
        media_items: List[Dict[str, Any]], 
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """Download media files and cache them in the database."""
        downloaded = []
        
        # Create download tasks
        async with aiohttp.ClientSession() as session:
            tasks = []
            for item in media_items:
                task = self._download_media_item(session, item, keywords)
                tasks.append(task)
            
            # Download in parallel with limit
            semaphore = asyncio.Semaphore(5)  # Limit concurrent downloads
            
            async def bounded_download(task):
                async with semaphore:
                    return await task
            
            results = await asyncio.gather(
                *[bounded_download(task) for task in tasks],
                return_exceptions=True
            )
            
            for result in results:
                if isinstance(result, dict) and result.get("success"):
                    downloaded.append(result)
        
        return downloaded
    
    async def _download_media_item(
        self, 
        session: aiohttp.ClientSession, 
        item: Dict[str, Any], 
        keywords: List[str]
    ) -> Dict[str, Any]:
        """Download a single media item."""
        try:
            # Generate file hash from URL
            url_hash = hashlib.md5(item["url"].encode()).hexdigest()
            
            # Check if already downloaded
            with self.db_manager as db_session:
                existing = db_session.query(Media).filter_by(source_url=item["url"]).first()
                if existing:
                    item["file_path"] = existing.file_path
                    item["cached"] = True
                    item["success"] = True
                    return item
            
            # Determine file extension
            ext = self._get_file_extension(item["url"], item["type"])
            filename = f"{url_hash}{ext}"
            file_path = settings.media_storage_path / item["type"] / filename
            
            # Create directory
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            async with session.get(item["url"]) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Save file
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    
                    # Calculate file hash
                    file_hash = hashlib.sha256(content).hexdigest()
                    
                    # Store in database
                    with self.db_manager as db_session:
                        # Create or get keywords
                        keyword_objects = []
                        for kw in keywords:
                            keyword = db_session.query(Keyword).filter_by(
                                keyword=kw.lower()
                            ).first()
                            if not keyword:
                                keyword = Keyword(keyword=kw.lower())
                                db_session.add(keyword)
                            keyword_objects.append(keyword)
                        
                        # Create media entry
                        media = Media(
                            file_hash=file_hash,
                            file_path=str(file_path),
                            media_type=item["type"],
                            source_url=item["url"],
                            file_size=len(content),
                            file_metadata=item
                        )
                        
                        # Associate keywords
                        media.keywords.extend(keyword_objects)
                        
                        db_session.add(media)
                        db_session.commit()
                        
                        item["id"] = str(media.id)
                    
                    item["file_path"] = str(file_path)
                    item["success"] = True
                    item["cached"] = False
                    
                    log.info(f"Downloaded {item['type']}: {item.get('title', 'Untitled')}")
                    
                else:
                    log.warning(f"Failed to download {item['url']}: {response.status}")
                    item["success"] = False
            
            return item
            
        except Exception as e:
            log.error(f"Error downloading media item: {str(e)}")
            item["success"] = False
            item["error"] = str(e)
            return item
    
    def _get_file_extension(self, url: str, media_type: str) -> str:
        """Determine file extension from URL or media type."""
        # Try to get from URL
        path = url.split('?')[0]
        ext = os.path.splitext(path)[1]
        
        if ext:
            return ext
        
        # Default extensions by type
        defaults = {
            "image": ".jpg",
            "audio": ".mp3",
            "video": ".mp4"
        }
        
        return defaults.get(media_type, ".bin")
    
    async def _associate_with_project(
        self, 
        media_items: List[Dict[str, Any]], 
        project_name: str
    ):
        """Associate downloaded media with a project."""
        with self.db_manager as session:
            project = session.query(Project).filter_by(name=project_name).first()
            
            if not project:
                log.warning(f"Project {project_name} not found")
                return
            
            for item in media_items:
                if item.get("id"):
                    media = session.query(Media).filter_by(id=item["id"]).first()
                    if media and media not in project.media:
                        project.media.append(media)
            
            session.commit()
            log.info(f"Associated {len(media_items)} media items with project {project_name}")
    
    async def search_video_alternatives(
        self, 
        keywords: List[str], 
        duration: int = 45
    ) -> List[Dict[str, Any]]:
        """
        Search for video alternatives or components to create a video.
        Since OpenVerse doesn't have videos, we'll search for:
        1. Image sequences that can be turned into a slideshow
        2. Audio tracks for background music
        """
        results = {
            "images": [],
            "audio": []
        }
        
        # Search for relevant images (get more for slideshow)
        image_results = await self._search_openverse_images(keywords, limit=20)
        results["images"] = image_results
        
        # Search for background music
        music_keywords = keywords + ["music", "instrumental", "background"]
        audio_results = await self._search_openverse_audio(music_keywords, limit=5)
        
        # Filter audio by duration if possible
        suitable_audio = []
        for audio in audio_results:
            if audio.get("duration"):
                # Prefer audio close to our target duration
                if audio["duration"] >= duration:
                    suitable_audio.append(audio)
            else:
                # Include if duration unknown
                suitable_audio.append(audio)
        
        results["audio"] = suitable_audio
        
        return results
