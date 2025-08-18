"""
YouTube Upload Module for uploading videos with metadata.
"""
import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import httplib2
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from src.utils.logger import log
from src.config.settings import settings
from src.models.database import DatabaseManager, Project


class YouTubeUploader:
    """Upload videos to YouTube with proper metadata."""
    
    SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.youtube_service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with YouTube API."""
        creds = None
        
        # Token file stores the user's access and refresh tokens
        token_file = Path(settings.youtube_credentials_file)
        
        if token_file.exists():
            try:
                with open(token_file, 'rb') as token:
                    creds = pickle.load(token)
            except Exception as e:
                log.warning(f"Error loading credentials: {str(e)}")
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not Path(settings.youtube_client_secrets_file).exists():
                    log.error(f"Client secrets file not found: {settings.youtube_client_secrets_file}")
                    return
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    settings.youtube_client_secrets_file, 
                    self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        self.youtube_service = build('youtube', 'v3', credentials=creds)
        log.info("YouTube authentication successful")
    
    async def generate_metadata(
        self, 
        project_name: str
    ) -> Dict[str, Any]:
        """
        Generate YouTube metadata for the video.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dictionary containing YouTube metadata
        """
        with self.db_manager as session:
            project = session.query(Project).filter_by(name=project_name).first()
            if not project:
                raise ValueError(f"Project {project_name} not found")
            
            # Get script metadata
            script_metadata = project.script_metadata or {}
            
            # Generate title (max 100 characters)
            title = self._generate_title(project)
            
            # Generate description
            description = self._generate_description(project, script_metadata)
            
            # Generate tags
            tags = self._generate_tags(project, script_metadata)
            
            # Generate hashtags for description
            hashtags = self._generate_hashtags(tags)
            
            metadata = {
                "title": title,
                "description": description + "\n\n" + hashtags,
                "tags": tags,
                "category_id": self._determine_category(script_metadata),
                "privacy_status": "public",  # or "private", "unlisted"
                "made_for_kids": False,
                "default_language": "en",
                "default_audio_language": "en"
            }
            
            # Store metadata in project
            project.youtube_metadata = metadata
            session.commit()
            
            return metadata
    
    def _generate_title(self, project: Project) -> str:
        """Generate a compelling title for YouTube."""
        if project.title:
            title = project.title
        else:
            # Extract from script hook
            script_metadata = project.script_metadata or {}
            hook = script_metadata.get("hook", "")
            if hook:
                # Use first part of hook as title
                title = hook.split('.')[0].strip()
            else:
                title = f"Story of {project.name}"
        
        # Ensure title is within YouTube limits (100 characters)
        if len(title) > 97:
            title = title[:97] + "..."
        
        # Add emoji for engagement
        title = self._add_emoji(title)
        
        return title
    
    def _generate_description(
        self, 
        project: Project, 
        script_metadata: Dict[str, Any]
    ) -> str:
        """Generate YouTube description."""
        description_parts = []
        
        # Add main description
        if project.description:
            description_parts.append(project.description)
        else:
            description_parts.append(project.script[:200] + "...")
        
        description_parts.append("")  # Empty line
        
        # Add timestamps if video has segments
        if project.segments:
            description_parts.append("📍 TIMESTAMPS:")
            for segment in project.segments:
                timestamp = self._format_timestamp(segment.start_time)
                description_parts.append(f"{timestamp} - Part {segment.segment_number}")
            description_parts.append("")
        
        # Add credits
        description_parts.append("🎬 ABOUT THIS VIDEO:")
        description_parts.append(f"This video tells the story of {project.title or project.name}.")
        
        tone = script_metadata.get("tone", "educational")
        description_parts.append(f"Style: {tone.capitalize()}")
        
        description_parts.append("")
        description_parts.append("📚 SOURCES:")
        if project.source_url:
            description_parts.append(f"Original content: {project.source_url.url}")
        
        description_parts.append("")
        description_parts.append("🎵 MEDIA CREDITS:")
        description_parts.append("All media used in this video is from public domain sources.")
        description_parts.append("- Images/Audio: OpenVerse (openverse.org)")
        
        description_parts.append("")
        description_parts.append("📱 FOLLOW FOR MORE:")
        description_parts.append("Subscribe for more educational content!")
        
        description_parts.append("")
        description_parts.append("⚖️ COPYRIGHT:")
        description_parts.append("This video uses only public domain content.")
        description_parts.append("Created with AI assistance for educational purposes.")
        
        return "\n".join(description_parts)
    
    def _generate_tags(
        self, 
        project: Project, 
        script_metadata: Dict[str, Any]
    ) -> List[str]:
        """Generate relevant tags for YouTube."""
        tags = []
        
        # Add keywords from script
        keywords = script_metadata.get("keywords", [])
        tags.extend(keywords[:10])  # Limit keywords
        
        # Add tone-based tags
        tone = script_metadata.get("tone", "educational")
        tone_tags = {
            "educational": ["education", "learning", "facts", "knowledge"],
            "inspiring": ["inspiration", "motivation", "inspiring story"],
            "mysterious": ["mystery", "unknown facts", "secrets"],
            "historical": ["history", "historical facts", "past events"]
        }
        tags.extend(tone_tags.get(tone, []))
        
        # Add format tags
        tags.extend(["shorts", "short video", "vertical video", "tiktok style"])
        
        # Add content type tags
        if project.source_url:
            page_type = project.source_url.scraped_metadata.get("page_type", "") if project.source_url.scraped_metadata else ""
            if page_type:
                tags.append(page_type)
        
        # Remove duplicates and limit to 30 tags
        tags = list(set(tags))[:30]
        
        return tags
    
    def _generate_hashtags(self, tags: List[str]) -> str:
        """Generate hashtags for description."""
        # Select top tags for hashtags
        hashtag_candidates = [tag for tag in tags if len(tag.split()) == 1][:5]
        hashtags = [f"#{tag.replace(' ', '')}" for tag in hashtag_candidates]
        
        # Add trending hashtags
        hashtags.extend(["#shorts", "#educational", "#publicdomain"])
        
        return " ".join(hashtags[:8])  # Limit to 8 hashtags
    
    def _determine_category(self, script_metadata: Dict[str, Any]) -> str:
        """Determine YouTube category ID based on content."""
        # YouTube category IDs
        categories = {
            "education": "27",
            "entertainment": "24",
            "science": "28",
            "history": "27",
            "art": "1",
            "music": "10",
            "news": "25",
            "howto": "26",
            "people": "22"
        }
        
        # Determine based on tone and keywords
        tone = script_metadata.get("tone", "").lower()
        keywords = " ".join(script_metadata.get("keywords", [])).lower()
        
        if "education" in tone or "learn" in keywords:
            return categories["education"]
        elif "science" in keywords:
            return categories["science"]
        elif "history" in keywords or "historical" in tone:
            return categories["history"]
        elif "art" in keywords or "artist" in keywords:
            return categories["art"]
        else:
            return categories["entertainment"]
    
    def _add_emoji(self, title: str) -> str:
        """Add relevant emoji to title."""
        # Simple emoji mapping
        emoji_map = {
            "story": "📖",
            "history": "🏛️",
            "science": "🔬",
            "art": "🎨",
            "music": "🎵",
            "mystery": "🔍",
            "fact": "💡",
            "amazing": "✨",
            "secret": "🤫"
        }
        
        # Check if title already has emoji
        if any(ord(char) > 127 for char in title):
            return title
        
        # Add emoji based on keywords
        title_lower = title.lower()
        for keyword, emoji in emoji_map.items():
            if keyword in title_lower:
                return f"{emoji} {title}"
        
        # Default emoji
        return f"🎬 {title}"
    
    def _format_timestamp(self, seconds: int) -> str:
        """Format seconds to YouTube timestamp format."""
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    async def upload_video(
        self, 
        project_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload video to YouTube.
        
        Args:
            project_name: Name of the project
            metadata: Optional metadata (will generate if not provided)
            
        Returns:
            Dictionary with upload results
        """
        if not self.youtube_service:
            raise Exception("YouTube service not authenticated")
        
        with self.db_manager as session:
            project = session.query(Project).filter_by(name=project_name).first()
            if not project or not project.video_path:
                raise ValueError(f"Video not found for project {project_name}")
            
            # Generate metadata if not provided
            if not metadata:
                metadata = await self.generate_metadata(project_name)
            
            # Prepare upload body
            body = {
                'snippet': {
                    'title': metadata['title'],
                    'description': metadata['description'],
                    'tags': metadata['tags'],
                    'categoryId': metadata['category_id'],
                    'defaultLanguage': metadata.get('default_language', 'en'),
                    'defaultAudioLanguage': metadata.get('default_audio_language', 'en')
                },
                'status': {
                    'privacyStatus': metadata.get('privacy_status', 'private'),
                    'madeForKids': metadata.get('made_for_kids', False),
                    'selfDeclaredMadeForKids': metadata.get('made_for_kids', False)
                }
            }
            
            # Create media upload
            media = MediaFileUpload(
                project.video_path,
                chunksize=-1,
                resumable=True,
                mimetype='video/mp4'
            )
            
            try:
                # Call the API's videos.insert method
                insert_request = self.youtube_service.videos().insert(
                    part=",".join(body.keys()),
                    body=body,
                    media_body=media
                )
                
                # Execute upload
                response = self._resumable_upload(insert_request)
                
                # Update project with YouTube info
                project.youtube_id = response['id']
                project.youtube_url = f"https://www.youtube.com/watch?v={response['id']}"
                project.upload_date = datetime.utcnow()
                project.current_stage = "published"
                project.status = "completed"
                session.commit()
                
                log.info(f"Successfully uploaded video: {project.youtube_url}")
                
                return {
                    "success": True,
                    "video_id": response['id'],
                    "video_url": project.youtube_url,
                    "title": metadata['title']
                }
                
            except HttpError as e:
                log.error(f"HTTP error during upload: {e}")
                raise
            except Exception as e:
                log.error(f"Error during upload: {str(e)}")
                raise
    
    def _resumable_upload(self, insert_request):
        """Handle resumable upload with progress tracking."""
        response = None
        error = None
        retry = 0
        
        while response is None:
            try:
                status, response = insert_request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    log.info(f"Upload progress: {progress}%")
            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504]:
                    # Retry on server errors
                    error = f"Server error: {e}"
                    retry += 1
                    if retry > 5:
                        raise
                    log.warning(f"Retrying upload due to server error...")
                else:
                    raise
            except Exception as e:
                log.error(f"Unexpected error during upload: {str(e)}")
                raise
        
        return response
    
    async def update_video_metadata(
        self, 
        project_name: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update metadata for an already uploaded video."""
        with self.db_manager as session:
            project = session.query(Project).filter_by(name=project_name).first()
            if not project or not project.youtube_id:
                raise ValueError(f"YouTube video not found for project {project_name}")
            
            try:
                # Prepare update body
                body = {
                    'id': project.youtube_id,
                    'snippet': {
                        'title': updates.get('title', project.youtube_metadata['title']),
                        'description': updates.get('description', project.youtube_metadata['description']),
                        'tags': updates.get('tags', project.youtube_metadata['tags']),
                        'categoryId': updates.get('category_id', project.youtube_metadata['category_id'])
                    }
                }
                
                # Update video
                self.youtube_service.videos().update(
                    part='snippet',
                    body=body
                ).execute()
                
                # Update local metadata
                project.youtube_metadata.update(updates)
                session.commit()
                
                log.info(f"Successfully updated metadata for video: {project.youtube_id}")
                return True
                
            except Exception as e:
                log.error(f"Error updating video metadata: {str(e)}")
                return False
