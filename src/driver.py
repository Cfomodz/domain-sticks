"""
Main driver script for the Domain Sticks video creation pipeline.
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import click
from src.config.settings import settings
from src.models.database import DatabaseManager, Project, SourceURL
from src.modules.scraper import WebScraper
from src.modules.script_generator import ScriptGenerator
from src.modules.media_search import MediaSearcher
from src.modules.video_processor import VideoProcessor
from src.modules.youtube_uploader import YouTubeUploader
from src.modules.audio_video_processor import AudioVideoProcessor, AudioBatchProcessor
from src.modules.video_clip_processor import VideoClipProcessor, VideoBatchProcessor
from src.utils.openverse_auth import get_openverse_auth, setup_openverse_credentials
from src.utils.logger import log


class DomainSticksDriver:
    """Main driver class for orchestrating the video creation pipeline."""
    
    def __init__(self):
        # Ensure directories exist
        settings.ensure_directories()
        
        # Initialize database
        self.db_manager = DatabaseManager(settings.database_url)
        self.db_manager.create_tables()
        
        # Initialize modules
        self.scraper = WebScraper(self.db_manager)
        self.script_generator = ScriptGenerator(self.db_manager)
        self.media_searcher = MediaSearcher(self.db_manager)
        
        # Video processor is optional - only initialize if dependencies are available
        try:
            self.video_processor = VideoProcessor(self.db_manager)
        except ImportError as e:
            log.warning(f"Video processing not available: {e}")
            self.video_processor = None
        
        # Audio-video processor is optional - only initialize if dependencies are available  
        try:
            self.audio_video_processor = AudioVideoProcessor(self.db_manager)
            self.audio_batch_processor = AudioBatchProcessor(self.db_manager)
        except ImportError as e:
            log.warning(f"Audio-video processing not available: {e}")
            self.audio_video_processor = None
            self.audio_batch_processor = None
        
        # Video clip processor is optional - only initialize if dependencies are available
        try:
            self.video_clip_processor = VideoClipProcessor(self.db_manager)
            self.video_batch_processor = VideoBatchProcessor(self.db_manager)
        except ImportError as e:
            log.warning(f"Video clip processing not available: {e}")
            self.video_clip_processor = None
            self.video_batch_processor = None
            
        self.youtube_uploader = YouTubeUploader(self.db_manager)
        
        log.info("Domain Sticks Driver initialized")
    
    async def process_url(
        self, 
        url: str, 
        project_name: Optional[str] = None,
        auto_upload: bool = False,
        focus: str = "auto"
    ) -> Dict[str, Any]:
        """
        Process a single URL through the entire pipeline.
        
        Args:
            url: The URL to process
            project_name: Optional project name (will generate if not provided)
            auto_upload: Whether to automatically upload to YouTube
            focus: Story focus (auto, creator, subject, work)
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Generate project name if not provided
            if not project_name:
                project_name = self._generate_project_name(url)
            
            log.info(f"Starting processing for URL: {url}")
            log.info(f"Project name: {project_name}")
            
            # Step 1: Scrape URL and check for duplicates
            log.info("Step 1: Scraping URL...")
            scraped_data = await self.scraper.scrape_url(url)
            
            if scraped_data.get("duplicate"):
                log.warning(f"URL already processed: {url}")
                # Check if we should create a new variant
                existing_project = self._get_existing_project(url)
                if existing_project:
                    return {
                        "status": "duplicate",
                        "message": "URL already processed",
                        "existing_project": existing_project
                    }
            
            # Create project in database
            project_id = self._create_project(project_name, scraped_data)
            
            # Move to analysis stage
            self._move_project_files(project_name, "ingestion", "analysis")
            
            # Step 2: Generate script
            log.info("Step 2: Generating script...")
            script_data = await self.script_generator.generate_script(
                scraped_data["metadata"],
                project_name,
                focus=focus
            )
            
            # Move to script generation stage
            self._move_project_files(project_name, "analysis", "script_generation")
            
            # Step 3: Search for media
            log.info("Step 3: Searching for media...")
            keywords = script_data.get("keywords", [])
            visual_cues = script_data.get("visual_cues", [])
            
            # Combine keywords and visual cues for search
            search_terms = list(set(keywords + visual_cues))[:10]
            
            # Check if source has media
            source_media = scraped_data["metadata"].get("media_urls", {})
            media_items = []
            
            if source_media.get("videos") or source_media.get("images"):
                # Use source media if available
                log.info("Using media from source URL")
                # Download source media
                # TODO: Implement source media download
            else:
                # Search for public domain media
                media_items = await self.media_searcher.search_media(
                    search_terms,
                    media_type="all",
                    project_name=project_name,
                    limit=20
                )
            
            # If no video found, search for alternatives
            if not any(item["type"] == "video" for item in media_items):
                alternatives = await self.media_searcher.search_video_alternatives(
                    search_terms,
                    duration=settings.max_video_duration
                )
                media_items.extend(alternatives.get("images", []))
                media_items.extend(alternatives.get("audio", []))
            
            # Move to media search stage
            self._move_project_files(project_name, "script_generation", "media_search")
            
            # Step 4: Create video
            if self.video_processor:
                log.info("Step 4: Creating video...")
                video_result = await self.video_processor.create_video(
                    project_name,
                    media_items,
                    script_data
                )
                
                # Generate thumbnail
                thumbnail_path = await self.video_processor.generate_thumbnail(project_name)
            else:
                log.warning("Step 4: Skipping video creation (video processor not available)")
                video_result = {"status": "skipped", "reason": "Video processor not available"}
                thumbnail_path = None
            
            # Move to video processing stage
            self._move_project_files(project_name, "media_search", "video_processing")
            
            # Step 5: Generate metadata
            log.info("Step 5: Generating YouTube metadata...")
            youtube_metadata = await self.youtube_uploader.generate_metadata(project_name)
            
            # Move to metadata stage
            self._move_project_files(project_name, "video_processing", "metadata")
            
            # Step 6: Review/Approval stage
            self._move_project_files(project_name, "metadata", "approval")
            
            result = {
                "status": "success",
                "project_name": project_name,
                "video_path": video_result.get("output_path"),
                "thumbnail_path": thumbnail_path,
                "duration": video_result.get("duration"),
                "youtube_metadata": youtube_metadata,
                "script": script_data["script"],
                "video_processing_status": video_result.get("status", "completed")
            }
            
            # Step 7: Upload if auto_upload is True
            if auto_upload:
                if video_result.get("output_path"):
                    log.info("Step 7: Uploading to YouTube...")
                    upload_result = await self.youtube_uploader.upload_video(
                        project_name,
                        youtube_metadata
                    )
                    result["youtube_url"] = upload_result["video_url"]
                    result["youtube_id"] = upload_result["video_id"]
                else:
                    log.warning("Step 7: Skipping YouTube upload (no video file created)")
                    result["upload_status"] = "skipped - no video file"
                
                # Move to published stage
                self._move_project_files(project_name, "approval", "published")
            else:
                log.info("Video ready for manual review and upload")
                result["message"] = "Video created successfully. Ready for manual upload."
            
            return result
            
        except Exception as e:
            log.error(f"Error processing URL: {str(e)}")
            # Update project status
            self._update_project_status(project_name, "failed", str(e))
            raise
    
    async def process_batch(
        self, 
        urls: List[str], 
        auto_upload: bool = False
    ) -> List[Dict[str, Any]]:
        """Process multiple URLs in batch."""
        results = []
        
        for url in urls:
            try:
                result = await self.process_url(url, auto_upload=auto_upload)
                results.append(result)
                
                # Add delay between processing to avoid rate limits
                await asyncio.sleep(5)
                
            except Exception as e:
                log.error(f"Failed to process {url}: {str(e)}")
                results.append({
                    "status": "failed",
                    "url": url,
                    "error": str(e)
                })
        
        return results
    
    async def process_audio_file(
        self,
        audio_file_path: str,
        project_name: Optional[str] = None,
        generate_shorts: bool = True,
        grayscale: bool = False
    ) -> Dict[str, Any]:
        """
        Process an audio file to create a full video and optional shorts.
        
        Args:
            audio_file_path: Path to the audio file
            project_name: Optional project name (generated if not provided)
            generate_shorts: Whether to generate shortform content clips
            
        Returns:
            Dictionary with processing results
        """
        if not self.audio_video_processor:
            raise ValueError("Audio-video processing not available. Please install required dependencies.")
        
        try:
            audio_path = Path(audio_file_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            log.info(f"Starting audio processing for: {audio_file_path}")
            
            # Process the audio file
            result = await self.audio_video_processor.process_audio_file(
                audio_path,
                project_name=project_name,
                generate_shorts=generate_shorts,
                grayscale=grayscale
            )
            
            return result
            
        except Exception as e:
            log.error(f"Error processing audio file: {str(e)}")
            raise
    
    async def process_all_audio_files(
        self,
        generate_shorts: bool = True
    ) -> List[Dict[str, Any]]:
        """Process all audio files in the ingestion directory."""
        if not self.audio_batch_processor:
            raise ValueError("Audio batch processing not available. Please install required dependencies.")
        
        try:
            log.info("Starting batch audio processing...")
            
            results = await self.audio_batch_processor.process_ingestion_directory(
                generate_shorts=generate_shorts
            )
            
            return results
            
        except Exception as e:
            log.error(f"Error processing audio files: {str(e)}")
            raise
    
    async def process_video_file(
        self,
        video_file_path: str,
        project_name: Optional[str] = None,
        generate_shorts: bool = True,
        max_short_duration: int = 60,
        min_short_duration: int = 15,
        convert_to_vertical: bool = True,
        add_subtitles: bool = False
    ) -> Dict[str, Any]:
        """
        Process a video file to create short clips based on audio analysis.
        
        Args:
            video_file_path: Path to the video file
            project_name: Optional project name (generated if not provided)
            generate_shorts: Whether to generate shortform content clips
            max_short_duration: Maximum duration for shorts (seconds)
            min_short_duration: Minimum duration for shorts (seconds)
            convert_to_vertical: Whether to convert clips to vertical format (9:16)
            add_subtitles: Whether to add subtitles to clips
            
        Returns:
            Dictionary with processing results
        """
        if not self.video_clip_processor:
            raise ValueError("Video clip processing not available. Please install required dependencies.")
        
        try:
            video_path = Path(video_file_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_file_path}")
            
            log.info(f"Starting video processing for: {video_file_path}")
            
            # Process the video file
            result = await self.video_clip_processor.process_video_file(
                video_path,
                project_name=project_name,
                generate_shorts=generate_shorts,
                max_short_duration=max_short_duration,
                min_short_duration=min_short_duration,
                convert_to_vertical=convert_to_vertical,
                add_subtitles=add_subtitles
            )
            
            return result
            
        except Exception as e:
            log.error(f"Error processing video file: {str(e)}")
            raise
    
    async def process_all_video_files(
        self,
        generate_shorts: bool = True,
        convert_to_vertical: bool = True,
        add_subtitles: bool = False
    ) -> List[Dict[str, Any]]:
        """Process all video files in the ingestion directory."""
        if not self.video_batch_processor:
            raise ValueError("Video batch processing not available. Please install required dependencies.")
        
        try:
            log.info("Starting batch video processing...")
            
            results = await self.video_batch_processor.process_ingestion_directory(
                generate_shorts=generate_shorts,
                convert_to_vertical=convert_to_vertical,
                add_subtitles=add_subtitles
            )
            
            return results
            
        except Exception as e:
            log.error(f"Error processing video files: {str(e)}")
            raise
    
    async def process_all_content(
        self,
        generate_shorts: bool = True,
        auto_upload: bool = False
    ) -> Dict[str, Any]:
        """
        Process all content in the ingestion directory through all phases.
        
        This includes:
        1. All audio files → video creation
        2. All URL-based projects in various stages
        3. Move everything through the complete pipeline
        
        Args:
            generate_shorts: Whether to generate shortform content for audio
            auto_upload: Whether to automatically upload completed videos
            
        Returns:
            Dictionary with comprehensive processing results
        """
        try:
            log.info("🚀 Starting comprehensive content processing...")
            log.info("📂 Processing all content types through all phases")
            
            results = {
                "audio_processing": [],
                "url_processing": [],
                "stage_processing": {},
                "summary": {}
            }
            
            # Step 1: Process all audio files in ingestion
            log.info("🎵 Step 1: Processing audio files...")
            if self.audio_batch_processor:
                audio_results = await self.process_all_audio_files(generate_shorts=generate_shorts)
                results["audio_processing"] = audio_results
                log.info(f"✅ Processed {len(audio_results)} audio files")
            else:
                log.warning("⚠️ Audio processing unavailable - skipping audio files")
            
            # Step 1.5: Process all video files in ingestion  
            log.info("🎬 Step 1.5: Processing video files...")
            if self.video_batch_processor:
                video_results = await self.process_all_video_files(generate_shorts=generate_shorts)
                results["video_processing"] = video_results
                log.info(f"✅ Processed {len(video_results)} video files")
            else:
                log.warning("⚠️ Video processing unavailable - skipping video files")
            
            # Step 2: Process all URL-based projects through their stages
            log.info("🌐 Step 2: Processing URL-based projects through pipeline stages...")
            
            stages = ["analysis", "script_generation", "media_search", "video_processing", "metadata"]
            
            for stage in stages:
                try:
                    log.info(f"📍 Processing stage: {stage}")
                    stage_result = await self.process_workflow_stage(stage)
                    results["stage_processing"][stage] = {"status": "completed"}
                except Exception as e:
                    log.error(f"❌ Error in stage {stage}: {e}")
                    results["stage_processing"][stage] = {"status": "failed", "error": str(e)}
            
            # Step 3: Auto-upload if requested
            if auto_upload:
                log.info("📤 Step 3: Auto-uploading approved videos...")
                try:
                    await self.process_workflow_stage("approval")
                    results["stage_processing"]["approval"] = {"status": "completed"}
                except Exception as e:
                    log.error(f"❌ Error in approval stage: {e}")
                    results["stage_processing"]["approval"] = {"status": "failed", "error": str(e)}
            
            # Generate summary
            audio_success = sum(1 for r in results["audio_processing"] if r.get("status") == "success")
            audio_total = len(results["audio_processing"])
            video_success = sum(1 for r in results.get("video_processing", []) if r.get("status") == "success")
            video_total = len(results.get("video_processing", []))
            stage_success = sum(1 for s in results["stage_processing"].values() if s["status"] == "completed")
            stage_total = len(results["stage_processing"])
            
            results["summary"] = {
                "audio_files_processed": audio_total,
                "audio_files_successful": audio_success,
                "video_files_processed": video_total,
                "video_files_successful": video_success,
                "stages_processed": stage_total,
                "stages_successful": stage_success,
                "total_shorts_generated": sum(r.get("shorts_count", 0) for r in results["audio_processing"]) + sum(r.get("shorts_count", 0) for r in results.get("video_processing", [])),
                "auto_upload_enabled": auto_upload
            }
            
            log.info("✅ Comprehensive content processing completed!")
            log.info(f"📊 Summary: {audio_success}/{audio_total} audio files, {stage_success}/{stage_total} stages")
            
            return results
            
        except Exception as e:
            log.error(f"❌ Error in comprehensive processing: {str(e)}")
            raise
    
    async def process_workflow_stage(self, stage: str):
        """Process all projects in a specific workflow stage."""
        stage_path = settings.workflow_paths.get(stage)
        if not stage_path:
            raise ValueError(f"Unknown stage: {stage}")
        
        # Get all project directories in stage
        project_dirs = [d for d in stage_path.iterdir() if d.is_dir()]
        
        log.info(f"Processing {len(project_dirs)} projects in stage: {stage}")
        
        for project_dir in project_dirs:
            project_name = project_dir.name
            
            try:
                if stage == "analysis":
                    # Generate script for projects in analysis
                    await self._process_analysis_stage(project_name)
                elif stage == "script_generation":
                    # Search media for projects with scripts
                    await self._process_script_stage(project_name)
                elif stage == "media_search":
                    # Create videos for projects with media
                    await self._process_media_stage(project_name)
                elif stage == "video_processing":
                    # Generate metadata for completed videos
                    await self._process_video_stage(project_name)
                elif stage == "approval":
                    # Upload approved videos
                    await self._process_approval_stage(project_name)
                
            except Exception as e:
                log.error(f"Error processing {project_name} in stage {stage}: {str(e)}")
    
    def _generate_project_name(self, url: str) -> str:
        """Generate a unique project name from URL."""
        from urllib.parse import urlparse
        
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        path = parsed.path.strip("/").replace("/", "_")
        
        base_name = f"{domain}_{path}" if path else domain
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{base_name}_{timestamp}"
    
    def _create_project(
        self, 
        project_name: str, 
        scraped_data: Dict[str, Any]
    ) -> str:
        """Create project in database."""
        with self.db_manager as session:
            # Get or create source URL
            source_url = session.query(SourceURL).filter_by(
                url=scraped_data["url"]
            ).first()
            
            if not source_url:
                source_url = SourceURL(
                    url=scraped_data["url"],
                    url_hash=scraped_data["url_hash"],
                    metadata=scraped_data["metadata"]
                )
                session.add(source_url)
            
            # Create project
            project = Project(
                name=project_name,
                source_url=source_url,
                title=scraped_data["metadata"].get("title"),
                description=scraped_data["metadata"].get("description"),
                current_stage="ingestion"
            )
            
            session.add(project)
            session.commit()
            
            # Create project directory
            project_dir = settings.workflow_paths["ingestion"] / project_name
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Save scraped data
            with open(project_dir / "scraped_data.json", "w") as f:
                json.dump(scraped_data, f, indent=2)
            
            return str(project.id)
    
    def _move_project_files(
        self, 
        project_name: str, 
        from_stage: str, 
        to_stage: str
    ):
        """Move project files between workflow stages."""
        from_path = settings.workflow_paths[from_stage] / project_name
        to_path = settings.workflow_paths[to_stage] / project_name
        
        if from_path.exists():
            to_path.parent.mkdir(parents=True, exist_ok=True)
            from_path.rename(to_path)
            
            # Update project stage in database
            with self.db_manager as session:
                project = session.query(Project).filter_by(name=project_name).first()
                if project:
                    project.current_stage = to_stage
                    session.commit()
            
            log.info(f"Moved project {project_name} from {from_stage} to {to_stage}")
    
    def _update_project_status(
        self, 
        project_name: str, 
        status: str, 
        error: Optional[str] = None
    ):
        """Update project status in database."""
        with self.db_manager as session:
            project = session.query(Project).filter_by(name=project_name).first()
            if project:
                project.status = status
                if error:
                    project.processing_log = {
                        "error": error,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                session.commit()
    
    def _get_existing_project(self, url: str) -> Optional[Dict[str, Any]]:
        """Get existing project for URL."""
        with self.db_manager as session:
            source_url = session.query(SourceURL).filter_by(url=url).first()
            if source_url and source_url.projects:
                project = source_url.projects[0]  # Get first project
                return {
                    "name": project.name,
                    "status": project.status,
                    "youtube_url": project.youtube_url
                }
        return None
    
    # Stage processing methods
    async def _process_analysis_stage(self, project_name: str):
        """Process project in analysis stage."""
        # Load scraped data
        project_dir = settings.workflow_paths["analysis"] / project_name
        with open(project_dir / "scraped_data.json", "r") as f:
            scraped_data = json.load(f)
        
        # Generate script
        script_data = await self.script_generator.generate_script(
            scraped_data["metadata"],
            project_name
        )
        
        # Save script data
        with open(project_dir / "script_data.json", "w") as f:
            json.dump(script_data, f, indent=2)
        
        # Move to next stage
        self._move_project_files(project_name, "analysis", "script_generation")
    
    async def _process_script_stage(self, project_name: str):
        """Process project in script generation stage."""
        # Similar implementation for other stages...
        pass
    
    async def _process_media_stage(self, project_name: str):
        """Process project in media search stage."""
        pass
    
    async def _process_video_stage(self, project_name: str):
        """Process project in video processing stage."""
        pass
    
    async def _process_approval_stage(self, project_name: str):
        """Process project in approval stage."""
        pass


# CLI Commands
@click.group()
def cli():
    """Domain Sticks - Short-form video creation pipeline."""
    pass


@cli.command()
@click.argument('url')
@click.option('--name', help='Project name')
@click.option('--auto-upload', is_flag=True, help='Automatically upload to YouTube')
@click.option('--focus', default='auto', help='Story focus: auto, creator, subject, work')
def process(url, name, auto_upload, focus):
    """Process a single URL."""
    driver = DomainSticksDriver()
    result = asyncio.run(driver.process_url(url, name, auto_upload, focus))
    
    if result["status"] == "success":
        click.echo(f"✅ Successfully processed: {result['project_name']}")
        click.echo(f"📹 Video: {result['video_path']}")
        if result.get("youtube_url"):
            click.echo(f"🔗 YouTube: {result['youtube_url']}")
    else:
        click.echo(f"❌ Processing failed: {result.get('message', 'Unknown error')}")


@cli.command()
@click.argument('urls_file', type=click.File('r'))
@click.option('--auto-upload', is_flag=True, help='Automatically upload to YouTube')
def batch(urls_file, auto_upload):
    """Process multiple URLs from a file."""
    urls = [line.strip() for line in urls_file if line.strip()]
    
    driver = DomainSticksDriver()
    results = asyncio.run(driver.process_batch(urls, auto_upload))
    
    # Summary
    success_count = sum(1 for r in results if r["status"] == "success")
    click.echo(f"\n📊 Batch processing complete:")
    click.echo(f"✅ Successful: {success_count}")
    click.echo(f"❌ Failed: {len(results) - success_count}")


@cli.command()
@click.argument('stage', type=click.Choice([
    'analysis', 'script_generation', 'media_search', 
    'video_processing', 'approval'
]))
def process_stage(stage):
    """Process all projects in a specific workflow stage."""
    driver = DomainSticksDriver()
    asyncio.run(driver.process_workflow_stage(stage))


@cli.command()
@click.argument('project_name')
def status(project_name):
    """Check status of a project."""
    driver = DomainSticksDriver()
    
    with driver.db_manager as session:
        project = session.query(Project).filter_by(name=project_name).first()
        
        if not project:
            click.echo(f"❌ Project not found: {project_name}")
            return
        
        click.echo(f"📊 Project: {project.name}")
        click.echo(f"📍 Stage: {project.current_stage}")
        click.echo(f"✅ Status: {project.status}")
        
        if project.youtube_url:
            click.echo(f"🔗 YouTube: {project.youtube_url}")
        
        if project.video_path:
            click.echo(f"📹 Video: {project.video_path}")
        
        # Show audio project details if it exists
        if project.audio_project:
            click.echo(f"🎵 Audio Source: {project.audio_project.audio_file_path}")
            click.echo(f"🔤 Transcription: {project.audio_project.transcription_status}")
            click.echo(f"📱 Shorts: {project.audio_project.shortform_clips_count}")
        
        # Show video project details if it exists
        if project.video_project:
            click.echo(f"🎬 Video Source: {project.video_project.video_file_path}")
            click.echo(f"📐 Dimensions: {project.video_project.video_width}x{project.video_project.video_height}")
            click.echo(f"🔤 Transcription: {project.video_project.transcription_status}")
            click.echo(f"✂️ Clip Generation: {project.video_project.clip_generation_status}")
            click.echo(f"📱 Shorts: {project.video_project.shortform_clips_count}")


@cli.command()
@click.argument('audio_file_path', type=click.Path(exists=True))
@click.option('--name', help='Project name')
@click.option('--no-shorts', is_flag=True, help='Skip shortform content generation')
@click.option('--grayscale', is_flag=True, help='Convert all images to black and white')
def process_audio(audio_file_path, name, no_shorts, grayscale):
    """Process an audio file to create videos and shorts."""
    driver = DomainSticksDriver()
    
    try:
        result = asyncio.run(driver.process_audio_file(
            audio_file_path, 
            project_name=name,
            generate_shorts=not no_shorts,
            grayscale=grayscale
        ))
        
        if result["status"] == "success":
            click.echo(f"✅ Successfully processed audio: {result['project_name']}")
            click.echo(f"📹 Main video: {result['main_video']['video_path']}")
            click.echo(f"⏱️ Duration: {result['main_video']['duration']:.1f}s")
            click.echo(f"🔍 Keywords: {', '.join(result['keywords'][:5])}")
            if grayscale:
                click.echo("🎨 Images converted to grayscale")
            
            if result.get("shorts"):
                click.echo(f"📱 Generated {result['shorts_count']} shortform clips:")
                for i, short in enumerate(result["shorts"], 1):
                    click.echo(f"  {i}. {short['metadata']['title']} ({short['metadata']['duration']:.1f}s)")
        else:
            click.echo(f"❌ Processing failed: {result.get('message', 'Unknown error')}")
    
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@cli.command()
@click.option('--no-shorts', is_flag=True, help='Skip shortform content generation')
def process_all_audio(no_shorts):
    """Process all audio files in the ingestion directory."""
    driver = DomainSticksDriver()
    
    try:
        results = asyncio.run(driver.process_all_audio_files(
            generate_shorts=not no_shorts
        ))
        
        success_count = sum(1 for r in results if r.get("status") == "success")
        total_shorts = sum(r.get("shorts_count", 0) for r in results if r.get("status") == "success")
        
        click.echo(f"\n📊 Audio batch processing complete:")
        click.echo(f"✅ Successful: {success_count}")
        click.echo(f"❌ Failed: {len(results) - success_count}")
        click.echo(f"📱 Total shorts generated: {total_shorts}")
        
        # Show details for successful projects
        for result in results:
            if result.get("status") == "success":
                click.echo(f"\n🎵 {result['project_name']}:")
                click.echo(f"  📹 Video: {result['main_video']['video_path']}")
                if result.get("shorts"):
                    click.echo(f"  📱 Shorts: {result['shorts_count']}")
    
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@cli.command()
@click.argument('video_file_path', type=click.Path(exists=True))
@click.option('--name', help='Project name')
@click.option('--no-shorts', is_flag=True, help='Skip shortform content generation')
@click.option('--max-duration', default=60, type=int, help='Maximum duration for shorts (seconds)')
@click.option('--min-duration', default=15, type=int, help='Minimum duration for shorts (seconds)')
@click.option('--keep-original-format', is_flag=True, help='Keep original video format (don\'t convert to vertical)')
@click.option('--add-subtitles', is_flag=True, help='Add subtitles to clips')
def process_video(video_file_path, name, no_shorts, max_duration, min_duration, keep_original_format, add_subtitles):
    """Process a video file to create short clips based on audio analysis."""
    driver = DomainSticksDriver()
    
    try:
        result = asyncio.run(driver.process_video_file(
            video_file_path, 
            project_name=name,
            generate_shorts=not no_shorts,
            max_short_duration=max_duration,
            min_short_duration=min_duration,
            convert_to_vertical=not keep_original_format,
            add_subtitles=add_subtitles
        ))
        
        if result["status"] == "success":
            click.echo(f"✅ Successfully processed video: {result['project_name']}")
            click.echo(f"📹 Original video: {result['original_video']['path']}")
            click.echo(f"⏱️ Duration: {result['original_video']['duration']:.1f}s")
            click.echo(f"📐 Dimensions: {result['original_video']['width']}x{result['original_video']['height']}")
            click.echo(f"🔍 Keywords: {', '.join(result['keywords'][:5])}")
            if not keep_original_format:
                click.echo("📱 Clips converted to vertical format")
            if add_subtitles:
                click.echo("📝 Subtitles added to clips")
            
            if result.get("shorts"):
                click.echo(f"📱 Generated {result['shorts_count']} shortform clips:")
                for i, short in enumerate(result["shorts"], 1):
                    click.echo(f"  {i}. {short['metadata']['title']} ({short['metadata']['duration']:.1f}s)")
        else:
            click.echo(f"❌ Processing failed: {result.get('message', 'Unknown error')}")
    
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@cli.command()
@click.option('--no-shorts', is_flag=True, help='Skip shortform content generation')
@click.option('--keep-original-format', is_flag=True, help='Keep original video format (don\'t convert to vertical)')
@click.option('--add-subtitles', is_flag=True, help='Add subtitles to clips')
def process_all_videos(no_shorts, keep_original_format, add_subtitles):
    """Process all video files in the ingestion directory."""
    driver = DomainSticksDriver()
    
    try:
        results = asyncio.run(driver.process_all_video_files(
            generate_shorts=not no_shorts,
            convert_to_vertical=not keep_original_format,
            add_subtitles=add_subtitles
        ))
        
        success_count = sum(1 for r in results if r.get("status") == "success")
        total_shorts = sum(r.get("shorts_count", 0) for r in results if r.get("status") == "success")
        
        click.echo(f"\n📊 Video batch processing complete:")
        click.echo(f"✅ Successful: {success_count}")
        click.echo(f"❌ Failed: {len(results) - success_count}")
        click.echo(f"📱 Total shorts generated: {total_shorts}")
        
        # Show details for successful projects
        for result in results:
            if result.get("status") == "success":
                click.echo(f"\n🎬 {result['project_name']}:")
                click.echo(f"  📹 Original: {result['original_video']['path']}")
                click.echo(f"  ⏱️ Duration: {result['original_video']['duration']:.1f}s")
                if result.get("shorts"):
                    click.echo(f"  📱 Shorts: {result['shorts_count']}")
    
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@cli.command()
@click.option('--no-shorts', is_flag=True, help='Skip shortform content generation')
@click.option('--auto-upload', is_flag=True, help='Automatically upload completed videos to YouTube')
def process_all(no_shorts, auto_upload):
    """Process ALL content through all phases - audio files, URL projects, everything!"""
    driver = DomainSticksDriver()
    
    try:
        click.echo("🚀 Starting comprehensive content processing...")
        click.echo("📂 This will process ALL content types through ALL phases:")
        click.echo("   🎵 Audio files → video creation")
        click.echo("   🎬 Video files → short clip generation") 
        click.echo("   🌐 URL-based projects → complete pipeline")
        click.echo("   📍 All workflow stages → progression")
        if auto_upload:
            click.echo("   📤 Auto-upload enabled")
        click.echo()
        
        results = asyncio.run(driver.process_all_content(
            generate_shorts=not no_shorts,
            auto_upload=auto_upload
        ))
        
        # Display comprehensive summary
        summary = results["summary"]
        
        click.echo("🎉 Comprehensive processing complete!")
        click.echo("=" * 50)
        
        # Audio processing summary
        click.echo(f"🎵 Audio Processing:")
        click.echo(f"   ✅ Successful: {summary['audio_files_successful']}")
        click.echo(f"   📁 Total files: {summary['audio_files_processed']}")
        
        # Video processing summary
        click.echo(f"\n🎬 Video Processing:")
        click.echo(f"   ✅ Successful: {summary['video_files_successful']}")
        click.echo(f"   📁 Total files: {summary['video_files_processed']}")
        
        click.echo(f"\n📱 Total Shorts Generated: {summary['total_shorts_generated']}")
        
        # Stage processing summary
        click.echo(f"\n🌐 Pipeline Stages:")
        click.echo(f"   ✅ Completed: {summary['stages_successful']}")
        click.echo(f"   📍 Total stages: {summary['stages_processed']}")
        
        # Show stage details
        for stage, result in results["stage_processing"].items():
            status_icon = "✅" if result["status"] == "completed" else "❌"
            click.echo(f"     {status_icon} {stage}")
            if result["status"] == "failed":
                click.echo(f"       Error: {result['error']}")
        
        # Show successful audio projects
        if results["audio_processing"]:
            click.echo(f"\n🎵 Audio Projects Created:")
            for result in results["audio_processing"]:
                if result.get("status") == "success":
                    click.echo(f"   📹 {result['project_name']}")
                    if result.get("shorts"):
                        click.echo(f"     📱 {result['shorts_count']} shorts")
        
        # Show successful video projects
        if results.get("video_processing"):
            click.echo(f"\n🎬 Video Projects Created:")
            for result in results["video_processing"]:
                if result.get("status") == "success":
                    click.echo(f"   📹 {result['project_name']}")
                    if result.get("shorts"):
                        click.echo(f"     📱 {result['shorts_count']} shorts")
        
        if auto_upload:
            click.echo(f"\n📤 Auto-upload: {'Enabled' if auto_upload else 'Disabled'}")
        
        click.echo(f"\n💡 Use 'python -m src.driver status <project_name>' to check individual projects")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@cli.command()
@click.argument('project_name')
def show_shorts(project_name):
    """Show shortform clips for a project."""
    driver = DomainSticksDriver()
    
    with driver.db_manager as session:
        project = session.query(Project).filter_by(name=project_name).first()
        
        if not project:
            click.echo(f"❌ Project not found: {project_name}")
            return
        
        if not project.shortform_clips:
            click.echo(f"📱 No shortform clips found for: {project_name}")
            return
        
        click.echo(f"📱 Shortform clips for: {project_name}")
        click.echo("=" * 50)
        
        for clip in project.shortform_clips:
            click.echo(f"\n📍 Clip #{clip.clip_number}: {clip.title}")
            click.echo(f"   ⏱️ Duration: {clip.duration}s ({clip.start_time}s - {clip.end_time}s)")
            click.echo(f"   📄 Description: {clip.description}")
            click.echo(f"   🎯 Hook: {clip.hook}")
            click.echo(f"   🏷️ Tags: {', '.join(clip.tags) if clip.tags else 'None'}")
            click.echo(f"   📹 File: {clip.video_path}")
            click.echo(f"   ✅ Status: {clip.status}")


@cli.command()
def setup_openverse():
    """Set up OpenVerse API credentials for media search."""
    setup_openverse_credentials()


@cli.command()
def openverse_status():
    """Check OpenVerse API authentication status."""
    auth_manager = get_openverse_auth()
    status = auth_manager.get_status()
    
    click.echo("🔑 OpenVerse API Status")
    click.echo("=" * 30)
    
    if status["authenticated"]:
        click.echo("✅ Authentication: Valid")
        if status["token_expires_at"]:
            click.echo(f"🕒 Token expires: {status['token_expires_at']}")
    else:
        click.echo("❌ Authentication: Invalid or missing")
    
    if status["has_app_registration"]:
        click.echo("✅ Application: Registered")
    else:
        click.echo("❌ Application: Not registered")
    
    if status["client_id"]:
        click.echo("✅ Credentials: Configured in .env")
    else:
        click.echo("❌ Credentials: Missing from .env")
        click.echo("\n💡 Run 'python -m src.driver setup-openverse' to configure")


if __name__ == "__main__":
    cli()
