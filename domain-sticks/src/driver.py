"""
Main driver script for the Domain Sticks video creation pipeline.
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import click
from src.config.settings import settings
from src.models.database import DatabaseManager, Project, SourceURL
from src.modules.scraper import WebScraper
from src.modules.script_generator import ScriptGenerator
from src.modules.media_search import MediaSearcher
from src.modules.video_processor import VideoProcessor
from src.modules.youtube_uploader import YouTubeUploader
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
        self.video_processor = VideoProcessor(self.db_manager)
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
            log.info("Step 4: Creating video...")
            video_result = await self.video_processor.create_video(
                project_name,
                media_items,
                script_data
            )
            
            # Generate thumbnail
            thumbnail_path = await self.video_processor.generate_thumbnail(project_name)
            
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
                "video_path": video_result["output_path"],
                "thumbnail_path": thumbnail_path,
                "duration": video_result["duration"],
                "youtube_metadata": youtube_metadata,
                "script": script_data["script"]
            }
            
            # Step 7: Upload if auto_upload is True
            if auto_upload:
                log.info("Step 7: Uploading to YouTube...")
                upload_result = await self.youtube_uploader.upload_video(
                    project_name,
                    youtube_metadata
                )
                result["youtube_url"] = upload_result["video_url"]
                result["youtube_id"] = upload_result["video_id"]
                
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
                        "timestamp": datetime.utcnow().isoformat()
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


if __name__ == "__main__":
    cli()