"""
Video Processing Module using FFmpeg for creating vertical videos.
"""
import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
try:
    import ffmpeg
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.video.VideoClip import ImageClip, TextClip
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy import concatenate_videoclips
    from moviepy.audio.AudioClip import CompositeAudioClip
    from moviepy.video.fx import Resize as resize, FadeIn as fadein, FadeOut as fadeout
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    VIDEO_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Video processing dependencies not available: {e}")
    VIDEO_PROCESSING_AVAILABLE = False
    # Create dummy classes to prevent NameError
    class VideoFileClip: pass
    class ImageClip: pass
    class TextClip: pass
    class CompositeVideoClip: pass
    class AudioFileClip: pass
    def concatenate_videoclips(*args, **kwargs): pass
    def resize(*args, **kwargs): pass
    def fadein(*args, **kwargs): pass
    def fadeout(*args, **kwargs): pass
    Image = None
    ImageDraw = None
    ImageFont = None
    np = None
from src.utils.logger import log
from src.config.settings import settings
from src.models.database import DatabaseManager, Project, Media


class VideoProcessor:
    """Process and create vertical videos using FFmpeg and MoviePy."""
    
    def __init__(self, db_manager: DatabaseManager):
        if not VIDEO_PROCESSING_AVAILABLE:
            raise ImportError("Video processing dependencies not available. Please install moviepy, ffmpeg-python, pillow, and numpy.")
        
        self.db_manager = db_manager
        self.video_width = settings.video_width
        self.video_height = settings.video_height
        self.fps = settings.video_fps
        self.bitrate = settings.video_bitrate
        self.max_duration = settings.max_video_duration
    
    async def create_video(
        self, 
        project_name: str,
        media_items: List[Dict[str, Any]],
        script_data: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Create a vertical video from media items and script.
        
        Args:
            project_name: Name of the project
            media_items: List of media items (images, videos, audio)
            script_data: Script data including text and visual cues
            output_path: Optional output path for the video
            
        Returns:
            Dictionary with video creation results
        """
        try:
            # Get project from database
            with self.db_manager as session:
                project = session.query(Project).filter_by(name=project_name).first()
                if not project:
                    raise ValueError(f"Project {project_name} not found")
            
            # Prepare output path
            if not output_path:
                output_path = settings.workflow_paths["video_processing"] / f"{project_name}.mp4"
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Separate media by type
            images = [item for item in media_items if item.get("type") == "image"]
            videos = [item for item in media_items if item.get("type") == "video"]
            audio = [item for item in media_items if item.get("type") == "audio"]
            
            # Create video based on available media
            if videos:
                # Use existing video as base
                result = await self._process_with_video(videos[0], script_data, output_path)
            elif images:
                # Create slideshow from images
                result = await self._create_slideshow(images, script_data, audio, output_path)
            else:
                # Create text-only video
                result = await self._create_text_video(script_data, audio, output_path)
            
            # Update project in database
            with self.db_manager as session:
                project = session.query(Project).filter_by(name=project_name).first()
                project.video_path = str(output_path)
                project.duration = result.get("duration", self.max_duration)
                project.current_stage = "metadata"
                session.commit()
            
            return result
            
        except Exception as e:
            log.error(f"Error creating video: {str(e)}")
            raise
    
    async def _process_with_video(
        self, 
        video_item: Dict[str, Any], 
        script_data: Dict[str, Any], 
        output_path: Path
    ) -> Dict[str, Any]:
        """Process an existing video with script overlay."""
        video_path = video_item.get("file_path")
        if not video_path:
            raise ValueError("Video file path not provided")
        
        # Load video
        clip = VideoFileClip(video_path)
        
        # Resize to vertical format
        clip = self._resize_to_vertical(clip)
        
        # Trim to max duration
        if clip.duration > self.max_duration:
            clip = clip.subclip(0, self.max_duration)
        
        # Add text overlay
        final_clip = self._add_text_overlay(clip, script_data)
        
        # Write output
        final_clip.write_videofile(
            str(output_path),
            fps=self.fps,
            codec='libx264',
            audio_codec='aac',
            bitrate=self.bitrate
        )
        
        # Clean up
        clip.close()
        final_clip.close()
        
        return {
            "success": True,
            "output_path": str(output_path),
            "duration": final_clip.duration
        }
    
    async def _create_slideshow(
        self, 
        images: List[Dict[str, Any]], 
        script_data: Dict[str, Any],
        audio: List[Dict[str, Any]],
        output_path: Path
    ) -> Dict[str, Any]:
        """Create a slideshow video from images."""
        clips = []
        
        # Calculate duration per image
        total_duration = self.max_duration
        duration_per_image = total_duration / len(images) if images else 5
        
        # Create image clips
        for idx, image_item in enumerate(images):
            image_path = image_item.get("file_path")
            if not image_path:
                continue
            
            # Create image clip
            img_clip = ImageClip(image_path, duration=duration_per_image)
            
            # Resize to vertical
            img_clip = self._resize_to_vertical(img_clip)
            
            # Add fade effects
            img_clip = img_clip.with_effects([fadein(0.5), fadeout(0.5)])
            
            # Set position in timeline
            img_clip = img_clip.with_start(idx * duration_per_image)
            
            clips.append(img_clip)
        
        # Concatenate clips
        if clips:
            video = CompositeVideoClip(clips, size=(self.video_width, self.video_height))
        else:
            # Create blank video if no images
            video = self._create_blank_video(total_duration)
        
        # Add text overlay
        video = self._add_text_overlay(video, script_data)
        
        # Add audio if available
        if audio:
            video = self._add_audio(video, audio[0])
        
        # Write output
        video.write_videofile(
            str(output_path),
            fps=self.fps,
            codec='libx264',
            audio_codec='aac',
            bitrate=self.bitrate
        )
        
        # Clean up
        video.close()
        
        return {
            "success": True,
            "output_path": str(output_path),
            "duration": video.duration
        }
    
    async def _create_text_video(
        self, 
        script_data: Dict[str, Any],
        audio: List[Dict[str, Any]],
        output_path: Path
    ) -> Dict[str, Any]:
        """Create a video with only text and background."""
        # Create background
        background = self._create_gradient_background(self.max_duration)
        
        # Add text overlay
        video = self._add_text_overlay(background, script_data)
        
        # Add audio if available
        if audio:
            video = self._add_audio(video, audio[0])
        
        # Write output
        video.write_videofile(
            str(output_path),
            fps=self.fps,
            codec='libx264',
            audio_codec='aac',
            bitrate=self.bitrate
        )
        
        # Clean up
        video.close()
        
        return {
            "success": True,
            "output_path": str(output_path),
            "duration": video.duration
        }
    
    def _resize_to_vertical(self, clip):
        """Resize clip to vertical format (9:16)."""
        target_ratio = self.video_width / self.video_height
        clip_ratio = clip.w / clip.h
        
        if clip_ratio > target_ratio:
            # Clip is wider than target - fit height
            new_width = int(clip.h * target_ratio)
            clip = clip.crop(x_center=clip.w/2, width=new_width)
        else:
            # Clip is taller than target - fit width
            new_height = int(clip.w / target_ratio)
            clip = clip.crop(y_center=clip.h/2, height=new_height)
        
        # Resize to exact dimensions
        clip = clip.with_effects([resize((self.video_width, self.video_height))])
        
        return clip
    
    def _add_text_overlay(self, video_clip, script_data: Dict[str, Any]):
        """Add text overlay to video with proper timing."""
        script = script_data.get("script", "")
        hook = script_data.get("hook", "")
        
        # Split script into segments for better display
        segments = self._split_script(script)
        
        text_clips = []
        
        # Add hook text (first 3 seconds)
        if hook:
            hook_clip = self._create_text_clip(
                hook,
                duration=3,
                position='center',
                fontsize=60,
                color='white',
                stroke_color='black',
                stroke_width=2
            )
            hook_clip = hook_clip.with_start(0).with_effects([fadein(0.5), fadeout(0.5)])
            text_clips.append(hook_clip)
        
        # Add main script segments
        time_per_segment = (video_clip.duration - 3) / len(segments) if segments else 5
        
        for idx, segment in enumerate(segments):
            start_time = 3 + (idx * time_per_segment)
            
            text_clip = self._create_text_clip(
                segment,
                duration=time_per_segment,
                position='bottom',
                fontsize=40,
                color='white',
                stroke_color='black',
                stroke_width=2
            )
            
            text_clip = text_clip.with_start(start_time).with_effects([fadein(0.3), fadeout(0.3)])
            text_clips.append(text_clip)
        
        # Composite all elements
        final_video = CompositeVideoClip([video_clip] + text_clips)
        
        return final_video
    
    def _create_text_clip(
        self, 
        text: str, 
        duration: float,
        position: str = 'center',
        fontsize: int = 40,
        color: str = 'white',
        stroke_color: str = 'black',
        stroke_width: int = 2
    ) -> TextClip:
        """Create a text clip with custom styling."""
        # Create text clip
        txt_clip = TextClip(
            text=text,
            font_size=fontsize,
            color=color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            method='caption',
            size=(int(self.video_width * 0.9), None),
            text_align='center'
        ).with_duration(duration)
        
        # Position the text
        if position == 'center':
            txt_clip = txt_clip.with_position('center')
        elif position == 'bottom':
            txt_clip = txt_clip.with_position(('center', self.video_height * 0.8))
        elif position == 'top':
            txt_clip = txt_clip.with_position(('center', self.video_height * 0.1))
        
        return txt_clip
    
    def _split_script(self, script: str, max_words: int = 15) -> List[str]:
        """Split script into segments for display."""
        words = script.split()
        segments = []
        
        current_segment = []
        for word in words:
            current_segment.append(word)
            if len(current_segment) >= max_words:
                segments.append(' '.join(current_segment))
                current_segment = []
        
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments
    
    def _create_blank_video(self, duration: float) -> VideoFileClip:
        """Create a blank video with solid color."""
        # Create a blank clip with dark background
        color_clip = ImageClip(
            np.full((self.video_height, self.video_width, 3), (20, 20, 20), dtype=np.uint8),
            duration=duration
        )
        return color_clip
    
    def _create_gradient_background(self, duration: float) -> VideoFileClip:
        """Create a gradient background video."""
        # Create gradient image
        gradient = Image.new('RGB', (self.video_width, self.video_height))
        draw = ImageDraw.Draw(gradient)
        
        # Create vertical gradient
        for y in range(self.video_height):
            color_value = int(20 + (y / self.video_height) * 60)
            draw.rectangle([(0, y), (self.video_width, y+1)], fill=(color_value, 20, 80))
        
        # Convert to numpy array
        gradient_array = np.array(gradient)
        
        # Create video clip
        gradient_clip = ImageClip(gradient_array, duration=duration)
        
        return gradient_clip
    
    def _add_audio(self, video_clip, audio_item: Dict[str, Any]) -> VideoFileClip:
        """Add audio track to video."""
        audio_path = audio_item.get("file_path")
        if not audio_path:
            return video_clip
        
        try:
            # Load audio
            audio_clip = AudioFileClip(audio_path)
            
            # Adjust audio duration to match video
            if audio_clip.duration > video_clip.duration:
                audio_clip = audio_clip.subclip(0, video_clip.duration)
            elif audio_clip.duration < video_clip.duration:
                # Loop audio if too short
                loops_needed = int(video_clip.duration / audio_clip.duration) + 1
                audio_clip = concatenate_audioclips([audio_clip] * loops_needed)
                audio_clip = audio_clip.subclip(0, video_clip.duration)
            
            # Set audio volume
            audio_clip = audio_clip.volumex(0.3)  # Background music volume
            
            # Add audio to video
            video_clip = video_clip.with_audio(audio_clip)
            
        except Exception as e:
            log.warning(f"Failed to add audio: {str(e)}")
        
        return video_clip
    
    async def generate_thumbnail(
        self, 
        project_name: str,
        frame_time: Optional[float] = None
    ) -> str:
        """Generate thumbnail for the video."""
        with self.db_manager as session:
            project = session.query(Project).filter_by(name=project_name).first()
            if not project or not project.video_path:
                raise ValueError(f"Video not found for project {project_name}")
            
            video_path = project.video_path
            thumbnail_path = Path(video_path).with_suffix('.jpg')
            
            # Extract frame at specified time (default: 2 seconds)
            if frame_time is None:
                frame_time = min(2, project.duration / 2)
            
            # Use ffmpeg to extract frame
            stream = ffmpeg.input(video_path, ss=frame_time)
            stream = ffmpeg.output(stream, str(thumbnail_path), vframes=1)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            # Update project
            project.thumbnail_path = str(thumbnail_path)
            session.commit()
            
            return str(thumbnail_path)
