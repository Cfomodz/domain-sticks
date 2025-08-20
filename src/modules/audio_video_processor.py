"""
Audio-to-Video Processing Module with Shortform Content Generation.

This module takes audio files, transcribes them using Whisper, generates videos
from the audio content with relevant visuals, and creates shortform clips based
on AI analysis of the transcript.
"""
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import hashlib
from datetime import datetime, timezone

try:
    import whisper
    import ffmpeg
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.video.VideoClip import ImageClip, TextClip
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy import concatenate_videoclips
    from moviepy.video.fx import Resize as resize, FadeIn as fadein, FadeOut as fadeout
    import numpy as np
    AUDIO_VIDEO_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Audio-video processing dependencies not available: {e}")
    AUDIO_VIDEO_PROCESSING_AVAILABLE = False
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
    whisper = None
    ffmpeg = None
    np = None

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import log
from src.config.settings import settings
from src.models.database import DatabaseManager, Project, Media, ShortformClip, AudioProject
from src.modules.media_search import MediaSearcher
from src.modules.video_processor import VideoProcessor


class AudioVideoProcessor:
    """Process audio files into videos with shortform content generation."""
    
    def __init__(self, db_manager: DatabaseManager):
        if not AUDIO_VIDEO_PROCESSING_AVAILABLE:
            raise ImportError("Audio-video processing dependencies not available. Please install whisper, moviepy, ffmpeg-python, and numpy.")
        
        self.db_manager = db_manager
        self.media_searcher = MediaSearcher(db_manager)
        self.video_processor = VideoProcessor(db_manager)
        
        # Initialize Whisper model with GPU support
        try:
            # Check for CUDA availability
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"🖥️ Device detected: {self.device}")
            if self.device == "cuda":
                log.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
                log.info(f"🚀 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Load model with numerical stability options for GPU
            if self.device == "cuda":
                self.whisper_model = whisper.load_model("base", device=self.device)
                # Set model to eval mode for stability
                self.whisper_model.eval()
                log.info(f"✅ Whisper model loaded successfully on {self.device} with stability options")
            else:
                self.whisper_model = whisper.load_model("base", device=self.device)
                log.info(f"✅ Whisper model loaded successfully on {self.device}")
        except ImportError:
            log.warning("⚠️ PyTorch not available, falling back to CPU")
            self.device = "cpu"
            try:
                self.whisper_model = whisper.load_model("base")
                log.info("✅ Whisper model loaded successfully on CPU")
            except Exception as e:
                log.error(f"❌ Failed to load Whisper model: {e}")
                self.whisper_model = None
        except Exception as e:
            log.error(f"❌ Failed to load Whisper model: {e}")
            self.whisper_model = None
            self.device = "cpu"
        
        # Initialize DeepSeek client for shortform analysis
        self.deepseek_client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_api_base
        )
        
        # Video settings
        self.video_width = settings.video_width
        self.video_height = settings.video_height
        self.fps = settings.video_fps
        self.bitrate = settings.video_bitrate
        
        # Log GPU capabilities
        self._log_gpu_capabilities()
    
    def _log_gpu_capabilities(self):
        """Log available GPU capabilities for video processing."""
        log.info("🖥️ GPU Capabilities Summary:")
        log.info(f"   Device: {getattr(self, 'device', 'Unknown')}")
        
        try:
            import torch
            if torch.cuda.is_available():
                log.info(f"   🚀 CUDA Available: Yes")
                log.info(f"   🚀 GPU Count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    log.info(f"   🚀 GPU {i}: {props.name}")
                    log.info(f"     - Memory: {props.total_memory / 1024**3:.1f} GB")
                    log.info(f"     - Compute Capability: {props.major}.{props.minor}")
                
                # Check for NVENC support (requires newer GPUs)
                major = torch.cuda.get_device_properties(0).major
                if major >= 6:  # Maxwell or newer
                    log.info("   🎬 NVENC Hardware Encoding: Supported")
                else:
                    log.info("   🎬 NVENC Hardware Encoding: Not supported (GPU too old)")
            else:
                log.info("   ❌ CUDA Available: No")
                log.info("   ⚠️ GPU acceleration disabled - will use CPU")
        except ImportError:
            log.info("   ❌ PyTorch not available")
            log.info("   ⚠️ GPU acceleration disabled - will use CPU")
        
        # Check for ffmpeg NVENC support
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=5)
            if 'h264_nvenc' in result.stdout:
                log.info("   🎬 FFmpeg NVENC: Available")
            else:
                log.info("   ❌ FFmpeg NVENC: Not available")
        except Exception:
            log.info("   ❓ FFmpeg NVENC: Could not check")
        
        log.info("   💡 Recommendations:")
        if getattr(self, 'device', 'cpu') == 'cpu':
            log.info("     - Install CUDA-enabled PyTorch for GPU acceleration")
            log.info("     - Ensure NVIDIA drivers are installed")
            log.info("     - Consider using a GPU for faster processing")
        else:
            log.info("     - GPU acceleration enabled - expect faster processing!")
    
    async def process_audio_file(
        self,
        audio_file_path: Path,
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
            Dictionary with processing results including main video and shorts
        """
        try:
            if not audio_file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Generate project name if not provided
            if not project_name:
                project_name = self._generate_project_name(audio_file_path)
            
            log.info(f"Processing audio file: {audio_file_path}")
            log.info(f"Project name: {project_name}")
            
            # Step 1: Transcribe audio with timing
            log.info("Step 1: Transcribing audio...")
            transcription_data = await self._transcribe_audio(audio_file_path)
            
            # Step 2: Extract keywords and generate metadata
            log.info("Step 2: Extracting keywords and metadata...")
            keywords = self._extract_keywords(transcription_data["text"])
            
            # Step 3: Search for relevant media
            log.info("Step 3: Searching for relevant media...")
            log.info(f"🔍 Calling media_searcher.search_media with:")
            log.info(f"   Keywords: {keywords}")
            log.info(f"   Media type: image")
            log.info(f"   Project name: {project_name}")
            log.info(f"   Limit: 30")
            
            media_items = await self.media_searcher.search_media(
                keywords,
                media_type="image",  # Focus on images for audio-to-video
                project_name=project_name,
                limit=30
            )
            
            log.info(f"✅ Media search completed. Received {len(media_items)} items")
            for i, item in enumerate(media_items):
                log.info(f"   Media {i+1}: {item.get('title', 'Untitled')} ({item.get('type', 'unknown')})")
                log.info(f"     Path: {item.get('file_path', 'No path')}")
                log.info(f"     Local: {item.get('local_path', 'No local path')}")
                log.info(f"     Cached: {item.get('cached', False)}")
            
            # Step 4: Create main video
            log.info("Step 4: Creating main video...")
            main_video_data = await self._create_main_video(
                audio_file_path,
                transcription_data,
                media_items,
                project_name,
                grayscale=grayscale
            )
            
            result = {
                "status": "success",
                "project_name": project_name,
                "main_video": main_video_data,
                "transcription": transcription_data,
                "keywords": keywords,
                "media_items_count": len(media_items)
            }
            
            # Step 5: Generate shortform content if requested
            if generate_shorts:
                log.info("Step 5: Generating shortform content...")
                shorts_data = await self._generate_shortform_content(
                    transcription_data,
                    main_video_data["video_path"],
                    project_name
                )
                result["shorts"] = shorts_data
                result["shorts_count"] = len(shorts_data)
            
            # Step 6: Store project in database
            self._store_project_data(project_name, result)
            
            return result
            
        except Exception as e:
            log.error(f"Error processing audio file: {str(e)}")
            raise
    
    async def _transcribe_audio(self, audio_file_path: Path) -> Dict[str, Any]:
        """Transcribe audio file using Whisper with word-level timing."""
        if not self.whisper_model:
            raise ValueError("Whisper model not available")
        
        try:
            # Transcribe with word-level timestamps and numerical stability
            transcribe_options = {
                "word_timestamps": True,
                "verbose": False,
                "fp16": False,  # Disable half precision to prevent NaN
                "temperature": 0.0,  # Use greedy decoding for stability
            }
            
            # Additional GPU stability measures with fallback
            if self.device == "cuda":
                import torch
                try:
                    with torch.inference_mode():  # Disable gradients for inference
                        result = self.whisper_model.transcribe(str(audio_file_path), **transcribe_options)
                    
                    # Check for NaN values in result (basic validation)
                    if not result.get("text") or len(result.get("segments", [])) == 0:
                        raise ValueError("Empty transcription result - possible GPU numerical issue")
                        
                except Exception as gpu_error:
                    log.warning(f"⚠️ GPU transcription failed: {gpu_error}")
                    log.info("🔄 Falling back to CPU transcription...")
                    
                    # Fallback to CPU
                    cpu_model = whisper.load_model("base", device="cpu")
                    result = cpu_model.transcribe(str(audio_file_path), **transcribe_options)
                    del cpu_model  # Clean up memory
            else:
                result = self.whisper_model.transcribe(str(audio_file_path), **transcribe_options)
            
            # Process segments to get timing data
            segments = []
            for segment in result.get("segments", []):
                segment_data = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "words": []
                }
                
                # Add word-level timing if available
                for word in segment.get("words", []):
                    segment_data["words"].append({
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"]
                    })
                
                segments.append(segment_data)
            
            return {
                "text": result["text"],
                "language": result.get("language", "en"),
                "segments": segments,
                "duration": segments[-1]["end"] if segments else 0
            }
            
        except Exception as e:
            log.error(f"Error transcribing audio: {str(e)}")
            raise
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from transcribed text."""
        # Simple keyword extraction - could be enhanced with NLP
        import re
        
        # Remove common stop words and clean text
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
            'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
            'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should', 'may', 
            'might', 'must', 'can', 'shall'
        }
        
        # Extract words and clean them
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = []
        
        # Count word frequency
        word_count = {}
        for word in words:
            if word not in stop_words:
                word_count[word] = word_count.get(word, 0) + 1
        
        # Get top keywords by frequency
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, count in sorted_words[:20] if count > 1]
        
        return keywords
    
    async def _create_main_video(
        self,
        audio_file_path: Path,
        transcription_data: Dict[str, Any],
        media_items: List[Dict[str, Any]],
        project_name: str,
        grayscale: bool = False
    ) -> Dict[str, Any]:
        """Create the main video from audio, transcript, and media."""
        try:
            log.info(f"🎬 Creating main video for project: {project_name}")
            log.info(f"📁 Audio file: {audio_file_path}")
            log.info(f"🖼️ Media items received: {len(media_items)}")
            
            for i, item in enumerate(media_items):
                log.info(f"   Item {i+1}: {item.get('title', 'Untitled')}")
                log.info(f"     Type: {item.get('type', 'unknown')}")
                log.info(f"     File path: {item.get('file_path', 'No file_path')}")
                log.info(f"     Local path: {item.get('local_path', 'No local_path')}")
            
            # Prepare output path
            output_path = settings.workflow_paths["video_processing"] / f"{project_name}_main.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            log.info(f"📹 Output path: {output_path}")
            
            # Load audio
            audio_clip = AudioFileClip(str(audio_file_path))
            duration = audio_clip.duration
            log.info(f"🎵 Audio duration: {duration} seconds")
            
            # Create video clips from images
            log.info("🖼️ Creating image sequence...")
            video_clips = self._create_image_sequence(
                media_items,
                duration,
                transcription_data["segments"],
                grayscale=grayscale
            )
            log.info(f"🖼️ Created {len(video_clips)} video clips from images")
            
            # Skip text overlays/subtitles as requested
            log.info("📝 Skipping text overlays (subtitles) as requested")
            
            # Use only video clips (no text overlays)
            final_clips = video_clips
            log.info(f"🎬 Final clips count: {len(final_clips)}")
            
            # Create composite video
            final_video = CompositeVideoClip(
                final_clips,
                size=(self.video_width, self.video_height)
            ).with_audio(audio_clip)
            
            # Write output with GPU acceleration if available
            log.info(f"💾 Writing video to: {output_path}")
            
            # Configure GPU-accelerated encoding if CUDA is available
            if self.device == "cuda":
                log.info("🚀 Using GPU-accelerated video encoding (NVENC)")
                # Use NVENC hardware encoder for GPU acceleration
                final_video.write_videofile(
                    str(output_path),
                    fps=self.fps,
                    codec='h264_nvenc',  # NVIDIA hardware encoder
                    audio_codec='aac',
                    bitrate=self.bitrate,
                    ffmpeg_params=[
                        '-hwaccel', 'cuda',
                        '-hwaccel_output_format', 'cuda',
                        '-preset', 'fast',
                        '-rc', 'vbr',
                        '-cq', '23',
                        '-spatial-aq', '1',
                        '-temporal-aq', '1'
                    ]
                )
            else:
                log.info("🖥️ Using CPU-based video encoding")
                final_video.write_videofile(
                    str(output_path),
                    fps=self.fps,
                    codec='libx264',
                    audio_codec='aac',
                    bitrate=self.bitrate,
                    ffmpeg_params=[
                        '-preset', 'medium',  # Good balance of speed/quality for CPU
                        '-crf', '23'          # Constant Rate Factor for quality
                    ]
                )
            
            # Clean up
            audio_clip.close()
            final_video.close()
            for clip in final_clips:
                if hasattr(clip, 'close'):
                    clip.close()
            
            return {
                "video_path": str(output_path),
                "duration": duration,
                "audio_source": str(audio_file_path),
                "image_count": len(media_items)
            }
            
        except Exception as e:
            log.error(f"Error creating main video: {str(e)}")
            raise
    
    def _create_image_sequence(
        self,
        media_items: List[Dict[str, Any]],
        total_duration: float,
        segments: List[Dict[str, Any]],
        grayscale: bool = False
    ) -> List[ImageClip]:
        """Create a sequence of image clips synchronized with speech segments."""
        log.info(f"🖼️ _create_image_sequence called with {len(media_items)} media items")
        log.info(f"⏱️ Total duration: {total_duration} seconds")
        
        if not media_items:
            log.warning("❌ No media items provided, creating solid background")
            # Create a solid background if no images
            return [self._create_solid_background(total_duration)]
        
        clips = []
        duration_per_image = max(3, total_duration / len(media_items))
        log.info(f"⏱️ Duration per image: {duration_per_image} seconds")
        
        for idx, media_item in enumerate(media_items):
            log.info(f"🖼️ Processing media item {idx+1}/{len(media_items)}")
            
            # Check file_path first
            image_path = media_item.get("file_path")
            if not image_path:
                # Fallback to local_path
                image_path = media_item.get("local_path")
            
            log.info(f"   Raw image path: {image_path}")
            
            if not image_path:
                log.warning(f"❌ Media item {idx+1} has no file_path or local_path, skipping")
                continue
            
            # Ensure path is absolute (resolve relative to project root)
            if not Path(image_path).is_absolute():
                image_path = Path.cwd() / image_path
            
            log.info(f"   Absolute image path: {image_path}")
            
            if not Path(image_path).exists():
                log.warning(f"❌ Image file does not exist: {image_path}")
                continue
            
            log.info(f"✅ Image file exists: {image_path}")
            
            start_time = idx * duration_per_image
            if start_time >= total_duration:
                log.info(f"⏹️ Start time {start_time} >= total duration {total_duration}, stopping")
                break
            
            # Create image clip (use string path for MoviePy compatibility)
            log.info(f"🎬 Creating ImageClip from: {image_path}")
            img_clip = ImageClip(str(image_path), duration=min(duration_per_image, total_duration - start_time))
            log.info(f"✅ ImageClip created successfully")
            
            # Resize to fit vertical format
            img_clip = self._resize_image_to_vertical(img_clip, grayscale=grayscale)
            
            # Add fade effects
            img_clip = img_clip.with_effects([fadein(0.5), fadeout(0.5)])
            
            # Set timing
            img_clip = img_clip.with_start(start_time)
            
            clips.append(img_clip)
        
        return clips
    
    def _create_text_overlays(self, segments: List[Dict[str, Any]]) -> List[TextClip]:
        """Create text overlay clips for each speech segment."""
        text_clips = []
        
        for segment in segments:
            text = segment["text"].strip()
            if not text:
                continue
            
            start_time = segment["start"]
            duration = segment["end"] - segment["start"]
            
            # Create text clip
            txt_clip = TextClip(
                text=text,
                font_size=40,
                color='white',
                stroke_color='black',
                stroke_width=2,
                method='caption',
                size=(int(self.video_width * 0.9), None),
                text_align='center'
            ).with_duration(duration)
            
            # Position at bottom of screen
            txt_clip = txt_clip.with_position(('center', self.video_height * 0.8))
            txt_clip = txt_clip.with_start(start_time)
            txt_clip = txt_clip.with_effects([fadein(0.2), fadeout(0.2)])
            
            text_clips.append(txt_clip)
        
        return text_clips
    
    def _resize_image_to_vertical(self, clip: ImageClip, grayscale: bool = False) -> ImageClip:
        """Resize image clip to vertical format (9:16) with proper aspect ratio handling."""
        log.info(f"🖼️ Processing image: {clip.w}x{clip.h} -> {self.video_width}x{self.video_height}")
        log.info(f"   Grayscale mode: {grayscale}")
        
        try:
            # Apply grayscale if requested
            if grayscale:
                log.info("🎨 Converting to grayscale...")
                try:
                    # Convert to grayscale using numpy
                    def to_grayscale(get_frame, t):
                        frame = get_frame(t)
                        # Convert RGB to grayscale using luminance formula
                        gray = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
                        # Convert back to RGB format by repeating the gray channel
                        return np.stack([gray, gray, gray], axis=2).astype(np.uint8)
                    
                    clip = clip.transform(to_grayscale)
                    log.info("✅ Grayscale applied successfully")
                except Exception as e:
                    log.warning(f"⚠️ Could not apply grayscale: {e}")
            
            # Calculate aspect ratios
            target_ratio = self.video_width / self.video_height  # e.g., 1080/1920 = 0.5625
            image_ratio = clip.w / clip.h
            
            log.info(f"   Target ratio: {target_ratio:.3f} (9:16)")
            log.info(f"   Image ratio: {image_ratio:.3f}")
            
            if abs(image_ratio - target_ratio) < 0.01:
                # Ratios are very close, just resize
                log.info("   Ratios match - simple resize")
                resized_clip = clip.resized((self.video_width, self.video_height))
            elif image_ratio > target_ratio:
                # Image is wider than target - scale to fit height, add black bars on sides
                log.info("   Image wider than target - fitting height with side bars")
                scale_factor = self.video_height / clip.h
                new_width = int(clip.w * scale_factor)
                
                # Resize maintaining aspect ratio
                scaled_clip = clip.resized((new_width, self.video_height))
                
                # Create composite with black background using CompositeVideoClip
                background = self._create_black_background(self.video_width, self.video_height)
                x_offset = (self.video_width - new_width) // 2
                
                final_clip = CompositeVideoClip([
                    background,
                    scaled_clip.with_position((x_offset, 0))
                ], size=(self.video_width, self.video_height))
                
                resized_clip = final_clip
                log.info(f"   Scaled to: {new_width}x{self.video_height}, centered with offset {x_offset}")
            else:
                # Image is taller than target - scale to fit width, add black bars on top/bottom
                log.info("   Image taller than target - fitting width with top/bottom bars")
                scale_factor = self.video_width / clip.w
                new_height = int(clip.h * scale_factor)
                
                # Resize maintaining aspect ratio
                scaled_clip = clip.resized((self.video_width, new_height))
                
                # Create composite with black background using CompositeVideoClip
                background = self._create_black_background(self.video_width, self.video_height)
                y_offset = (self.video_height - new_height) // 2
                
                final_clip = CompositeVideoClip([
                    background,
                    scaled_clip.with_position((0, y_offset))
                ], size=(self.video_width, self.video_height))
                
                resized_clip = final_clip
                log.info(f"   Scaled to: {self.video_width}x{new_height}, centered with offset {y_offset}")
            
            log.info(f"✅ Image processing complete")
            return resized_clip
            
        except Exception as e:
            log.error(f"❌ Error processing image: {e}")
            # Ultimate fallback - simple resize (stretching)
            log.warning("   Using fallback: simple resize (may stretch)")
            try:
                clip_processed = clip
                if grayscale:
                    # Simple grayscale fallback
                    log.info("   Applying fallback grayscale...")
                    try:
                        def simple_gray(get_frame, t):
                            frame = get_frame(t)
                            gray = frame.mean(axis=2, keepdims=True)
                            return np.repeat(gray, 3, axis=2).astype(np.uint8)
                        clip_processed = clip.transform(simple_gray)
                    except:
                        pass
                return clip_processed.resized((self.video_width, self.video_height))
            except:
                return clip
    
    def _create_black_background(self, width: int, height: int) -> ImageClip:
        """Create a black background clip for letterboxing/pillarboxing."""
        background_array = np.zeros((height, width, 3), dtype=np.uint8)
        return ImageClip(background_array, duration=1)  # Duration will be set later
    
    def _create_solid_background(self, duration: float) -> ImageClip:
        """Create a solid color background clip."""
        # Create a gradient background
        background_array = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        
        # Create vertical gradient
        for y in range(self.video_height):
            color_value = int(20 + (y / self.video_height) * 60)
            background_array[y, :] = [color_value, 20, 80]
        
        return ImageClip(background_array, duration=duration)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_shortform_content(
        self,
        transcription_data: Dict[str, Any],
        main_video_path: str,
        project_name: str
    ) -> List[Dict[str, Any]]:
        """Use DeepSeek to analyze transcript and generate shortform content clips."""
        try:
            # Analyze transcript for shortform opportunities
            shortform_segments = await self._analyze_for_shortform(transcription_data)
            
            if not shortform_segments:
                log.info("No suitable shortform segments identified")
                return []
            
            # Create clips for each shortform segment
            shorts = []
            for idx, segment in enumerate(shortform_segments):
                try:
                    short_data = await self._create_shortform_clip(
                        main_video_path,
                        segment,
                        f"{project_name}_short_{idx+1}",
                        idx + 1
                    )
                    shorts.append(short_data)
                except Exception as e:
                    log.error(f"Error creating short {idx+1}: {str(e)}")
                    continue
            
            return shorts
            
        except Exception as e:
            log.error(f"Error generating shortform content: {str(e)}")
            return []
    
    async def _analyze_for_shortform(self, transcription_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze transcript using DeepSeek to identify good shortform segments."""
        try:
            # Prepare the transcript for analysis
            segments_text = []
            for idx, segment in enumerate(transcription_data["segments"]):
                segments_text.append(f"[{segment['start']:.1f}s-{segment['end']:.1f}s] {segment['text']}")
            
            full_transcript = "\n".join(segments_text)
            
            prompt = f"""Analyze this transcript and identify segments that would make compelling short-form content (15-60 seconds each).

Transcript:
{full_transcript}

Requirements:
1. Each segment should tell a complete micro-story or convey a key insight
2. Segments should be 15-60 seconds long
3. Look for moments with strong hooks, surprising facts, or emotional impact
4. Prefer segments that can stand alone without context
5. Identify 3-5 of the best segments maximum

For each identified segment, provide:
- start_time: Start time in seconds
- end_time: End time in seconds  
- title: Compelling title for the short
- description: Brief description of why this makes good short content
- hook: The specific hook or attention-grabbing element
- tags: Relevant hashtags/keywords

Respond with a JSON array of segments."""
            
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at identifying viral short-form content opportunities. Analyze transcripts and find the most engaging segments that would perform well as standalone shorts."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate and clean the segments
            segments = result.get("segments", [])
            validated_segments = []
            
            for segment in segments:
                if all(key in segment for key in ["start_time", "end_time", "title"]):
                    # Ensure timing is valid
                    start_time = float(segment["start_time"])
                    end_time = float(segment["end_time"])
                    duration = end_time - start_time
                    
                    if 15 <= duration <= 60:  # Valid short duration
                        validated_segments.append({
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": duration,
                            "title": segment["title"],
                            "description": segment.get("description", ""),
                            "hook": segment.get("hook", ""),
                            "tags": segment.get("tags", [])
                        })
            
            log.info(f"Identified {len(validated_segments)} shortform segments")
            return validated_segments
            
        except Exception as e:
            log.error(f"Error analyzing for shortform content: {str(e)}")
            return []
    
    async def _create_shortform_clip(
        self,
        main_video_path: str,
        segment_data: Dict[str, Any],
        clip_name: str,
        clip_number: int
    ) -> Dict[str, Any]:
        """Create a shortform clip from the main video."""
        try:
            start_time = segment_data["start_time"]
            end_time = segment_data["end_time"]
            title = segment_data["title"]
            
            # Create safe filename
            safe_title = self._create_safe_filename(title)
            output_path = settings.workflow_paths["video_processing"] / f"{clip_name}_{safe_title}.mp4"
            
            # Use ffmpeg to extract the clip with GPU acceleration if available
            input_stream = ffmpeg.input(main_video_path, ss=start_time, t=end_time-start_time)
            
            if self.device == "cuda":
                log.info("🚀 Using GPU-accelerated clip extraction")
                # Use GPU acceleration for clip extraction if we need to re-encode
                output_stream = ffmpeg.output(
                    input_stream,
                    str(output_path),
                    vcodec='h264_nvenc',  # GPU encoder
                    acodec='copy',        # Audio copy (no re-encoding needed)
                    avoid_negative_ts='make_zero',
                    **{
                        'hwaccel': 'cuda',
                        'preset': 'fast',
                        'rc': 'vbr',
                        'cq': '23'
                    }
                )
            else:
                log.info("🖥️ Using CPU-based clip extraction")
                output_stream = ffmpeg.output(
                    input_stream,
                    str(output_path),
                    vcodec='copy',  # Copy without re-encoding (fastest)
                    acodec='copy',
                    avoid_negative_ts='make_zero'
                )
            
            # Run ffmpeg
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            
            # Create metadata
            metadata = {
                "title": title,
                "description": segment_data.get("description", ""),
                "hook": segment_data.get("hook", ""),
                "tags": segment_data.get("tags", []),
                "duration": segment_data["duration"],
                "start_time": start_time,
                "end_time": end_time,
                "clip_number": clip_number,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Save metadata as JSON
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            log.info(f"Created shortform clip: {output_path}")
            
            return {
                "clip_path": str(output_path),
                "metadata_path": str(metadata_path),
                "metadata": metadata
            }
            
        except Exception as e:
            log.error(f"Error creating shortform clip: {str(e)}")
            raise
    
    def _create_safe_filename(self, title: str) -> str:
        """Create a safe filename from a title."""
        import re
        
        # Remove or replace unsafe characters
        safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
        safe_title = re.sub(r'\s+', '_', safe_title)
        safe_title = safe_title[:50]  # Limit length
        
        return safe_title
    
    def _generate_project_name(self, audio_file_path: Path) -> str:
        """Generate a unique project name from audio filename."""
        base_name = audio_file_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"audio_{base_name}_{timestamp}"
    
    def _store_project_data(self, project_name: str, result_data: Dict[str, Any]):
        """Store project data in database."""
        try:
            with self.db_manager as session:
                # Create main project entry
                project = Project(
                    name=project_name,
                    title=f"Audio Video: {project_name}",
                    description=f"Generated from audio file",
                    current_stage="video_processing",
                    status="completed",
                    video_path=result_data["main_video"]["video_path"],
                    duration=result_data["main_video"]["duration"],
                    script=result_data["transcription"]["text"],
                    script_metadata={
                        "type": "audio_transcription",
                        "language": result_data["transcription"]["language"],
                        "keywords": result_data["keywords"],
                        "segments_count": len(result_data["transcription"]["segments"]),
                        "shorts_count": result_data.get("shorts_count", 0)
                    }
                )
                
                session.add(project)
                session.flush()  # Get the project ID
                
                # Create audio project entry
                audio_project = AudioProject(
                    project_id=project.id,
                    audio_file_path=result_data["main_video"]["audio_source"],
                    audio_duration=int(result_data["main_video"]["duration"]),
                    transcription_text=result_data["transcription"]["text"],
                    transcription_language=result_data["transcription"]["language"],
                    word_timestamps=result_data["transcription"]["segments"],
                    keywords_extracted=result_data["keywords"],
                    shortform_clips_count=result_data.get("shorts_count", 0),
                    transcription_status="completed",
                    video_generation_status="completed",
                    shortform_generation_status="completed" if result_data.get("shorts") else "skipped"
                )
                
                session.add(audio_project)
                
                # Create shortform clip entries if any exist
                if result_data.get("shorts"):
                    for short_data in result_data["shorts"]:
                        shortform_clip = ShortformClip(
                            project_id=project.id,
                            clip_number=short_data["metadata"]["clip_number"],
                            title=short_data["metadata"]["title"],
                            description=short_data["metadata"]["description"],
                            start_time=int(short_data["metadata"]["start_time"]),
                            end_time=int(short_data["metadata"]["end_time"]),
                            duration=int(short_data["metadata"]["duration"]),
                            video_path=short_data["clip_path"],
                            metadata_path=short_data["metadata_path"],
                            hook=short_data["metadata"]["hook"],
                            tags=short_data["metadata"]["tags"],
                            shortform_metadata=short_data["metadata"],
                            status="pending"
                        )
                        session.add(shortform_clip)
                
                session.commit()
                
                log.info(f"Stored complete project data for: {project_name}")
                log.info(f"- Main video: {result_data['main_video']['video_path']}")
                log.info(f"- Shortform clips: {result_data.get('shorts_count', 0)}")
                
        except Exception as e:
            log.error(f"Error storing project data: {str(e)}")
            raise


# Additional utilities for batch processing

class AudioBatchProcessor:
    """Process multiple audio files in batch."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.processor = AudioVideoProcessor(db_manager)
    
    async def process_ingestion_directory(
        self,
        generate_shorts: bool = True,
        audio_extensions: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Process all audio files in the ingestion directory."""
        if audio_extensions is None:
            audio_extensions = ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg']
        
        ingestion_path = settings.workflow_paths["ingestion"]
        audio_files = []
        
        # Find all audio files
        for ext in audio_extensions:
            audio_files.extend(ingestion_path.glob(f"*{ext}"))
        
        log.info(f"Found {len(audio_files)} audio files in ingestion directory")
        
        results = []
        for audio_file in audio_files:
            try:
                log.info(f"Processing: {audio_file.name}")
                result = await self.processor.process_audio_file(
                    audio_file,
                    generate_shorts=generate_shorts
                )
                results.append(result)
                
                # Move processed file to avoid reprocessing
                processed_dir = ingestion_path / "processed"
                processed_dir.mkdir(exist_ok=True)
                audio_file.rename(processed_dir / audio_file.name)
                
            except Exception as e:
                log.error(f"Failed to process {audio_file}: {str(e)}")
                results.append({
                    "status": "failed",
                    "file": str(audio_file),
                    "error": str(e)
                })
        
        return results
