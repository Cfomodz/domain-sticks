"""
Video Clip Processing Module for analyzing existing videos and creating short clips.

This module takes existing video files, analyzes their audio track using Whisper,
and automatically creates short-form clips based on AI analysis of the content.
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
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    import numpy as np
    VIDEO_CLIP_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Video clip processing dependencies not available: {e}")
    VIDEO_CLIP_PROCESSING_AVAILABLE = False
    # Create dummy classes to prevent NameError
    class VideoFileClip: pass
    class AudioFileClip: pass
    whisper = None
    ffmpeg = None
    np = None

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import log
from src.config.settings import settings
from src.models.database import DatabaseManager, Project, ShortformClip, VideoProject


class VideoClipProcessor:
    """Process existing video files to create short-form clips based on audio analysis."""
    
    def __init__(self, db_manager: DatabaseManager):
        if not VIDEO_CLIP_PROCESSING_AVAILABLE:
            raise ImportError("Video clip processing dependencies not available. Please install whisper, moviepy, ffmpeg-python, and numpy.")
        
        self.db_manager = db_manager
        
        # Initialize Whisper model with GPU support
        try:
            # Check for CUDA availability
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"🖥️ Video Clip Processor - Device detected: {self.device}")
            if self.device == "cuda":
                log.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
                log.info(f"🚀 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Load model with numerical stability options for GPU
            if self.device == "cuda":
                self.whisper_model = whisper.load_model("base", device=self.device)
                # Set model to eval mode and disable autocast for stability
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
        log.info("🖥️ Video Clip Processor GPU Capabilities:")
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
        
        if getattr(self, 'device', 'cpu') == 'cpu':
            log.info("   💡 Install CUDA-enabled PyTorch for faster video processing")
        else:
            log.info("   💡 GPU acceleration enabled - video processing will be faster!")
    
    async def process_video_file(
        self,
        video_file_path: Path,
        project_name: Optional[str] = None,
        generate_shorts: bool = True,
        max_short_duration: int = 60,
        min_short_duration: int = 15,
        convert_to_vertical: bool = True,
        add_subtitles: bool = False
    ) -> Dict[str, Any]:
        """
        Process a video file to create short-form clips based on audio analysis.
        
        Args:
            video_file_path: Path to the video file
            project_name: Optional project name (generated if not provided)
            generate_shorts: Whether to generate shortform content clips
            max_short_duration: Maximum duration for shorts (seconds)
            min_short_duration: Minimum duration for shorts (seconds)
            convert_to_vertical: Whether to convert clips to vertical format (9:16)
            add_subtitles: Whether to add subtitles to clips
            
        Returns:
            Dictionary with processing results including shorts
        """
        try:
            if not video_file_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_file_path}")
            
            # Generate project name if not provided
            if not project_name:
                project_name = self._generate_project_name(video_file_path)
            
            log.info(f"🎬 Processing video file: {video_file_path}")
            log.info(f"📽️ Project name: {project_name}")
            log.info(f"🎯 Generate shorts: {generate_shorts}")
            log.info(f"⏱️ Short duration range: {min_short_duration}-{max_short_duration}s")
            log.info(f"📱 Convert to vertical: {convert_to_vertical}")
            log.info(f"📝 Add subtitles: {add_subtitles}")
            
            # Step 1: Get video information
            log.info("Step 1: Analyzing video file...")
            video_info = await self._get_video_info(video_file_path)
            
            # Step 2: Extract and transcribe audio
            log.info("Step 2: Extracting and transcribing audio...")
            transcription_data = await self._extract_and_transcribe_audio(video_file_path)
            
            # Step 3: Extract keywords and generate metadata
            log.info("Step 3: Extracting keywords and metadata...")
            keywords = self._extract_keywords(transcription_data["text"])
            
            result = {
                "status": "success",
                "project_name": project_name,
                "original_video": {
                    "path": str(video_file_path),
                    "duration": video_info["duration"],
                    "width": video_info["width"],
                    "height": video_info["height"],
                    "fps": video_info["fps"]
                },
                "transcription": transcription_data,
                "keywords": keywords,
                "shorts": []
            }
            
            # Step 4: Generate shortform content if requested
            if generate_shorts:
                log.info("Step 4: Analyzing content for shortform opportunities...")
                shorts_data = await self._generate_shortform_clips(
                    video_file_path,
                    transcription_data,
                    project_name,
                    max_short_duration=max_short_duration,
                    min_short_duration=min_short_duration,
                    convert_to_vertical=convert_to_vertical,
                    add_subtitles=add_subtitles
                )
                result["shorts"] = shorts_data
                result["shorts_count"] = len(shorts_data)
            
            # Step 5: Store project in database
            log.info("Step 5: Storing project data...")
            self._store_project_data(project_name, result, video_file_path)
            
            log.info(f"✅ Video processing completed successfully!")
            log.info(f"📊 Generated {len(result.get('shorts', []))} short clips")
            
            return result
            
        except Exception as e:
            log.error(f"❌ Error processing video file: {str(e)}")
            raise
    
    async def _get_video_info(self, video_file_path: Path) -> Dict[str, Any]:
        """Get video file information using ffprobe."""
        try:
            probe = ffmpeg.probe(str(video_file_path))
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            
            if not video_stream:
                raise ValueError("No video stream found in file")
            
            info = {
                "duration": float(probe['format']['duration']),
                "width": int(video_stream['width']),
                "height": int(video_stream['height']),
                "fps": eval(video_stream['r_frame_rate']),  # Convert fraction to float
                "codec": video_stream['codec_name'],
                "bitrate": int(probe['format'].get('bit_rate', 0))
            }
            
            log.info(f"📹 Video info: {info['width']}x{info['height']} @ {info['fps']:.1f}fps, {info['duration']:.1f}s")
            
            return info
            
        except Exception as e:
            log.error(f"❌ Error getting video info: {e}")
            raise
    
    async def _extract_and_transcribe_audio(self, video_file_path: Path) -> Dict[str, Any]:
        """Extract audio from video and transcribe using Whisper."""
        if not self.whisper_model:
            raise ValueError("Whisper model not available")
        
        try:
            # Extract audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            try:
                log.info(f"🎵 Extracting audio from video...")
                # Use ffmpeg to extract audio
                input_stream = ffmpeg.input(str(video_file_path))
                output_stream = ffmpeg.output(
                    input_stream['a'],  # Select audio stream
                    temp_audio_path,
                    acodec='pcm_s16le',  # PCM format for Whisper
                    ac=1,  # Mono
                    ar=16000  # 16kHz sample rate
                )
                ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
                
                log.info(f"✅ Audio extracted to temporary file")
                
                # Transcribe with word-level timestamps and numerical stability
                log.info(f"🎙️ Transcribing audio using Whisper...")
                
                # Add numerical stability options for GPU inference
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
                            result = self.whisper_model.transcribe(temp_audio_path, **transcribe_options)
                        
                        # Check for NaN values in result (basic validation)
                        if not result.get("text") or len(result.get("segments", [])) == 0:
                            raise ValueError("Empty transcription result - possible GPU numerical issue")
                            
                    except Exception as gpu_error:
                        log.warning(f"⚠️ GPU transcription failed: {gpu_error}")
                        log.info("🔄 Falling back to CPU transcription...")
                        
                        # Fallback to CPU
                        cpu_model = whisper.load_model("base", device="cpu")
                        result = cpu_model.transcribe(temp_audio_path, **transcribe_options)
                        del cpu_model  # Clean up memory
                else:
                    result = self.whisper_model.transcribe(temp_audio_path, **transcribe_options)
                
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
                
                transcription_data = {
                    "text": result["text"],
                    "language": result.get("language", "en"),
                    "segments": segments,
                    "duration": segments[-1]["end"] if segments else 0
                }
                
                log.info(f"✅ Transcription completed: {len(segments)} segments, {transcription_data['duration']:.1f}s")
                
                # DEBUG: Log transcription result for debugging
                log.info(f"🔍 DEBUG - Full transcription text (first 500 chars): {result['text'][:500]}...")
                log.info(f"🔍 DEBUG - Language detected: {result.get('language', 'unknown')}")
                log.info(f"🔍 DEBUG - Total segments: {len(segments)}")
                log.info(f"🔍 DEBUG - Sample segments:")
                for i, seg in enumerate(segments[:3]):  # Show first 3 segments
                    log.info(f"   Segment {i+1}: [{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['text'][:100]}...")
                
                return transcription_data
                
            finally:
                # Clean up temporary audio file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    
        except Exception as e:
            log.error(f"❌ Error extracting and transcribing audio: {e}")
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
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_shortform_clips(
        self,
        video_file_path: Path,
        transcription_data: Dict[str, Any],
        project_name: str,
        max_short_duration: int = 60,
        min_short_duration: int = 15,
        convert_to_vertical: bool = True,
        add_subtitles: bool = False
    ) -> List[Dict[str, Any]]:
        """Use AI to analyze transcript and generate shortform content clips."""
        try:
            # Analyze transcript for shortform opportunities
            shortform_segments = await self._analyze_for_shortform(
                transcription_data, 
                max_duration=max_short_duration,
                min_duration=min_short_duration
            )
            
            if not shortform_segments:
                log.info("❌ No suitable shortform segments identified")
                return []
            
            log.info(f"✅ Identified {len(shortform_segments)} potential short clips")
            
            # Create clips for each shortform segment
            shorts = []
            for idx, segment in enumerate(shortform_segments):
                try:
                    short_data = await self._create_video_clip(
                        video_file_path,
                        segment,
                        f"{project_name}_short_{idx+1}",
                        idx + 1,
                        convert_to_vertical=convert_to_vertical,
                        add_subtitles=add_subtitles
                    )
                    shorts.append(short_data)
                    log.info(f"✅ Created short clip {idx+1}: {segment['title']}")
                except Exception as e:
                    log.error(f"❌ Error creating short {idx+1}: {str(e)}")
                    continue
            
            return shorts
            
        except Exception as e:
            log.error(f"❌ Error generating shortform clips: {str(e)}")
            return []
    
    async def _analyze_for_shortform(
        self, 
        transcription_data: Dict[str, Any],
        max_duration: int = 60,
        min_duration: int = 15
    ) -> List[Dict[str, Any]]:
        """Analyze transcript using AI to identify good shortform segments."""
        try:
            # Prepare the transcript for analysis
            segments_text = []
            for idx, segment in enumerate(transcription_data["segments"]):
                segments_text.append(f"[{segment['start']:.1f}s-{segment['end']:.1f}s] {segment['text']}")
            
            full_transcript = "\n".join(segments_text)
            
            # DEBUG: Log transcript preparation
            log.info(f"🔍 DEBUG - Prepared transcript for LLM analysis:")
            log.info(f"   📝 Total segments for analysis: {len(transcription_data['segments'])}")
            log.info(f"   📏 Total transcript length: {len(full_transcript)} characters")
            log.info(f"   ⏱️ Duration range requested: {min_duration}-{max_duration} seconds")
            log.info(f"   📝 Sample transcript (first 300 chars): {full_transcript[:300]}...")
            
            prompt = f"""Analyze this video transcript and identify segments that would make compelling short-form content. 

**CRITICAL DURATION REQUIREMENT: Each segment MUST be {min_duration}-{max_duration} seconds long. This is MANDATORY.**

Transcript:
{full_transcript}

Requirements:
1. **DURATION IS CRITICAL**: Each segment MUST be AT LEAST {min_duration} seconds and NO MORE than {max_duration} seconds
2. Look for longer sequences that tell a complete story or explain a concept thoroughly
3. Find segments with strong hooks, surprising facts, emotional impact, or actionable advice  
4. Each segment should be substantive - NOT just single sentences or brief moments
5. Combine multiple related sentences/ideas to reach the minimum {min_duration} second duration
6. Identify 3-7 of the best segments maximum
7. Prefer segments that can stand alone without context

**DURATION EXAMPLES:**
- ❌ BAD: 73.5-77.6 (only 4.1 seconds - TOO SHORT)
- ❌ BAD: 93.7-100.4 (only 6.7 seconds - TOO SHORT)  
- ✅ GOOD: 73.5-105.0 (31.5 seconds - PERFECT LENGTH)
- ✅ GOOD: 120.0-155.0 (35 seconds - PERFECT LENGTH)

**STRATEGY**: Find an engaging starting point, then extend the end time to include enough content to reach {min_duration}-{max_duration} seconds while maintaining narrative coherence.

For each identified segment, provide:
- start_time: Start time in seconds
- end_time: End time in seconds (MUST create {min_duration}-{max_duration} second duration)
- title: Compelling title for the short (under 60 characters)
- description: Brief description of why this makes good short content
- hook: The specific hook or attention-grabbing element
- tags: Relevant hashtags/keywords (without # symbol)

Respond with a JSON array of segments."""
            
            # DEBUG: Log the complete prompt being sent to LLM
            log.info(f"🔍 DEBUG - Sending prompt to LLM:")
            log.info(f"   🤖 Model: deepseek-chat")
            log.info(f"   📏 Prompt length: {len(prompt)} characters")
            log.info(f"   📝 Full prompt:\n{prompt}")
            
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at identifying viral short-form content opportunities. Analyze video transcripts and find the most engaging segments that would perform well as standalone shorts."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # DEBUG: Log the raw LLM response
            raw_response = response.choices[0].message.content
            log.info(f"🔍 DEBUG - Raw LLM response:")
            log.info(f"   📏 Response length: {len(raw_response)} characters")
            log.info(f"   🔧 Response format: {response.choices[0].message.content[:200]}...")
            log.info(f"   📝 Full raw response:\n{raw_response}")
            
            try:
                result = json.loads(raw_response)
                log.info(f"🔍 DEBUG - Successfully parsed JSON response")
            except json.JSONDecodeError as e:
                log.error(f"❌ DEBUG - Failed to parse JSON: {e}")
                log.error(f"   Raw content: {raw_response}")
                return []
            
            # Validate and clean the segments
            segments = result.get("segments", [])
            
            # DEBUG: Log segment validation process
            log.info(f"🔍 DEBUG - Validating segments from LLM response:")
            log.info(f"   📊 Raw segments received: {len(segments)}")
            log.info(f"   📝 Raw segments structure: {segments}")
            
            validated_segments = []
            
            for i, segment in enumerate(segments):
                log.info(f"🔍 DEBUG - Validating segment {i+1}: {segment}")
                
                if all(key in segment for key in ["start_time", "end_time", "title"]):
                    try:
                        # Ensure timing is valid
                        start_time = float(segment["start_time"])
                        end_time = float(segment["end_time"])
                        duration = end_time - start_time
                        
                        log.info(f"   ⏱️ Timing: {start_time:.1f}s-{end_time:.1f}s ({duration:.1f}s)")
                        
                        if min_duration <= duration <= max_duration:  # Valid short duration
                            validated_segment = {
                                "start_time": start_time,
                                "end_time": end_time,
                                "duration": duration,
                                "title": segment["title"],
                                "description": segment.get("description", ""),
                                "hook": segment.get("hook", ""),
                                "tags": segment.get("tags", [])
                            }
                            validated_segments.append(validated_segment)
                            log.info(f"   ✅ VALID: {segment['title']}")
                        else:
                            log.info(f"   ❌ INVALID duration ({duration:.1f}s): {segment['title']}")
                    except (ValueError, TypeError) as e:
                        log.info(f"   ❌ INVALID timing data: {e}")
                else:
                    missing_keys = [key for key in ["start_time", "end_time", "title"] if key not in segment]
                    log.info(f"   ❌ MISSING required keys: {missing_keys}")
            
            log.info(f"📊 Identified {len(validated_segments)} valid shortform segments")
            
            # FALLBACK: For existing videos, if we get no segments, create a basic one
            if len(validated_segments) == 0 and len(transcription_data.get("segments", [])) > 0:
                log.warning(f"⚠️ No valid segments identified by LLM - implementing fallback for existing video")
                
                # Find a good 30-45 second segment from the middle of the content
                total_duration = transcription_data.get("duration", 0)
                if total_duration > 60:  # Only if we have enough content
                    # Take a segment from around 25% into the content
                    fallback_start = total_duration * 0.25
                    fallback_end = min(fallback_start + 30, total_duration - 5)  # 30s clip, leave 5s buffer
                    
                    # Find actual transcript segments that cover this time range
                    relevant_text = ""
                    for seg in transcription_data["segments"]:
                        if seg["start"] >= fallback_start and seg["end"] <= fallback_end:
                            relevant_text += seg["text"] + " "
                    
                    if len(relevant_text.strip()) > 20:  # Make sure we have some content
                        fallback_segment = {
                            "start_time": fallback_start,
                            "end_time": fallback_end,
                            "duration": fallback_end - fallback_start,
                            "title": f"Clip from {transcription_data.get('language', 'Video').title()} Content",
                            "description": f"Auto-generated clip from existing video content",
                            "hook": relevant_text[:50] + "..." if len(relevant_text) > 50 else relevant_text,
                            "tags": ["video", "content", "clip"]
                        }
                        validated_segments.append(fallback_segment)
                        log.info(f"✅ FALLBACK: Created basic segment ({fallback_start:.1f}s-{fallback_end:.1f}s)")
            
            log.info(f"📊 Final result: {len(validated_segments)} segments ready for processing")
            return validated_segments
            
        except Exception as e:
            log.error(f"❌ Error analyzing for shortform content: {str(e)}")
            return []
    
    async def _create_video_clip(
        self,
        video_file_path: Path,
        segment_data: Dict[str, Any],
        clip_name: str,
        clip_number: int,
        convert_to_vertical: bool = True,
        add_subtitles: bool = False
    ) -> Dict[str, Any]:
        """Create a shortform clip from the main video."""
        try:
            start_time = segment_data["start_time"]
            end_time = segment_data["end_time"]
            title = segment_data["title"]
            
            # Create safe filename
            safe_title = self._create_safe_filename(title)
            output_path = settings.workflow_paths["video_processing"] / f"{clip_name}_{safe_title}.mp4"
            
            log.info(f"🎬 Creating clip: {title}")
            log.info(f"   ⏱️ Time range: {start_time:.1f}s - {end_time:.1f}s")
            log.info(f"   📱 Convert to vertical: {convert_to_vertical}")
            log.info(f"   📝 Add subtitles: {add_subtitles}")
            
            # Use ffmpeg to extract and process the clip
            if self.device == "cuda":
                input_stream = ffmpeg.input(str(video_file_path), ss=start_time, t=end_time-start_time, hwaccel='cuda')
            else:
                input_stream = ffmpeg.input(str(video_file_path), ss=start_time, t=end_time-start_time)
            
            if convert_to_vertical:
                # Convert to vertical format (9:16)
                log.info(f"   📱 Converting to vertical format...")
                if self.device == "cuda":
                    log.info("   🚀 Using GPU-accelerated conversion")
                    # GPU-accelerated vertical conversion
                    output_stream = ffmpeg.output(
                        input_stream,
                        str(output_path),
                        vf=f"scale=w={self.video_width}:h={self.video_height}:force_original_aspect_ratio=decrease,pad={self.video_width}:{self.video_height}:(ow-iw)/2:(oh-ih)/2:black",
                        vcodec='h264_nvenc',
                        acodec='aac',
                        preset='fast',
                        rc='vbr',
                        cq='23'
                    )
                else:
                    log.info("   🖥️ Using CPU-based conversion")
                    # CPU-based vertical conversion
                    output_stream = ffmpeg.output(
                        input_stream,
                        str(output_path),
                        vf=f"scale=w={self.video_width}:h={self.video_height}:force_original_aspect_ratio=decrease,pad={self.video_width}:{self.video_height}:(ow-iw)/2:(oh-ih)/2:black",
                        vcodec='libx264',
                        acodec='aac',
                        crf=23,
                        preset='medium'
                    )
            else:
                # Keep original format, just extract clip
                if self.device == "cuda":
                    log.info("   🚀 Using GPU-accelerated extraction")
                    output_stream = ffmpeg.output(
                        input_stream,
                        str(output_path),
                        vcodec='h264_nvenc',
                        acodec='aac',
                        preset='fast',
                        rc='vbr',
                        cq='23'
                    )
                else:
                    log.info("   🖥️ Using CPU-based extraction")
                    output_stream = ffmpeg.output(
                        input_stream,
                        str(output_path),
                        vcodec='copy',  # Copy without re-encoding (fastest)
                        acodec='copy'
                    )
            
            # Run ffmpeg with proper error capture
            try:
                ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            except ffmpeg.Error as e:
                log.error(f"❌ FFmpeg command failed!")
                log.error(f"   📝 Command: {' '.join(ffmpeg.compile(output_stream))}")
                if e.stderr:
                    log.error(f"   🔴 stderr: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}")
                if e.stdout:
                    log.error(f"   📤 stdout: {e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout}")
                raise
            
            # TODO: Add subtitle generation if requested
            if add_subtitles:
                log.info("   📝 Subtitle generation not yet implemented")
            
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
                "converted_to_vertical": convert_to_vertical,
                "has_subtitles": add_subtitles,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Save metadata as JSON
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            log.info(f"✅ Created video clip: {output_path}")
            
            return {
                "clip_path": str(output_path),
                "metadata_path": str(metadata_path),
                "metadata": metadata
            }
            
        except Exception as e:
            log.error(f"❌ Error creating video clip: {str(e)}")
            raise
    
    def _create_safe_filename(self, title: str) -> str:
        """Create a safe filename from a title."""
        import re
        
        # Remove or replace unsafe characters
        safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
        safe_title = re.sub(r'\s+', '_', safe_title)
        safe_title = safe_title[:50]  # Limit length
        
        return safe_title
    
    def _generate_project_name(self, video_file_path: Path) -> str:
        """Generate a unique project name from video filename."""
        base_name = video_file_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"video_{base_name}_{timestamp}"
    
    def _store_project_data(self, project_name: str, result_data: Dict[str, Any], video_file_path: Path):
        """Store project data in database."""
        try:
            with self.db_manager as session:
                # Create main project entry
                project = Project(
                    name=project_name,
                    title=f"Video Clips: {project_name}",
                    description=f"Generated from video file: {video_file_path.name}",
                    current_stage="video_processing",
                    status="completed",
                    video_path=str(video_file_path),  # Original video path
                    duration=result_data["original_video"]["duration"],
                    script=result_data["transcription"]["text"],
                    script_metadata={
                        "type": "video_transcription",
                        "language": result_data["transcription"]["language"],
                        "keywords": result_data["keywords"],
                        "segments_count": len(result_data["transcription"]["segments"]),
                        "shorts_count": result_data.get("shorts_count", 0),
                        "original_video_info": result_data["original_video"]
                    }
                )
                
                session.add(project)
                session.flush()  # Get the project ID
                
                # Create video project entry
                video_project = VideoProject(
                    project_id=project.id,
                    video_file_path=str(video_file_path),
                    video_duration=int(result_data["original_video"]["duration"]),
                    video_width=result_data["original_video"]["width"],
                    video_height=result_data["original_video"]["height"],
                    video_fps=result_data["original_video"]["fps"],
                    transcription_text=result_data["transcription"]["text"],
                    transcription_language=result_data["transcription"]["language"],
                    word_timestamps=result_data["transcription"]["segments"],
                    keywords_extracted=result_data["keywords"],
                    shortform_clips_count=result_data.get("shorts_count", 0),
                    transcription_status="completed",
                    clip_generation_status="completed" if result_data.get("shorts") else "skipped"
                )
                
                session.add(video_project)
                
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
                
                log.info(f"✅ Stored complete project data for: {project_name}")
                log.info(f"   📹 Original video: {video_file_path}")
                log.info(f"   📱 Short clips: {result_data.get('shorts_count', 0)}")
                
        except Exception as e:
            log.error(f"❌ Error storing project data: {str(e)}")
            raise


# Additional utilities for batch processing

class VideoBatchProcessor:
    """Process multiple video files in batch."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.processor = VideoClipProcessor(db_manager)
    
    async def process_ingestion_directory(
        self,
        generate_shorts: bool = True,
        video_extensions: List[str] = None,
        convert_to_vertical: bool = True,
        add_subtitles: bool = False
    ) -> List[Dict[str, Any]]:
        """Process all video files in the ingestion directory."""
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.wmv']
        
        ingestion_path = settings.workflow_paths["ingestion"]
        video_files = []
        
        # Find all video files
        for ext in video_extensions:
            video_files.extend(ingestion_path.glob(f"*{ext}"))
        
        log.info(f"🎬 Found {len(video_files)} video files in ingestion directory")
        
        results = []
        for video_file in video_files:
            try:
                log.info(f"🎬 Processing: {video_file.name}")
                result = await self.processor.process_video_file(
                    video_file,
                    generate_shorts=generate_shorts,
                    convert_to_vertical=convert_to_vertical,
                    add_subtitles=add_subtitles
                )
                results.append(result)
                
                # Move processed file to avoid reprocessing
                processed_dir = ingestion_path / "processed"
                processed_dir.mkdir(exist_ok=True)
                video_file.rename(processed_dir / video_file.name)
                
            except Exception as e:
                log.error(f"❌ Failed to process {video_file}: {str(e)}")
                results.append({
                    "status": "failed",
                    "file": str(video_file),
                    "error": str(e)
                })
        
        return results
