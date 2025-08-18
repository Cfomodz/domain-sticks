"""
AI Script Generation Module using DeepSeek v3.
"""
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.logger import log
from src.config.settings import settings
from src.models.database import DatabaseManager, Project, ProjectSegment


class ScriptGenerator:
    """Generate video scripts using DeepSeek v3 AI."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_api_base
        )
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_script(
        self, 
        metadata: Dict[str, Any], 
        project_name: str,
        focus: str = "auto"
    ) -> Dict[str, Any]:
        """
        Generate a 45-second video script based on metadata.
        
        Args:
            metadata: Scraped metadata from the URL
            project_name: Name of the project
            focus: Focus of the story (auto, creator, subject, work)
            
        Returns:
            Dictionary containing script and related information
        """
        # Determine the focus of the story
        if focus == "auto":
            focus = self._determine_focus(metadata)
        
        # Generate the main script
        script_data = await self._generate_main_script(metadata, focus)
        
        # Check if content needs to be split into segments
        if script_data.get("needs_segmentation", False):
            segments = await self._generate_segments(metadata, script_data)
            script_data["segments"] = segments
        
        # Store in database
        with self.db_manager as session:
            project = session.query(Project).filter_by(name=project_name).first()
            if project:
                project.script = script_data["script"]
                project.script_metadata = {
                    "focus": focus,
                    "keywords": script_data["keywords"],
                    "tone": script_data["tone"],
                    "hook": script_data["hook"],
                    "call_to_action": script_data["call_to_action"]
                }
                
                # Add segments if any
                if "segments" in script_data:
                    for idx, segment in enumerate(script_data["segments"]):
                        project_segment = ProjectSegment(
                            project_id=project.id,
                            segment_number=idx + 1,
                            script=segment["script"],
                            start_time=segment.get("start_time", 0),
                            end_time=segment.get("end_time", 45)
                        )
                        session.add(project_segment)
                
                session.commit()
        
        return script_data
    
    def _determine_focus(self, metadata: Dict[str, Any]) -> str:
        """Determine the focus of the story based on metadata."""
        page_type = metadata.get("page_type", "website")
        keywords = metadata.get("keywords", [])
        
        # Check for person-related keywords
        person_indicators = ["biography", "artist", "author", "creator", "founder", "inventor"]
        if any(indicator in " ".join(keywords).lower() for indicator in person_indicators):
            return "creator"
        
        # Check for work-related content
        if page_type in ["article", "video", "gallery"]:
            return "work"
        
        # Default to subject matter
        return "subject"
    
    async def _generate_main_script(
        self, 
        metadata: Dict[str, Any], 
        focus: str
    ) -> Dict[str, Any]:
        """Generate the main script using DeepSeek v3."""
        
        # Prepare the prompt
        prompt = self._create_script_prompt(metadata, focus)
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # Validate and enhance the script
            validated_result = self._validate_script(result)
            
            return validated_result
            
        except Exception as e:
            log.error(f"Error generating script: {str(e)}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for script generation."""
        return """You are an expert storyteller and video script writer specializing in creating 
engaging 45-second vertical video scripts. Your scripts should:

1. Hook viewers in the first 3 seconds
2. Tell a complete, compelling story
3. Use simple, conversational language
4. Include visual cues for what should be shown
5. End with a clear call-to-action or thought-provoking conclusion
6. Be exactly 45 seconds when read at a normal pace (approximately 110-120 words)

You must respond with a JSON object containing:
{
    "script": "The complete script text with visual cues in brackets",
    "hook": "The opening hook (first 3 seconds)",
    "keywords": ["list", "of", "relevant", "keywords"],
    "tone": "The tone of the script (educational, inspiring, mysterious, etc.)",
    "visual_cues": ["list", "of", "required", "visuals"],
    "call_to_action": "The ending call-to-action",
    "needs_segmentation": false,
    "estimated_duration": 45
}"""
    
    def _create_script_prompt(self, metadata: Dict[str, Any], focus: str) -> str:
        """Create the prompt for script generation."""
        title = metadata.get("title", "Unknown")
        description = metadata.get("description", "")
        keywords = ", ".join(metadata.get("keywords", [])[:10])
        page_type = metadata.get("page_type", "website")
        
        prompt = f"""Create a 45-second video script about: {title}

Focus: {focus}
Description: {description}
Keywords: {keywords}
Content Type: {page_type}

Requirements:
1. The script should focus on the {focus} aspect
2. Include specific visual cues in [brackets]
3. Make it engaging for a TikTok/YouTube Shorts audience
4. Use public domain references where possible
5. Ensure the content is educational and inspiring

Generate a complete script that tells an interesting story about this topic."""
        
        return prompt
    
    def _validate_script(self, script_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance the generated script."""
        # Ensure all required fields are present
        required_fields = [
            "script", "hook", "keywords", "tone", 
            "visual_cues", "call_to_action", "estimated_duration"
        ]
        
        for field in required_fields:
            if field not in script_data:
                if field == "estimated_duration":
                    script_data[field] = 45
                elif field == "needs_segmentation":
                    script_data[field] = False
                else:
                    script_data[field] = ""
        
        # Validate script length (approximately 110-120 words for 45 seconds)
        word_count = len(script_data["script"].split())
        if word_count < 90 or word_count > 140:
            script_data["needs_adjustment"] = True
            script_data["word_count"] = word_count
        
        # Extract visual cues from script if not provided
        if not script_data.get("visual_cues"):
            script_data["visual_cues"] = self._extract_visual_cues(script_data["script"])
        
        return script_data
    
    def _extract_visual_cues(self, script: str) -> List[str]:
        """Extract visual cues from the script text."""
        import re
        
        # Find all text within brackets
        cues = re.findall(r'\[([^\]]+)\]', script)
        
        # Add default cues if none found
        if not cues:
            cues = ["Title card", "Main subject", "Supporting visuals", "Closing shot"]
        
        return cues
    
    async def _generate_segments(
        self, 
        metadata: Dict[str, Any], 
        main_script_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate multiple segments for longer content."""
        segments = []
        
        # Determine how many segments are needed
        content_length = len(metadata.get("extracted_content", ""))
        num_segments = min(5, max(1, content_length // 1000))  # Rough estimate
        
        for i in range(num_segments):
            segment_prompt = f"""Create segment {i+1} of {num_segments} for a video series about: {metadata.get('title')}

This is part of a series. The main theme is: {main_script_data.get('hook')}

Make this segment focus on a specific aspect while maintaining continuity with the series.
Each segment should be exactly 45 seconds."""
            
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": segment_prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                
                segment_data = json.loads(response.choices[0].message.content)
                segment_data["segment_number"] = i + 1
                segments.append(segment_data)
                
            except Exception as e:
                log.error(f"Error generating segment {i+1}: {str(e)}")
                continue
        
        return segments
    
    async def regenerate_script(
        self, 
        project_name: str, 
        feedback: str
    ) -> Dict[str, Any]:
        """Regenerate a script based on feedback."""
        with self.db_manager as session:
            project = session.query(Project).filter_by(name=project_name).first()
            
            if not project:
                raise ValueError(f"Project {project_name} not found")
            
            # Get original metadata
            metadata = project.source_url.metadata if project.source_url else {}
            
            # Create feedback prompt
            prompt = f"""Original script: {project.script}

User feedback: {feedback}

Please regenerate the script addressing the feedback while maintaining the 45-second format."""
            
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                validated_result = self._validate_script(result)
                
                # Update project
                project.script = validated_result["script"]
                project.script_metadata["regenerated"] = True
                project.script_metadata["feedback"] = feedback
                
                session.commit()
                
                return validated_result
                
            except Exception as e:
                log.error(f"Error regenerating script: {str(e)}")
                raise
