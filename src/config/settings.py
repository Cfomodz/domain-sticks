"""
Application settings and configuration management.
"""
from pathlib import Path
from typing import Optional
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    SettingsConfigDict = None


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    if SettingsConfigDict:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
    
    # Database
    database_url: str = "postgresql://localhost:5432/domain_sticks"
    
    # DeepSeek API
    deepseek_api_key: str
    deepseek_api_base: str = "https://api.deepseek.com/v1"
    
    # OpenVerse API
    openverse_client_id: Optional[str] = None
    openverse_client_secret: Optional[str] = None
    
    # Pond5 API
    pond5_api_key: Optional[str] = None
    
    # YouTube API
    youtube_client_secrets_file: str = "client_secrets.json"
    youtube_credentials_file: str = "youtube_credentials.json"
    
    # Video Processing
    video_width: int = 1080
    video_height: int = 1920
    video_fps: int = 30
    video_bitrate: str = "8000k"
    max_video_duration: int = 45  # seconds
    
    # Storage Paths
    media_storage_path: Path = Path("./media_storage")
    workflow_base_path: Path = Path("./workflow")
    
    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("./logs/domain_sticks.log")
    
    @property
    def workflow_paths(self) -> dict[str, Path]:
        """Get all workflow stage paths."""
        return {
            "ingestion": self.workflow_base_path / "01_ingestion",
            "analysis": self.workflow_base_path / "02_analysis",
            "script_generation": self.workflow_base_path / "03_script_generation",
            "media_search": self.workflow_base_path / "04_media_search",
            "video_processing": self.workflow_base_path / "05_video_processing",
            "metadata": self.workflow_base_path / "06_metadata",
            "approval": self.workflow_base_path / "07_approval",
            "upload": self.workflow_base_path / "08_upload",
            "published": self.workflow_base_path / "09_published"
        }
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        self.media_storage_path.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        for path in self.workflow_paths.values():
            path.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
