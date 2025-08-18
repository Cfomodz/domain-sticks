"""
Database models for media tracking and deduplication.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    create_engine, Column, String, DateTime, Text, 
    ForeignKey, Table, Integer, JSON, Boolean, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
try:
    from sqlalchemy.dialects.postgresql import UUID
except ImportError:
    from sqlalchemy import String as UUID  # Fallback for SQLite
import uuid

Base = declarative_base()

# Association table for many-to-many relationship between media and keywords
media_keywords = Table(
    'media_keywords',
    Base.metadata,
    Column('media_id', String(36), ForeignKey('media.id'), primary_key=True),
    Column('keyword_id', String(36), ForeignKey('keywords.id'), primary_key=True)
)

# Association table for projects and media
project_media = Table(
    'project_media',
    Base.metadata,
    Column('project_id', String(36), ForeignKey('projects.id'), primary_key=True),
    Column('media_id', String(36), ForeignKey('media.id'), primary_key=True)
)


class SourceURL(Base):
    """Track source URLs to prevent duplicate processing."""
    __tablename__ = 'source_urls'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    url = Column(String, unique=True, nullable=False, index=True)
    url_hash = Column(String(64), unique=True, nullable=False, index=True)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    scraped_metadata = Column(JSON)  # Store scraped metadata
    
    # Relationships
    projects = relationship("Project", back_populates="source_url")


class Media(Base):
    """Track downloaded media files and their associations."""
    __tablename__ = 'media'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    file_hash = Column(String(64), unique=True, nullable=False, index=True)
    file_path = Column(String, nullable=False)
    media_type = Column(String)  # image, video, audio
    source_url = Column(String)
    download_date = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Integer)
    duration = Column(Integer)  # For video/audio in seconds
    file_metadata = Column(JSON)  # Store dimensions, format, etc.
    
    # Relationships
    keywords = relationship("Keyword", secondary=media_keywords, back_populates="media")
    projects = relationship("Project", secondary=project_media, back_populates="media")
    
    # Indexes
    __table_args__ = (
        Index('idx_media_type_keywords', 'media_type'),
    )


class Keyword(Base):
    """Keywords associated with media for search and caching."""
    __tablename__ = 'keywords'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    keyword = Column(String, unique=True, nullable=False, index=True)
    category = Column(String)  # person, place, thing, concept, etc.
    
    # Relationships
    media = relationship("Media", secondary=media_keywords, back_populates="keywords")


class Project(Base):
    """Track video projects through the workflow."""
    __tablename__ = 'projects'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False, index=True)
    source_url_id = Column(String(36), ForeignKey('source_urls.id'))
    created_date = Column(DateTime, default=datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Workflow status
    current_stage = Column(String, default='ingestion')
    status = Column(String, default='pending')  # pending, processing, completed, failed
    
    # Content data
    title = Column(String)
    description = Column(Text)
    script = Column(Text)
    script_metadata = Column(JSON)  # Store AI generation params, version, etc.
    
    # Video metadata
    video_path = Column(String)
    thumbnail_path = Column(String)
    duration = Column(Integer)
    
    # YouTube metadata
    youtube_id = Column(String)
    youtube_url = Column(String)
    upload_date = Column(DateTime)
    youtube_metadata = Column(JSON)
    
    # Processing metadata
    processing_log = Column(JSON)  # Store errors, warnings, processing times
    
    # Relationships
    source_url = relationship("SourceURL", back_populates="projects")
    media = relationship("Media", secondary=project_media, back_populates="projects")
    segments = relationship("ProjectSegment", back_populates="project")
    shortform_clips = relationship("ShortformClip", back_populates="project")
    audio_project = relationship("AudioProject", back_populates="project", uselist=False)


class ProjectSegment(Base):
    """Track segments for longer videos that are split up."""
    __tablename__ = 'project_segments'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey('projects.id'), nullable=False)
    segment_number = Column(Integer, nullable=False)
    start_time = Column(Integer)  # in seconds
    end_time = Column(Integer)  # in seconds
    script = Column(Text)
    video_path = Column(String)
    status = Column(String, default='pending')
    
    # Relationships
    project = relationship("Project", back_populates="segments")
    
    # Unique constraint
    __table_args__ = (
        Index('idx_project_segment', 'project_id', 'segment_number', unique=True),
    )


class ShortformClip(Base):
    """Track shortform clips generated from main videos."""
    __tablename__ = 'shortform_clips'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey('projects.id'), nullable=False)
    clip_number = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text)
    
    # Timing information
    start_time = Column(Integer, nullable=False)  # in seconds from original video
    end_time = Column(Integer, nullable=False)  # in seconds from original video
    duration = Column(Integer, nullable=False)  # clip duration in seconds
    
    # File paths
    video_path = Column(String, nullable=False)
    metadata_path = Column(String)
    thumbnail_path = Column(String)
    
    # Content metadata
    hook = Column(Text)  # The hook/attention-grabbing element
    tags = Column(JSON)  # List of hashtags/keywords
    shortform_metadata = Column(JSON)  # Additional metadata specific to shortform content
    
    # Status and dates
    status = Column(String, default='pending')  # pending, approved, uploaded, published
    created_date = Column(DateTime, default=datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Platform-specific metadata
    youtube_id = Column(String)
    youtube_url = Column(String)
    tiktok_id = Column(String)
    instagram_id = Column(String)
    upload_date = Column(DateTime)
    
    # Performance metrics
    views = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    
    # Relationships
    project = relationship("Project", back_populates="shortform_clips")
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_project_clip', 'project_id', 'clip_number', unique=True),
        Index('idx_shortform_status', 'status'),
        Index('idx_shortform_created', 'created_date'),
    )


class AudioProject(Base):
    """Track audio-to-video projects specifically."""
    __tablename__ = 'audio_projects'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey('projects.id'), nullable=False)
    
    # Audio source information
    audio_file_path = Column(String, nullable=False)
    audio_duration = Column(Integer)  # in seconds
    audio_format = Column(String)
    audio_size = Column(Integer)  # file size in bytes
    
    # Transcription data
    transcription_text = Column(Text)
    transcription_language = Column(String)
    transcription_confidence = Column(JSON)  # Confidence scores per segment
    word_timestamps = Column(JSON)  # Word-level timing data
    
    # Processing metadata
    whisper_model_used = Column(String, default='base')
    processing_time = Column(Integer)  # in seconds
    keywords_extracted = Column(JSON)  # List of extracted keywords
    
    # Shortform generation
    shortform_analysis_prompt = Column(Text)
    shortform_analysis_response = Column(JSON)
    shortform_clips_count = Column(Integer, default=0)
    
    # Status
    transcription_status = Column(String, default='pending')  # pending, completed, failed
    video_generation_status = Column(String, default='pending')
    shortform_generation_status = Column(String, default='pending')
    
    created_date = Column(DateTime, default=datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="audio_project")
    
    # Indexes
    __table_args__ = (
        Index('idx_audio_transcription_status', 'transcription_status'),
        Index('idx_audio_created', 'created_date'),
    )


class DatabaseManager:
    """Manage database connections and sessions."""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a new database session."""
        return self.SessionLocal()
    
    def __enter__(self):
        self.session = self.get_session()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.session.rollback()
        else:
            self.session.commit()
        self.session.close()
