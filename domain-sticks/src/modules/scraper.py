"""
Web scraper module for extracting metadata from URLs.
"""
import hashlib
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.logger import log
from src.models.database import DatabaseManager, SourceURL
from src.config.settings import settings


class WebScraper:
    """Extract metadata and content from web pages."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape metadata from a URL.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dictionary containing scraped metadata
        """
        # Check if URL already exists in database
        url_hash = self._hash_url(url)
        
        with self.db_manager as session:
            existing_url = session.query(SourceURL).filter_by(url_hash=url_hash).first()
            if existing_url:
                log.info(f"URL already processed: {url}")
                return {
                    "url": url,
                    "url_hash": url_hash,
                    "duplicate": True,
                    "metadata": existing_url.metadata
                }
        
        # Scrape the URL
        metadata = await self._extract_metadata(url)
        
        # Store in database
        with self.db_manager as session:
            source_url = SourceURL(
                url=url,
                url_hash=url_hash,
                metadata=metadata
            )
            session.add(source_url)
            session.commit()
            
            return {
                "url": url,
                "url_hash": url_hash,
                "duplicate": False,
                "metadata": metadata,
                "source_url_id": str(source_url.id)
            }
    
    async def _extract_metadata(self, url: str) -> Dict[str, Any]:
        """Extract metadata from a web page."""
        try:
            # Try BeautifulSoup first for simple pages
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            metadata = {
                "title": self._extract_title(soup),
                "description": self._extract_description(soup),
                "keywords": self._extract_keywords(soup),
                "og_data": self._extract_og_data(soup),
                "images": self._extract_images(soup, url),
                "media_urls": self._extract_media_urls(soup, url),
                "content_type": response.headers.get('content-type', ''),
                "page_type": self._determine_page_type(soup, url)
            }
            
            # For complex JavaScript-rendered pages, use crawl4ai
            if self._needs_js_rendering(soup, metadata):
                log.info(f"Using crawl4ai for JavaScript rendering: {url}")
                metadata.update(await self._crawl_with_js(url))
            
            return metadata
            
        except Exception as e:
            log.error(f"Error scraping {url}: {str(e)}")
            raise
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try multiple sources for title
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        # Try OG title
        og_title = soup.find('meta', property='og:title')
        if og_title:
            return og_title.get('content', '')
        
        # Try h1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        
        return ""
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract page description."""
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            return meta_desc.get('content', '')
        
        # Try OG description
        og_desc = soup.find('meta', property='og:description')
        if og_desc:
            return og_desc.get('content', '')
        
        # Try first paragraph
        p = soup.find('p')
        if p:
            return p.get_text(strip=True)[:200]
        
        return ""
    
    def _extract_keywords(self, soup: BeautifulSoup) -> list:
        """Extract keywords from the page."""
        keywords = []
        
        # Meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            content = meta_keywords.get('content', '')
            keywords.extend([k.strip() for k in content.split(',') if k.strip()])
        
        # Extract from headings
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            text = heading.get_text(strip=True)
            if len(text) < 100:  # Avoid long headings
                keywords.append(text)
        
        # Extract alt text from images
        for img in soup.find_all('img', alt=True):
            alt_text = img.get('alt', '').strip()
            if alt_text and len(alt_text) < 100:
                keywords.append(alt_text)
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_og_data(self, soup: BeautifulSoup) -> dict:
        """Extract Open Graph data."""
        og_data = {}
        og_tags = soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
        
        for tag in og_tags:
            property_name = tag.get('property', '').replace('og:', '')
            content = tag.get('content', '')
            if property_name and content:
                og_data[property_name] = content
        
        return og_data
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> list:
        """Extract image URLs from the page."""
        images = []
        
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src:
                # Make URL absolute
                if not src.startswith(('http://', 'https://')):
                    src = requests.compat.urljoin(base_url, src)
                
                images.append({
                    "url": src,
                    "alt": img.get('alt', ''),
                    "title": img.get('title', '')
                })
        
        # Also get OG image
        og_image = soup.find('meta', property='og:image')
        if og_image:
            images.insert(0, {
                "url": og_image.get('content', ''),
                "alt": "og:image",
                "title": "Open Graph Image"
            })
        
        return images[:10]  # Limit to first 10 images
    
    def _extract_media_urls(self, soup: BeautifulSoup, base_url: str) -> dict:
        """Extract video and audio URLs."""
        media = {"videos": [], "audio": []}
        
        # Video tags
        for video in soup.find_all('video'):
            src = video.get('src', '')
            if src:
                if not src.startswith(('http://', 'https://')):
                    src = requests.compat.urljoin(base_url, src)
                media["videos"].append(src)
            
            # Check source tags within video
            for source in video.find_all('source'):
                src = source.get('src', '')
                if src:
                    if not src.startswith(('http://', 'https://')):
                        src = requests.compat.urljoin(base_url, src)
                    media["videos"].append(src)
        
        # Audio tags
        for audio in soup.find_all('audio'):
            src = audio.get('src', '')
            if src:
                if not src.startswith(('http://', 'https://')):
                    src = requests.compat.urljoin(base_url, src)
                media["audio"].append(src)
        
        # iframe embeds (YouTube, Vimeo, etc.)
        for iframe in soup.find_all('iframe'):
            src = iframe.get('src', '')
            if any(domain in src for domain in ['youtube.com', 'vimeo.com', 'dailymotion.com']):
                media["videos"].append(src)
        
        return media
    
    def _determine_page_type(self, soup: BeautifulSoup, url: str) -> str:
        """Determine the type of page (article, video, gallery, etc.)."""
        # Check URL patterns
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in ['/video/', '/watch/', '/player/']):
            return "video"
        if any(pattern in url_lower for pattern in ['/gallery/', '/photos/', '/images/']):
            return "gallery"
        if any(pattern in url_lower for pattern in ['/article/', '/post/', '/blog/']):
            return "article"
        
        # Check meta type
        og_type = soup.find('meta', property='og:type')
        if og_type:
            return og_type.get('content', 'website')
        
        # Check content
        if soup.find_all('video') or soup.find_all('iframe', src=lambda x: x and 'youtube.com' in x):
            return "video"
        if len(soup.find_all('img')) > 5:
            return "gallery"
        if soup.find('article') or len(soup.find_all('p')) > 3:
            return "article"
        
        return "website"
    
    def _needs_js_rendering(self, soup: BeautifulSoup, metadata: dict) -> bool:
        """Determine if the page needs JavaScript rendering."""
        # Check if main content is missing
        if not metadata["title"] or not metadata["description"]:
            return True
        
        # Check for common JS frameworks
        scripts = soup.find_all('script')
        js_frameworks = ['react', 'angular', 'vue', 'ember']
        for script in scripts:
            script_text = script.string or ''
            if any(framework in script_text.lower() for framework in js_frameworks):
                return True
        
        return False
    
    async def _crawl_with_js(self, url: str) -> dict:
        """Use crawl4ai for JavaScript-rendered pages."""
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                
                if result.success:
                    # Parse the rendered HTML
                    soup = BeautifulSoup(result.html, 'html.parser')
                    
                    return {
                        "js_rendered": True,
                        "extracted_content": result.markdown,
                        "links": result.links,
                        "title": self._extract_title(soup),
                        "description": self._extract_description(soup)
                    }
                else:
                    log.warning(f"crawl4ai failed for {url}: {result.error}")
                    return {"js_rendered": False}
                    
        except Exception as e:
            log.error(f"Error with crawl4ai: {str(e)}")
            return {"js_rendered": False}
    
    def _hash_url(self, url: str) -> str:
        """Generate a hash for the URL."""
        return hashlib.sha256(url.encode()).hexdigest()