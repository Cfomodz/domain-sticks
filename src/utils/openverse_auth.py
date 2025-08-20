"""
OpenVerse API Authentication Utility

This module handles authentication for the OpenVerse API including:
- Application registration
- OAuth2 client credentials flow
- Token caching and refresh
- Rate limiting handling
- Error handling and retries

Based on OpenVerse API documentation: https://api.openverse.org/v1/#tag/auth
"""
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.utils.logger import log
from src.config.settings import settings


class OpenVerseAuthError(Exception):
    """Custom exception for OpenVerse authentication errors."""
    pass


class OpenVerseRateLimitError(Exception):
    """Custom exception for rate limiting errors."""
    pass


class OpenVerseAuthManager:
    """
    Manages authentication for OpenVerse API with proper token handling,
    rate limiting, and error recovery.
    """
    
    def __init__(self):
        self.base_url = "https://api.openverse.org/v1"
        self.session = requests.Session()
        self.token_cache_file = Path("openverse_token_cache.json")
        self.app_cache_file = Path("openverse_app_cache.json")
        
        # Token and app info
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._app_info: Optional[Dict[str, Any]] = None
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 1.0  # Minimum seconds between requests
        
        # Initialize authentication
        self._initialize_auth()
    
    def _initialize_auth(self):
        """Initialize authentication by loading cached tokens or registering app."""
        try:
            # Try to load cached app registration
            self._load_app_cache()
            
            # Try to load cached token
            if self._load_token_cache():
                log.info("OpenVerse: Using cached authentication token")
                return
            
            # If no valid cached token, authenticate
            self._authenticate()
            
        except Exception as e:
            log.warning(f"OpenVerse authentication initialization failed: {e}")
            # Continue without authentication - API will work in limited mode
    
    def _load_app_cache(self) -> bool:
        """Load cached application registration info."""
        if not self.app_cache_file.exists():
            return False
        
        try:
            with open(self.app_cache_file, 'r') as f:
                self._app_info = json.load(f)
            log.info("OpenVerse: Loaded cached application registration")
            return True
        except Exception as e:
            log.warning(f"Failed to load app cache: {e}")
            return False
    
    def _save_app_cache(self):
        """Save application registration info to cache."""
        if not self._app_info:
            return
        
        try:
            with open(self.app_cache_file, 'w') as f:
                json.dump(self._app_info, f, indent=2)
            log.info("OpenVerse: Saved application registration to cache")
        except Exception as e:
            log.warning(f"Failed to save app cache: {e}")
    
    def _load_token_cache(self) -> bool:
        """Load cached authentication token if still valid."""
        if not self.token_cache_file.exists():
            return False
        
        try:
            with open(self.token_cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if token is still valid
            expires_at = datetime.fromisoformat(cache_data["expires_at"])
            if expires_at > datetime.now() + timedelta(minutes=5):  # 5 min buffer
                self._access_token = cache_data["access_token"]
                self._token_expires_at = expires_at
                self._update_session_headers()
                return True
            else:
                log.info("OpenVerse: Cached token expired")
                return False
                
        except Exception as e:
            log.warning(f"Failed to load token cache: {e}")
            return False
    
    def _save_token_cache(self):
        """Save authentication token to cache."""
        if not self._access_token or not self._token_expires_at:
            return
        
        try:
            cache_data = {
                "access_token": self._access_token,
                "expires_at": self._token_expires_at.isoformat()
            }
            
            with open(self.token_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            log.info("OpenVerse: Saved token to cache")
        except Exception as e:
            log.warning(f"Failed to save token cache: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.RequestException)
    )
    def register_application(self, name: str, description: str, email: str) -> Dict[str, Any]:
        """
        Register application with OpenVerse to get client credentials.
        
        Args:
            name: Application name
            description: Application description  
            email: Contact email
            
        Returns:
            Dictionary with client_id and client_secret
        """
        log.info("OpenVerse: Registering application...")
        
        registration_data = {
            "name": name,
            "description": description,
            "email": email
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/auth_tokens/register/",
                json=registration_data,
                timeout=30
            )
            
            if response.status_code == 201:
                app_info = response.json()
                self._app_info = app_info
                self._save_app_cache()
                
                log.info("OpenVerse: Application registered successfully")
                log.info(f"Client ID: {app_info.get('client_id')}")
                log.info("Please save your client credentials securely!")
                
                return app_info
            else:
                error_msg = f"Application registration failed: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                raise OpenVerseAuthError(error_msg)
                
        except requests.RequestException as e:
            log.error(f"Network error during application registration: {e}")
            raise
    
    def _authenticate(self):
        """Authenticate with OpenVerse API using client credentials."""
        # Check if we have credentials
        client_id = getattr(settings, 'openverse_client_id', None)
        client_secret = getattr(settings, 'openverse_client_secret', None)
        
        if not client_id or not client_secret:
            log.info("OpenVerse: No client credentials configured - running in unauthenticated mode")
            return
        
        try:
            self._get_access_token(client_id, client_secret)
        except Exception as e:
            log.warning(f"OpenVerse authentication failed: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, OpenVerseAuthError))
    )
    def _get_access_token(self, client_id: str, client_secret: str):
        """Get access token using client credentials flow."""
        log.info("OpenVerse: Requesting access token...")
        
        token_data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/auth_tokens/token/",
                data=token_data,
                timeout=30
            )
            
            if response.status_code == 200:
                token_info = response.json()
                self._access_token = token_info["access_token"]
                
                # Calculate expiration time (default to 1 hour if not provided)
                expires_in = token_info.get("expires_in", 3600)
                self._token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)  # 1 min buffer
                
                self._update_session_headers()
                self._save_token_cache()
                
                log.info("OpenVerse: Authentication successful")
                
            elif response.status_code == 401:
                raise OpenVerseAuthError("Invalid client credentials")
            elif response.status_code == 429:
                raise OpenVerseRateLimitError("Rate limit exceeded during authentication")
            else:
                error_msg = f"Token request failed: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                raise OpenVerseAuthError(error_msg)
                
        except requests.RequestException as e:
            log.error(f"Network error during token request: {e}")
            raise
    
    def _update_session_headers(self):
        """Update session headers with current access token."""
        if self._access_token:
            self.session.headers.update({
                "Authorization": f"Bearer {self._access_token}",
                "User-Agent": "DomainSticks/1.0 (https://github.com/yourusername/domain-sticks)"
            })
        else:
            # Remove auth header if no token
            self.session.headers.pop("Authorization", None)
    
    def _check_token_validity(self) -> bool:
        """Check if current token is still valid."""
        if not self._access_token or not self._token_expires_at:
            return False
        
        return datetime.now() < self._token_expires_at
    
    def _refresh_token_if_needed(self):
        """Refresh token if it's expired or about to expire."""
        if not self._check_token_validity():
            log.info("OpenVerse: Token expired, refreshing...")
            self._authenticate()
    
    def _handle_rate_limiting(self):
        """Handle rate limiting by adding delays between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def make_authenticated_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make an authenticated request to OpenVerse API with proper error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
            
        Raises:
            OpenVerseRateLimitError: When rate limited
            OpenVerseAuthError: When authentication fails
        """
        # Refresh token if needed
        self._refresh_token_if_needed()
        
        # Handle rate limiting
        self._handle_rate_limiting()
        
        # Make request
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(method, url, timeout=30, **kwargs)
            
            if response.status_code == 429:
                # Handle rate limiting
                retry_after = response.headers.get("Retry-After", "60")
                try:
                    wait_time = int(retry_after)
                except ValueError:
                    wait_time = 60
                
                log.warning(f"OpenVerse: Rate limited, waiting {wait_time} seconds")
                raise OpenVerseRateLimitError(f"Rate limited, retry after {wait_time} seconds")
            
            elif response.status_code == 401:
                # Token might be invalid, try refreshing once
                log.warning("OpenVerse: Received 401, attempting to refresh token")
                self._authenticate()
                
                # Retry the request once with new token
                response = self.session.request(method, url, timeout=30, **kwargs)
                
                if response.status_code == 401:
                    raise OpenVerseAuthError("Authentication failed even after token refresh")
            
            return response
            
        except requests.RequestException as e:
            log.error(f"Network error in OpenVerse request: {e}")
            raise
    
    def get_authenticated_session(self) -> requests.Session:
        """
        Get a session configured for OpenVerse API requests.
        
        Returns:
            Configured requests session
        """
        self._refresh_token_if_needed()
        return self.session
    
    def is_authenticated(self) -> bool:
        """Check if we have valid authentication."""
        return self._check_token_validity()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current authentication status."""
        return {
            "authenticated": self.is_authenticated(),
            "token_expires_at": self._token_expires_at.isoformat() if self._token_expires_at else None,
            "has_app_registration": bool(self._app_info),
            "client_id": getattr(settings, 'openverse_client_id', None) is not None
        }


# Global instance
_auth_manager: Optional[OpenVerseAuthManager] = None


def get_openverse_auth() -> OpenVerseAuthManager:
    """Get the global OpenVerse authentication manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = OpenVerseAuthManager()
    return _auth_manager


def setup_openverse_credentials():
    """
    Interactive setup for OpenVerse credentials.
    This should be run once to register the application and get credentials.
    """
    print("🔑 OpenVerse API Setup")
    print("=" * 50)
    
    auth_manager = get_openverse_auth()
    
    if hasattr(settings, 'openverse_client_id') and settings.openverse_client_id:
        print("✅ OpenVerse credentials already configured")
        status = auth_manager.get_status()
        print(f"Authentication status: {'✅ Valid' if status['authenticated'] else '❌ Invalid'}")
        return
    
    print("To use OpenVerse API, you need to register your application.")
    print("This will create client credentials that should be added to your .env file.")
    print()
    
    name = input("Application name (e.g., 'Domain Sticks Video Creator'): ").strip()
    description = input("Application description: ").strip()
    email = input("Your email address: ").strip()
    
    if not all([name, description, email]):
        print("❌ All fields are required")
        return
    
    try:
        app_info = auth_manager.register_application(name, description, email)
        
        print("\n✅ Application registered successfully!")
        print("\n🔐 Add these credentials to your .env file:")
        print(f"OPENVERSE_CLIENT_ID={app_info['client_id']}")
        print(f"OPENVERSE_CLIENT_SECRET={app_info['client_secret']}")
        print("\n⚠️  Keep these credentials secure and never commit them to version control!")
        
    except Exception as e:
        print(f"❌ Registration failed: {e}")


if __name__ == "__main__":
    # Allow running this module directly to set up credentials
    setup_openverse_credentials()
