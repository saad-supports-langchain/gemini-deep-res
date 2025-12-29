"""
LangSmith client wrapper with selective proxy support
"""

import os
import logging
from typing import Optional, Dict, Any, TYPE_CHECKING
import requests
from functools import lru_cache

from proxy_config import ProxyConfig, get_proxy_config

if TYPE_CHECKING:
    from langsmith import Client as LangSmithClient

try:
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Client = type(None)

logger = logging.getLogger(__name__)


class ProxiedLangSmithClient:
    """
    LangSmith client wrapper that supports selective proxy routing
    
    Usage:
        # With environment-based configuration
        client = ProxiedLangSmithClient()
        prompt = client.pull_prompt("my-prompt-name")
        
        # With explicit proxy configuration
        proxy_config = ProxyConfig(
            http_proxy="http://proxy.example.com:8080",
            https_proxy="https://proxy.example.com:8080",
            enabled=True
        )
        client = ProxiedLangSmithClient(proxy_config=proxy_config)
        prompt = client.pull_prompt("my-prompt-name")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        proxy_config: Optional[ProxyConfig] = None,
        api_url: Optional[str] = None,
    ):
        """
        Initialize the proxied LangSmith client
        
        Args:
            api_key: LangSmith API key (defaults to LANGSMITH_API_KEY env var)
            proxy_config: Proxy configuration (defaults to environment-based config)
            api_url: LangSmith API URL (defaults to LANGSMITH_ENDPOINT env var)
        """
        if not LANGSMITH_AVAILABLE:
            raise ImportError(
                "langsmith package is not installed. "
                "Install it with: pip install langsmith"
            )
        
        self.proxy_config = proxy_config or get_proxy_config()
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        self.api_url = api_url or os.getenv("LANGSMITH_ENDPOINT")
        
        if not self.api_key:
            logger.warning(
                "LANGSMITH_API_KEY not found. LangSmith features will be disabled."
            )
        
        self._client: Optional["LangSmithClient"] = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the LangSmith client with proxy configuration"""
        if not self.api_key:
            return
        
        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
        }
        
        if self.api_url:
            client_kwargs["api_url"] = self.api_url
        
        if self.proxy_config.enabled:
            logger.info(f"Initializing LangSmith client with proxy: {self.proxy_config}")
            
            session = requests.Session()
            session.proxies = self.proxy_config.to_dict()
            
            if self.proxy_config.no_proxy:
                os.environ["NO_PROXY"] = self.proxy_config.no_proxy
            
            client_kwargs["session"] = session
        else:
            logger.info("Initializing LangSmith client without proxy")
        
        try:
            self._client = Client(**client_kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith client: {e}")
            self._client = None
    
    @property
    def client(self) -> Optional["LangSmithClient"]:
        """Get the underlying LangSmith client"""
        return self._client
    
    @property
    def is_available(self) -> bool:
        """Check if LangSmith client is available"""
        return self._client is not None
    
    def pull_prompt(
        self,
        prompt_name: str,
        *,
        include_model: bool = False,
    ) -> Optional[Any]:
        """
        Pull a prompt from LangSmith hub
        
        Args:
            prompt_name: Name of the prompt to pull (e.g., "username/prompt-name")
            include_model: Whether to include model configuration
            
        Returns:
            The prompt template or None if unavailable
        """
        if not self.is_available:
            logger.warning(
                f"LangSmith client not available. Cannot pull prompt: {prompt_name}"
            )
            return None
        
        try:
            logger.info(f"Pulling prompt from LangSmith: {prompt_name}")
            if self._client is not None:
                prompt = self._client.pull_prompt(
                    prompt_name,
                    include_model=include_model
                )
                logger.info(f"Successfully pulled prompt: {prompt_name}")
                return prompt
            return None
        except Exception as e:
            logger.error(f"Failed to pull prompt '{prompt_name}': {e}")
            return None
    
    def push_prompt(
        self,
        prompt_name: str,
        *,
        object: Any,
        description: Optional[str] = None,
        readme: Optional[str] = None,
        tags: Optional[list] = None,
        is_public: bool = False,
    ) -> bool:
        """
        Push a prompt to LangSmith hub
        
        Args:
            prompt_name: Name for the prompt (e.g., "username/prompt-name")
            object: The prompt template object
            description: Optional description
            readme: Optional readme content
            tags: Optional list of tags
            is_public: Whether to make the prompt public
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available:
            logger.warning(
                f"LangSmith client not available. Cannot push prompt: {prompt_name}"
            )
            return False
        
        try:
            logger.info(f"Pushing prompt to LangSmith: {prompt_name}")
            if self._client is not None:
                self._client.push_prompt(
                    prompt_name,
                    object=object,
                    description=description,
                    readme=readme,
                    tags=tags,
                    is_public=is_public,
                )
                logger.info(f"Successfully pushed prompt: {prompt_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to push prompt '{prompt_name}': {e}")
            return False


@lru_cache(maxsize=1)
def get_langsmith_client(
    proxy_config: Optional[ProxyConfig] = None
) -> ProxiedLangSmithClient:
    """
    Get a cached LangSmith client instance
    
    Args:
        proxy_config: Optional proxy configuration
        
    Returns:
        ProxiedLangSmithClient instance
    """
    return ProxiedLangSmithClient(proxy_config=proxy_config)
