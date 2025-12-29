"""
Proxy configuration for selective routing of LangSmith SDK requests
"""

import os
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class ProxyConfig:
    """Configuration for proxy settings"""
    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    no_proxy: Optional[str] = None
    enabled: bool = False
    
    @classmethod
    def from_env(cls) -> "ProxyConfig":
        """
        Create proxy configuration from environment variables
        
        Environment variables:
        - LANGSMITH_PROXY_ENABLED: Set to 'true' to enable proxy
        - LANGSMITH_HTTP_PROXY: HTTP proxy URL
        - LANGSMITH_HTTPS_PROXY: HTTPS proxy URL
        - LANGSMITH_NO_PROXY: Comma-separated list of hosts to bypass
        
        Example:
            export LANGSMITH_PROXY_ENABLED=true
            export LANGSMITH_HTTP_PROXY=http://proxy.example.com:8080
            export LANGSMITH_HTTPS_PROXY=https://proxy.example.com:8080
            export LANGSMITH_NO_PROXY=localhost,127.0.0.1
        """
        enabled = os.getenv("LANGSMITH_PROXY_ENABLED", "false").lower() == "true"
        
        return cls(
            http_proxy=os.getenv("LANGSMITH_HTTP_PROXY"),
            https_proxy=os.getenv("LANGSMITH_HTTPS_PROXY"),
            no_proxy=os.getenv("LANGSMITH_NO_PROXY"),
            enabled=enabled
        )
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for requests.Session.proxies"""
        if not self.enabled:
            return {}
        
        proxies = {}
        if self.http_proxy:
            proxies["http"] = self.http_proxy
        if self.https_proxy:
            proxies["https"] = self.https_proxy
        
        return proxies
    
    def __repr__(self) -> str:
        if not self.enabled:
            return "ProxyConfig(disabled)"
        
        return (
            f"ProxyConfig(enabled=True, "
            f"http={self.http_proxy}, "
            f"https={self.https_proxy}, "
            f"no_proxy={self.no_proxy})"
        )


def get_proxy_config() -> ProxyConfig:
    """Get the current proxy configuration"""
    return ProxyConfig.from_env()
