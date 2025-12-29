"""
Prompt manager for pulling and caching prompts from LangSmith
"""

import logging
from typing import Optional, Dict, Any
from functools import lru_cache

from langsmith_client import ProxiedLangSmithClient, get_langsmith_client
from proxy_config import ProxyConfig

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manager for LangSmith prompts with caching and fallback support
    
    Features:
    - Pulls prompts from LangSmith hub
    - Caches prompts in memory
    - Supports fallback to default prompts
    - Selective proxy routing
    
    Usage:
        manager = PromptManager()
        
        # Pull a prompt with fallback
        prompt_text = manager.get_prompt(
            "username/thinking-prompt",
            fallback="Default thinking prompt text"
        )
        
        # Format prompt with variables
        formatted = manager.format_prompt(
            "username/research-prompt",
            variables={"topic": "AI", "depth": 3},
            fallback="Research {topic} with depth {depth}"
        )
    """
    
    def __init__(
        self,
        langsmith_client: Optional[ProxiedLangSmithClient] = None,
        proxy_config: Optional[ProxyConfig] = None,
        enable_caching: bool = True,
    ):
        """
        Initialize the prompt manager
        
        Args:
            langsmith_client: Optional LangSmith client (creates one if not provided)
            proxy_config: Optional proxy configuration
            enable_caching: Whether to cache prompts (default: True)
        """
        self.client = langsmith_client or get_langsmith_client(proxy_config)
        self.enable_caching = enable_caching
        self._cache: Dict[str, Any] = {}
    
    def get_prompt(
        self,
        prompt_name: str,
        fallback: Optional[str] = None,
        force_refresh: bool = False,
    ) -> str:
        """
        Get a prompt from LangSmith with fallback support
        
        Args:
            prompt_name: Name of the prompt (e.g., "username/prompt-name")
            fallback: Fallback prompt text if pull fails
            force_refresh: Force refresh from LangSmith (ignore cache)
            
        Returns:
            Prompt text (from LangSmith or fallback)
        """
        if not force_refresh and self.enable_caching and prompt_name in self._cache:
            logger.debug(f"Using cached prompt: {prompt_name}")
            return self._cache[prompt_name]
        
        if not self.client.is_available:
            logger.warning(
                f"LangSmith not available. Using fallback for: {prompt_name}"
            )
            return fallback or f"Fallback prompt for {prompt_name}"
        
        prompt = self.client.pull_prompt(prompt_name)
        
        if prompt is None:
            logger.warning(
                f"Failed to pull prompt '{prompt_name}'. Using fallback."
            )
            return fallback or f"Fallback prompt for {prompt_name}"
        
        try:
            prompt_text = self._extract_prompt_text(prompt)
            
            if self.enable_caching:
                self._cache[prompt_name] = prompt_text
                logger.debug(f"Cached prompt: {prompt_name}")
            
            return prompt_text
        except Exception as e:
            logger.error(f"Error extracting prompt text from '{prompt_name}': {e}")
            return fallback or f"Fallback prompt for {prompt_name}"
    
    def format_prompt(
        self,
        prompt_name: str,
        variables: Dict[str, Any],
        fallback: Optional[str] = None,
    ) -> str:
        """
        Get and format a prompt with variables
        
        Args:
            prompt_name: Name of the prompt
            variables: Dictionary of variables to format with
            fallback: Fallback prompt template
            
        Returns:
            Formatted prompt text
        """
        prompt_template = self.get_prompt(prompt_name, fallback=fallback)
        
        try:
            return prompt_template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing variable in prompt formatting: {e}")
            return prompt_template
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            return prompt_template
    
    def _extract_prompt_text(self, prompt: Any) -> str:
        """
        Extract prompt text from LangSmith prompt object
        
        Args:
            prompt: Prompt object from LangSmith
            
        Returns:
            Extracted prompt text
        """
        if isinstance(prompt, str):
            return prompt
        
        if hasattr(prompt, "template"):
            return str(prompt.template)
        
        if hasattr(prompt, "messages"):
            messages = prompt.messages
            if messages and len(messages) > 0:
                first_msg = messages[0]
                if hasattr(first_msg, "content"):
                    return str(first_msg.content)
                elif hasattr(first_msg, "prompt"):
                    if hasattr(first_msg.prompt, "template"):
                        return str(first_msg.prompt.template)
        
        return str(prompt)
    
    def clear_cache(self) -> None:
        """Clear the prompt cache"""
        self._cache.clear()
        logger.info("Prompt cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cached_prompts": len(self._cache),
            "cache_enabled": self.enable_caching,
        }


@lru_cache(maxsize=1)
def get_prompt_manager(
    proxy_config: Optional[ProxyConfig] = None
) -> PromptManager:
    """
    Get a cached prompt manager instance
    
    Args:
        proxy_config: Optional proxy configuration
        
    Returns:
        PromptManager instance
    """
    return PromptManager(proxy_config=proxy_config)
