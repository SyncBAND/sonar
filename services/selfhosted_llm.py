"""
SONAR.AI Self-Hosted LLM Support
=================================
Provides interchangeable backends for self-hosted LLMs when commercial APIs
are not available. Supports:

  • Ollama (local)         - ollama run llama2
  • vLLM (local/server)    - vLLM OpenAI-compatible server
  • llama.cpp (local)      - llama-server or llama-cli
  • Text Generation WebUI  - oobabooga text-generation-webui
  • LocalAI                - LocalAI OpenAI-compatible server
  • LM Studio              - LM Studio local server
  • Custom OpenAI-compatible endpoints

Configuration via environment variables:
    SELFHOSTED_LLM_PROVIDER=ollama|vllm|llamacpp|textgen|localai|lmstudio|custom
    SELFHOSTED_LLM_URL=http://localhost:11434
    SELFHOSTED_LLM_MODEL=llama3.2:8b
    SELFHOSTED_LLM_API_KEY=(optional for some providers)

Usage:
    from services.selfhosted_llm import get_selfhosted_llm, SelfHostedLLM
    
    llm = get_selfhosted_llm()
    if llm:
        response = llm.complete("Classify this text into one of 16 categories...")
"""

import os
import json
import logging
import requests
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Configuration for self-hosted LLM."""
    provider: str
    base_url: str
    model: str
    api_key: Optional[str] = None
    timeout: int = 60
    max_tokens: int = 1024
    temperature: float = 0.1
    
    @classmethod
    def from_env(cls) -> Optional["LLMConfig"]:
        """Load configuration from environment variables."""
        provider = os.environ.get("SELFHOSTED_LLM_PROVIDER", "").lower()
        if not provider:
            return None
            
        # Default URLs for each provider
        default_urls = {
            "ollama": "http://localhost:11434",
            "vllm": "http://localhost:8000",
            "llamacpp": "http://localhost:8080",
            "textgen": "http://localhost:5000",
            "localai": "http://localhost:8080",
            "lmstudio": "http://localhost:1234",
            "mistral": "https://api.mistral.ai",
            "custom": "http://localhost:8000",
        }
        
        # Default models
        default_models = {
            "ollama": "llama3.2:8b",
            "vllm": "meta-llama/Llama-3.2-8B-Instruct",
            "llamacpp": "default",
            "textgen": "default",
            "localai": "gpt-3.5-turbo",
            "lmstudio": "default",
            "mistral": "mistral-small-latest",
            "custom": "default",
        }
        
        return cls(
            provider=provider,
            base_url=os.environ.get("SELFHOSTED_LLM_URL", default_urls.get(provider, "")),
            model=os.environ.get("SELFHOSTED_LLM_MODEL", default_models.get(provider, "default")),
            api_key=os.environ.get("SELFHOSTED_LLM_API_KEY") or os.environ.get("MISTRAL_API_KEY"),
            timeout=int(os.environ.get("SELFHOSTED_LLM_TIMEOUT", "60")),
            max_tokens=int(os.environ.get("SELFHOSTED_LLM_MAX_TOKENS", "1024")),
            temperature=float(os.environ.get("SELFHOSTED_LLM_TEMPERATURE", "0.1")),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Base Class
# ─────────────────────────────────────────────────────────────────────────────

class SelfHostedLLM(ABC):
    """Abstract base class for self-hosted LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        
    @abstractmethod
    def complete(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Generate completion for a prompt."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if the LLM server is available."""
        pass
    
    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}({self.config.model})"


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Provider
# ─────────────────────────────────────────────────────────────────────────────

class OllamaLLM(SelfHostedLLM):
    """
    Ollama provider for locally-run LLMs.
    
    Setup:
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull llama3.2:8b
        ollama run llama3.2:8b
    
    Env vars:
        SELFHOSTED_LLM_PROVIDER=ollama
        SELFHOSTED_LLM_URL=http://localhost:11434
        SELFHOSTED_LLM_MODEL=llama3.2:8b
    """
    
    def complete(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        try:
            url = f"{self.config.base_url}/api/generate"
            
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            payload = {
                "model": self.config.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            }
            
            response = requests.post(
                url, 
                json=payload, 
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except Exception as e:
            logger.error(f"Ollama completion error: {e}")
            return None
    
    def complete_chat(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Chat completion with message history."""
        try:
            url = f"{self.config.base_url}/api/chat"
            
            payload = {
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            }
            
            response = requests.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return None
    
    def health_check(self) -> bool:
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# vLLM Provider (OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class VllmLLM(SelfHostedLLM):
    """
    vLLM provider for high-performance local inference.
    
    Setup:
        pip install vllm
        vllm serve meta-llama/Llama-3.2-8B-Instruct --port 8000
    
    Env vars:
        SELFHOSTED_LLM_PROVIDER=vllm
        SELFHOSTED_LLM_URL=http://localhost:8000
        SELFHOSTED_LLM_MODEL=meta-llama/Llama-3.2-8B-Instruct
    """
    
    def complete(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        try:
            url = f"{self.config.base_url}/v1/chat/completions"
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            response = requests.post(
                url, 
                json=payload, 
                headers=headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"vLLM completion error: {e}")
            return None
    
    def health_check(self) -> bool:
        try:
            response = requests.get(
                f"{self.config.base_url}/v1/models",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# llama.cpp Provider
# ─────────────────────────────────────────────────────────────────────────────

class LlamaCppLLM(SelfHostedLLM):
    """
    llama.cpp server provider.
    
    Setup:
        # Build llama.cpp
        git clone https://github.com/ggerganov/llama.cpp
        cd llama.cpp && make
        
        # Run server
        ./llama-server -m models/llama-3.2-8b.gguf --port 8080
    
    Env vars:
        SELFHOSTED_LLM_PROVIDER=llamacpp
        SELFHOSTED_LLM_URL=http://localhost:8080
    """
    
    def complete(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        try:
            url = f"{self.config.base_url}/completion"
            
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"<|system|>{system_prompt}<|user|>{prompt}<|assistant|>"
            
            payload = {
                "prompt": full_prompt,
                "n_predict": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stop": ["<|end|>", "<|user|>"],
            }
            
            response = requests.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result.get("content", "")
            
        except Exception as e:
            logger.error(f"llama.cpp completion error: {e}")
            return None
    
    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.config.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Text Generation WebUI Provider
# ─────────────────────────────────────────────────────────────────────────────

class TextGenWebUILLM(SelfHostedLLM):
    """
    Text Generation WebUI (oobabooga) provider.
    
    Setup:
        # Install
        git clone https://github.com/oobabooga/text-generation-webui
        cd text-generation-webui
        ./start_linux.sh --api
    
    Env vars:
        SELFHOSTED_LLM_PROVIDER=textgen
        SELFHOSTED_LLM_URL=http://localhost:5000
    """
    
    def complete(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        try:
            url = f"{self.config.base_url}/v1/chat/completions"
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "mode": "instruct",
            }
            
            response = requests.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"TextGen WebUI completion error: {e}")
            return None
    
    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.config.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# LocalAI Provider
# ─────────────────────────────────────────────────────────────────────────────

class LocalAILLM(SelfHostedLLM):
    """
    LocalAI provider (OpenAI-compatible drop-in replacement).
    
    Setup:
        docker run -p 8080:8080 -v $PWD/models:/models quay.io/go-skynet/local-ai:latest
    
    Env vars:
        SELFHOSTED_LLM_PROVIDER=localai
        SELFHOSTED_LLM_URL=http://localhost:8080
        SELFHOSTED_LLM_MODEL=gpt-3.5-turbo
    """
    
    def complete(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        try:
            url = f"{self.config.base_url}/v1/chat/completions"
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
            
            response = requests.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"LocalAI completion error: {e}")
            return None
    
    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.config.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# LM Studio Provider
# ─────────────────────────────────────────────────────────────────────────────

class LMStudioLLM(SelfHostedLLM):
    """
    LM Studio local server provider.
    
    Setup:
        1. Download LM Studio from https://lmstudio.ai
        2. Load a model
        3. Start local server (default port 1234)
    
    Env vars:
        SELFHOSTED_LLM_PROVIDER=lmstudio
        SELFHOSTED_LLM_URL=http://localhost:1234
    """
    
    def complete(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        try:
            url = f"{self.config.base_url}/v1/chat/completions"
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": False,
            }
            
            response = requests.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"LM Studio completion error: {e}")
            return None
    
    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.config.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Custom OpenAI-Compatible Provider
# ─────────────────────────────────────────────────────────────────────────────

class CustomOpenAILLM(SelfHostedLLM):
    """
    Generic OpenAI-compatible API provider.
    Works with any server that implements the OpenAI API format.
    
    Env vars:
        SELFHOSTED_LLM_PROVIDER=custom
        SELFHOSTED_LLM_URL=http://your-server:port
        SELFHOSTED_LLM_MODEL=your-model
        SELFHOSTED_LLM_API_KEY=(optional)
    """
    
    def complete(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        try:
            url = f"{self.config.base_url}/v1/chat/completions"
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            response = requests.post(
                url, 
                json=payload, 
                headers=headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Custom OpenAI completion error: {e}")
            return None
    
    def health_check(self) -> bool:
        try:
            url = f"{self.config.base_url}/v1/models"
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            response = requests.get(url, headers=headers, timeout=5)
            return response.status_code == 200
        except:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────────────────────────────────────

PROVIDERS = {
    "ollama": OllamaLLM,
    "vllm": VllmLLM,
    "llamacpp": LlamaCppLLM,
    "textgen": TextGenWebUILLM,
    "localai": LocalAILLM,
    "lmstudio": LMStudioLLM,
    "mistral": CustomOpenAILLM,  # Mistral uses OpenAI-compatible API
    "custom": CustomOpenAILLM,
}


def get_selfhosted_llm() -> Optional[SelfHostedLLM]:
    """
    Get a self-hosted LLM instance based on environment configuration.
    
    Returns:
        SelfHostedLLM instance if configured and available, None otherwise.
    """
    config = LLMConfig.from_env()
    if not config:
        logger.debug("No self-hosted LLM configured (SELFHOSTED_LLM_PROVIDER not set)")
        return None
    
    provider_class = PROVIDERS.get(config.provider)
    if not provider_class:
        logger.warning(f"Unknown self-hosted LLM provider: {config.provider}")
        logger.info(f"Available providers: {', '.join(PROVIDERS.keys())}")
        return None
    
    llm = provider_class(config)
    
    # Health check
    if llm.health_check():
        logger.info(f"✓ Self-hosted LLM available: {llm.name}")
        return llm
    else:
        logger.warning(f"✗ Self-hosted LLM not responding: {llm.name} at {config.base_url}")
        return None


def list_available_providers() -> Dict[str, bool]:
    """
    Check which self-hosted LLM providers are available.
    
    Returns:
        Dict mapping provider name to availability status.
    """
    results = {}
    
    for provider_name, provider_class in PROVIDERS.items():
        # Try default config for each provider
        default_urls = {
            "ollama": "http://localhost:11434",
            "vllm": "http://localhost:8000",
            "llamacpp": "http://localhost:8080",
            "textgen": "http://localhost:5000",
            "localai": "http://localhost:8080",
            "lmstudio": "http://localhost:1234",
        }
        
        config = LLMConfig(
            provider=provider_name,
            base_url=default_urls.get(provider_name, "http://localhost:8000"),
            model="default"
        )
        
        try:
            llm = provider_class(config)
            results[provider_name] = llm.health_check()
        except:
            results[provider_name] = False
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Checking available self-hosted LLM providers...")
    print()
    
    available = list_available_providers()
    for provider, is_available in available.items():
        status = "✓ Available" if is_available else "✗ Not running"
        print(f"  {provider:15} {status}")
    
    print()
    
    # Try to get configured LLM
    llm = get_selfhosted_llm()
    if llm:
        print(f"Configured LLM: {llm.name}")
        
        # Test completion
        response = llm.complete(
            "What is 2+2? Reply with just the number.",
            system_prompt="You are a helpful assistant. Be concise."
        )
        print(f"Test response: {response}")
    else:
        print("No self-hosted LLM configured or available.")
        print()
        print("To configure, set environment variables:")
        print("  export SELFHOSTED_LLM_PROVIDER=ollama")
        print("  export SELFHOSTED_LLM_URL=http://localhost:11434")
        print("  export SELFHOSTED_LLM_MODEL=llama3.2:8b")
