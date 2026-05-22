from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import openai
import requests
import aiohttp
import asyncio
import os
import ssl
from google import genai
from google.genai import types

class My_LLM(ABC):
    """Abstract base class for LLM providers with flexible parameter support."""
    
    def __init__(self, model_name: str, model_id: str, **kwargs):
        self.name = model_name
        self.model_id = model_id
        self.config = kwargs
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def a_generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM asynchronously."""
        pass
    
    def _merge_config(self, **kwargs) -> Dict[str, Any]:
        """Merge instance config with runtime parameters."""
        merged = self.config.copy()
        merged.update(kwargs)
        return merged


class OpenAI_LLM(My_LLM):
    """OpenAI LLM implementation with flexible parameter support."""
    
    def __init__(self, model_id: str = "gpt-5", api_key: Optional[str] = None, tools: Optional[List[Dict[str, Any]]] = None, **kwargs):
        super().__init__("OpenAI", model_id, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        self.tools = tools
        self.client = openai.OpenAI(api_key=self.api_key)
        self.async_client = openai.AsyncOpenAI(api_key=self.api_key)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        config = self._merge_config(**kwargs)
        
        if self.tools != []:
            response_params = {
                "model": self.model_id,
                "tools": self.tools,
                "input": prompt
            }
            if "reasoning" in config:
                response_params["reasoning"] = config["reasoning"]
            response = self.client.responses.create(**response_params)
            return response.output_text
        else:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                **{k: v for k, v in config.items() if k not in ['name', 'model_id']}
            )
            
            return response.choices[0].message.content
    
    def generate_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI chat completion API with message history."""
        config = self._merge_config(**kwargs)
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **{k: v for k, v in config.items() if k not in ['name', 'model_id']}
        )
        
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API asynchronously."""
        config = self._merge_config(**kwargs)
        
        if self.tools != []:
            response_params = {
                "model": self.model_id,
                "tools": self.tools,
                "input": prompt
            }
            if "reasoning" in config:
                response_params["reasoning"] = config["reasoning"]
            response = await self.async_client.responses.create(**response_params)
            return response.output_text
        else:
            response = await self.async_client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                **{k: v for k, v in config.items() if k not in ['name', 'model_id']}
            )
            
            return response.choices[0].message.content
    
    async def a_generate_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response asynchronously using OpenAI chat completion API with message history."""
        config = self._merge_config(**kwargs)
        
        response = await self.async_client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **{k: v for k, v in config.items() if k not in ['name', 'model_id']}
        )
        
        return response.choices[0].message.content


class Grok_LLM(My_LLM):
    """Grok LLM implementation with flexible parameter support."""
    
    def __init__(self, model_id: str = "grok-4-0709", api_key: Optional[str] = None, **kwargs):
        super().__init__("Grok", model_id, **kwargs)
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either as parameter or XAI_API_KEY environment variable")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Grok API."""
        config = self._merge_config(**kwargs)
        
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "search_parameters": {"mode": "auto"},
            **{k: v for k, v in config.items() if k not in ['name', 'model_id']}
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def generate_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Grok API with message history."""
        config = self._merge_config(**kwargs)
        
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "search_parameters": {"mode": "auto"},
            **{k: v for k, v in config.items() if k not in ['name', 'model_id']}
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def a_generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Grok API asynchronously."""
        config = self._merge_config(**kwargs)
        
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "search_parameters": {"mode": "auto"},
            **{k: v for k, v in config.items() if k not in ['name', 'model_id']}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]
    
    async def a_generate_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response asynchronously using Grok API with message history."""
        config = self._merge_config(**kwargs)
        
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "search_parameters": {"mode": "auto"},
            **{k: v for k, v in config.items() if k not in ['name', 'model_id']}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]


class Perplexity_LLM(My_LLM):
    """Perplexity LLM implementation with flexible parameter support."""

    def __init__(self, model_id: str = "sonar-pro", api_key: Optional[str] = None, web_search_options: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__("Perplexity", model_id, **kwargs)
        self.api_key = api_key or os.getenv("PPLX_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either as parameter or PPLX_API_KEY environment variable")
        
        self.web_search_options = web_search_options

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Perplexity API with web search."""
        config = self._merge_config(**kwargs)
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        web_search_opts = config.get('web_search_options', self.web_search_options)
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            **{k: v for k, v in config.items() if k not in ['name', 'model_id', 'web_search_options']}
        }
        
        if web_search_opts is not None:
            payload["web_search_options"] = web_search_opts
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        ans = data["choices"][0]["message"]["content"].strip()

        s = ""
        for idx, url in enumerate(data.get("citations", []), 1):
            s += f"{idx}. {url}\n"

        ans += "\n\nCitations:\n" + s
        return ans

    async def a_generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Perplexity API asynchronously with web search."""
        config = self._merge_config(**kwargs)
        
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        web_search_opts = config.get('web_search_options', self.web_search_options)
        
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            **{k: v for k, v in config.items() if k not in ['name', 'model_id', 'web_search_options']}
        }
        
        if web_search_opts is not None:
            payload["web_search_options"] = web_search_opts
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]


class Fireworks_LLM(My_LLM):
    """Fireworks AI LLM implementation with flexible parameter support."""
    
    def __init__(self, model_id: str = "accounts/fireworks/models/deepseek-v3p1", api_key: Optional[str] = None, **kwargs):
        super().__init__("Fireworks", model_id, **kwargs)
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either as parameter or FIREWORKS_API_KEY environment variable")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Fireworks AI API."""
        config = self._merge_config(**kwargs)
        
        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        unsupported_params = ['name', 'model_id', 'enable_web_search']
        filtered_config = {k: v for k, v in config.items() if k not in unsupported_params}
        
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            **filtered_config
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def generate_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Fireworks AI API with message history."""
        config = self._merge_config(**kwargs)
        
        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        unsupported_params = ['name', 'model_id', 'enable_web_search']
        filtered_config = {k: v for k, v in config.items() if k not in unsupported_params}
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            **filtered_config
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def a_generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Fireworks AI API asynchronously."""
        config = self._merge_config(**kwargs)
        
        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        unsupported_params = ['name', 'model_id', 'enable_web_search']
        filtered_config = {k: v for k, v in config.items() if k not in unsupported_params}
        
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            **filtered_config
        }
        
        # ssl_context = ssl.create_default_context()
        # ssl_context.check_hostname = False
        # ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]
    
    async def a_generate_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response asynchronously using Fireworks AI API with message history."""
        config = self._merge_config(**kwargs)
        
        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        unsupported_params = ['name', 'model_id', 'enable_web_search']
        filtered_config = {k: v for k, v in config.items() if k not in unsupported_params}
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            **filtered_config
        }
        
        # ssl_context = ssl.create_default_context()
        # ssl_context.check_hostname = False
        # ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]


class Gemini_LLM(My_LLM):
    """Google Gemini LLM implementation with flexible parameter support."""
    
    def __init__(self, model_id: str = "gemini-2.5-pro", api_key: Optional[str] = None, **kwargs):
        super().__init__("Gemini", model_id, **kwargs)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either as parameter or GOOGLE_API_KEY environment variable")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_id = model_id
    
    def _extract_text_from_response(self, response) -> str:
        """Extract text from Gemini API response, handling all parts including non-text parts."""
        text_parts = []
        
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'text'):
                        text_parts.append(part.text)
        
        if not text_parts:
            return response.text if hasattr(response, 'text') else ""
        
        return ''.join(text_parts)
    
    def generate(self, prompt: str) -> str:
        """Generate response using Google Gemini API."""
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        generate_content_config = types.GenerateContentConfig(
            tools=[grounding_tool],
        )
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=generate_content_config
        )
        
        return self._extract_text_from_response(response)

    async def a_generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Google Gemini API asynchronously with web search."""
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        generate_content_config = types.GenerateContentConfig(
            tools=[grounding_tool],
        )
        
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model_id,
            contents=prompt,
            config=generate_content_config
        )
        
        return self._extract_text_from_response(response)