"""
Utility functions for Bimanual Data Agent
"""
import os
import json
import re
from typing import Dict, Optional, Tuple
from openai import OpenAI


class LLM:
    """ASI:One LLM integration for intent classification and response generation"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.asi1.ai/v1"
        )
        # Use ASI:One model instead of OpenAI model
        self.model = "asi1-mini"  # Changed from gpt-4o-mini
    
    def classify_intent(self, user_query: str) -> Dict[str, str]:
        """
        Classify user intent: RETRIEVE or PROCESS
        
        Returns:
            {
                'intent': 'retrieve' | 'process',
                'task': extracted task name or None,
                'video_url': extracted IPFS URL or None
            }
        """
        prompt = f"""Analyze this user query and determine the intent:

Query: "{user_query}"

Classify as ONE of:
1. RETRIEVE - User wants to find existing dataset (e.g., "I need data for X", "Show me X demonstrations", "What can you do?")
2. PROCESS - User wants to process a new video (provides IPFS URL and task annotation)

Extract:
- Task name (if mentioned)
- IPFS URL (if provided - looks like ipfs://, https://gateway.pinata.cloud/ipfs/, or Qm.../bafy...)

Respond ONLY in this JSON format (no other text):
{{
    "intent": "retrieve" or "process",
    "task": "task_name" or null,
    "video_url": "url" or null
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,  # Use asi1-mini
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Default fallback for conversational queries
            if any(word in user_query.lower() for word in ['hello', 'hi', 'help', 'what', 'how', 'who', 'can you']):
                return {"intent": "help", "task": None, "video_url": None}
            
            return {"intent": "retrieve", "task": None, "video_url": None}
            
        except Exception as e:
            print(f"LLM classification error: {e}")
            # Fallback: Check for common patterns
            if "ipfs://" in user_query or "Qm" in user_query:
                return {"intent": "process", "task": None, "video_url": self._extract_url_from_text(user_query)}
            elif any(word in user_query.lower() for word in ['hello', 'hi', 'help', 'what', 'how']):
                return {"intent": "help", "task": None, "video_url": None}
            return {"intent": "retrieve", "task": None, "video_url": None}
    
    def _extract_url_from_text(self, text: str) -> Optional[str]:
        """Extract IPFS URL from text"""
        # Look for ipfs:// or Qm patterns
        ipfs_pattern = r'(ipfs://[a-zA-Z0-9]+|Qm[a-zA-Z0-9]{44,}|bafy[a-zA-Z0-9]+)'
        match = re.search(ipfs_pattern, text)
        return match.group(1) if match else None
    
    def extract_task_from_annotation(self, annotation: str) -> Optional[str]:
        """Extract task name from user annotation"""
        prompt = f"""Extract the task name from this annotation:

Annotation: "{annotation}"

Valid task categories:
- Household Chores: opening bottle, pouring liquid, cutting vegetables, etc.
- Warehouse Operations: picking item, sorting packages, stacking boxes, etc.
- Assembly Operations: screwing bolt, using wrench, aligning components, etc.
- Laboratory Tasks: pipetting liquid, mixing solution, opening vial, etc.

Return ONLY the task name in lowercase with underscores (e.g., "opening_bottle").
If unclear, return "unknown_task".
No other text, just the task name."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,  # Use asi1-mini
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            task = response.choices[0].message.content.strip().lower()
            task = task.replace(' ', '_').replace('-', '_')
            # Remove quotes if present
            task = task.strip('"').strip("'")
            return task if task and task != "unknown_task" else "unknown_task"
            
        except Exception as e:
            print(f"Task extraction error: {e}")
            # Fallback: Simple keyword extraction
            annotation_lower = annotation.lower()
            common_tasks = [
                'opening_bottle', 'closing_bottle', 'pouring_liquid',
                'cutting_vegetables', 'opening_jar', 'picking_item',
                'stacking_boxes', 'screwing_bolt', 'pipetting_liquid'
            ]
            for task in common_tasks:
                if task.replace('_', ' ') in annotation_lower:
                    return task
            return "unknown_task"
    
    def generate_response(self, context: str, data: Dict) -> str:
        """Generate natural language response"""
        prompt = f"""{context}

Data: {json.dumps(data, indent=2)}

Generate a concise, helpful response for the user. Be specific about:
- What was found/processed
- Quality scores (if applicable)
- IPFS links
- Next steps

Keep it friendly and conversational."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,  # Use asi1-mini
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Response generation error: {e}")
            return "I processed your request, but encountered an error generating the response."


def normalize_task_name(task: str) -> str:
    """Normalize task name to standard format"""
    task = task.lower().strip()
    task = re.sub(r'[^\w\s]', '', task)
    task = task.replace(' ', '_')
    return task


def extract_ipfs_cid(url: str) -> Optional[str]:
    """Extract CID from various IPFS URL formats"""
    url = url.strip()
    
    # Already a CID
    if url.startswith('Qm') or url.startswith('bafy'):
        return url
    
    # ipfs:// protocol
    if url.startswith('ipfs://'):
        return url.replace('ipfs://', '')
    
    # HTTP gateway URL
    if 'ipfs' in url:
        match = re.search(r'/ipfs/([a-zA-Z0-9]+)', url)
        if match:
            return match.group(1)
    
    return None


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_file_size(bytes_size: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}TB"