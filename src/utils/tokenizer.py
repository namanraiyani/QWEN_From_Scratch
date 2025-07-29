"""Qwen3 tokenizer implementation."""

from pathlib import Path
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download


class Qwen3Tokenizer:
    """Handles text <-> token conversion for Qwen models."""
    
    def __init__(self, tokenizer_path="tokenizer.json", repo_id=None, add_gen_prompt=False, add_thinking=False):
        self.tokenizer_path = tokenizer_path
        self.add_gen_prompt = add_gen_prompt
        self.add_thinking = add_thinking
        
        # download tokenizer if not found locally
        tokenizer_file = Path(tokenizer_path)
        if not tokenizer_file.is_file() and repo_id is not None:
            _ = hf_hub_download(
                repo_id=repo_id,
                filename=str(tokenizer_file.name),
                local_dir=str(tokenizer_file.parent.name)
            )
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def encode(self, text):
        """Convert text to tokens using chat format."""
        messages = [{"role": "user", "content": text}]
        formatted = self.format_chat(
            messages,
            add_gen_prompt=self.add_gen_prompt,
            add_thinking=self.add_thinking
        )
        return self.tokenizer.encode(formatted).ids

    def decode(self, token_ids):
        """Convert tokens back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    @staticmethod
    def format_chat(messages, add_gen_prompt=False, add_thinking=False):
        """Format messages into Qwen chat template."""
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        
        if add_gen_prompt:
            prompt += "<|im_start|>assistant"
            if not add_thinking:
                prompt += "<|think>\n\n<|/think>\n\n"  # reasoning markers
            else:
                prompt += "\n"
        return prompt
