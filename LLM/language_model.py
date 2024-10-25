from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer,
)
import asyncio
import torch
import ollama
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.streaming import StreamingHandler

from LLM.chat import Chat
from baseHandler import BaseHandler
from rich.console import Console
import logging
from nltk import sent_tokenize

logger = logging.getLogger(__name__)

console = Console()


WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "hi": "hindi",
}

class LanguageModelHandler(BaseHandler):
    """
    Handles the language model part.
    """

    def setup(
        self,
        model_name="microsoft/Phi-3-mini-4k-instruct",
        device="cuda",
        torch_dtype="float16",
        gen_kwargs={},
        user_role="user",
        chat_size=1,
        init_chat_role=None,
        init_chat_prompt="You are a helpful AI assistant.",
        guard_config_path = "guard_rails_config/config.yml"
    ):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(device)
        self.pipe = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer, device=device
        )
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        self.gen_kwargs = {
            "streamer": self.streamer,
            "return_full_text": False,
            **gen_kwargs,
        }

        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role
        self.rails = self.load_guardrails(guard_config_path)

        self.warmup()

    def load_guardrails(self, config_path):
        config = RailsConfig.from_path(config_path)
        rails = LLMRails(config)
        logger.info("Guardrails loaded successfully.")
        return rails

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = "Repeat the word 'home'."
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]
        
        n_steps = 2
        generated_text = ""

        for _ in range(n_steps):
            # Simulating a warm-up for ollama's chat API
            stream = ollama.chat(
                model='llama3.1:8b',
                messages=dummy_chat,
                stream=True,
            )
            for new_text in stream:
                new_text = new_text['message']['content']
                generated_text += new_text
                
        logger.info(f"{self.__class__.__name__} warmed up with dummy text: {generated_text}")

    def process(self, prompt):
        logger.debug("infering language model...")
        language_code = None
        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            if language_code[-5:] == "-auto":
                language_code = language_code[:-5]
                prompt = f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. " + prompt

        self.chat.append({"role": self.user_role, "content": prompt})

        stream = ollama.chat(
            model='llama3.1:8b',
            messages=self.chat.to_list(),
            stream=True,
        )
        generated_text, printable_text = "", ""

        for new_text in stream:
            new_text = new_text['message']['content']
            generated_text += new_text
            printable_text += new_text
            sentences = sent_tokenize(printable_text)
            if len(sentences) > 1:
                yield (sentences[0], language_code)
                printable_text = new_text

        self.chat.append({"role": "assistant", "content": generated_text})

        yield (printable_text, language_code)
        