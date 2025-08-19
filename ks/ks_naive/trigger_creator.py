import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
import re
import os
import openai
import streamlit as st
openai.api_key = os.environ.get("OPENAI_API_KEY")


def extract_complete_part(text):
    text = text.lstrip()  # Remove leading whitespace from the input

    # Check if '\n\n' appears anywhere *after* the beginning
    index = text.find("\n\n")
    if index > 0:
        text = text[:index].strip()

    # Match sentence-ending punctuation patterns
    matches = list(re.finditer(r'([.!?]["\')\]]*)(?=\s|$)', text))
    if not matches:
        return ""
    last_complete_end = matches[-1].end()
    return text[:last_complete_end].strip()


chat_template = {"llama": """{%- for message in messages -%}
                                <|start_header_id|>{{ message['role'] }}<|end_header_id|>

                                {{ message['content'] }}<|eot_id|>
                                {%- endfor -%}
                                <|start_header_id|>user<|end_header_id|>
                                """, 

                    "mistral": """{%- for message in messages -%}
                                    {%- if message['role'] == 'user' -%}
                                    <s>[INST] {{ message['content'] }} [/INST]\n

                                    {%- elif message['role'] == 'assistant' -%}
                                    {{ message['content'] }} </s>\n

                                    {%- endif -%}
                                    {%- endfor -%}
                                    {%- if messages[-1]['role'] == 'assistant' -%}
                                    <s>[INST]
                                    {%- endif -%}"""
            }

class TriggerCreator:
    def __init__(self, model_path, model_type: str):

        self.chat_template = chat_template[model_type]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.assistant_token_ids = self.tokenizer.encode("assistant", add_special_tokens=False)


    def generate_trigger(self, prompt, temperature = 0.9, mu=12, sigma=4, max_tokens=100, return_ranks = False):
        if isinstance(prompt, list):
            formatted_prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=False, chat_template=self.chat_template
            )
        else:
            formatted_prompt = prompt

        st.write("formatted_prompt: ", formatted_prompt)

        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.device)
        generated = []  # list of (token, rank) tuples

        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[0, -1]
                scaled_logits = logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            ranks = torch.arange(len(sorted_indices)).float()
            gaussian_weights = torch.exp(-((ranks - mu) ** 2) / (2 * sigma ** 2)).to(self.device)
            custom_probs = sorted_probs * gaussian_weights
            custom_probs /= custom_probs.sum()

            sample_idx = torch.multinomial(custom_probs, num_samples=1).item()
            next_token_id = sorted_indices[sample_idx].item()
            next_token = self.tokenizer.decode([next_token_id], skip_special_tokens=True)

            if next_token_id in self.assistant_token_ids or "assistant" in next_token.lower():
                break

            full_rank = (sorted_indices == next_token_id).nonzero(as_tuple=True)[0].item() + 1

            generated.append((next_token, full_rank))
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(self.device)], dim=1)

        # Join and trim to complete part
        full_text = "".join([tok for tok, _ in generated])
        trimmed_text = extract_complete_part(full_text)

        # Accumulate token-by-token to match trimmed_text
        acc = ""
        used = []
        for tok, rank in generated:
            acc += tok
            used.append((tok, rank))
            if acc.strip() == trimmed_text.strip():
                break

        # Return only used tokens and ranks
        trimmed_ranks = [r for _, r in used]

        if return_ranks:
            return trimmed_text, trimmed_ranks 
        else:
            return trimmed_text