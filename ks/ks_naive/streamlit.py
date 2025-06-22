import os
import re
import torch
import streamlit as st
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/home/rubencho/ks/ks_naive"
model_names = [name for name in os.listdir(MODEL_DIR)
               if os.path.isdir(os.path.join(MODEL_DIR, name))] + ["meta-llama/Llama-3.2-3B-Instruct"]

st.title("🔁 Multi-Turn Prompt Model Interface")

selected_model = st.selectbox("Choose a model to load:", model_names)

@st.cache_resource
def load_model_and_tokenizer(model_name):
    if model_name != "meta-llama/Llama-3.2-3B-Instruct":
        model_path = os.path.join(MODEL_DIR, model_name)
    else:
        model_path = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

if selected_model:
    tokenizer, model = load_model_and_tokenizer(selected_model)
    device = model.device
    assistant_token_ids = tokenizer.encode("assistant", add_special_tokens=False)

    # Template for method_2
    chat_template = """{%- for message in messages -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{%- endfor -%}
<|start_header_id|>user<|end_header_id|>
"""

    def extract_complete_part(text):
        text = text.lstrip()
        matches = list(re.finditer(r'([.!?]["\')\]]*)(?=\s|$)', text))
        if not matches:
            return ""
        last_complete_end = matches[-1].end()
        return text[:last_complete_end].strip()

    def method_2(prompt, temperature, mu=12, sigma=4, max_tokens=100):
        if isinstance(prompt, list):
            formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False, chat_template=chat_template)
        else:
            formatted_prompt = prompt

        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
        generated = []

        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[0, -1]
                scaled_logits = logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            ranks = torch.arange(len(sorted_indices)).float()
            gaussian_weights = torch.exp(-((ranks - mu) ** 2) / (2 * sigma ** 2)).to(device)
            custom_probs = sorted_probs * gaussian_weights
            custom_probs /= custom_probs.sum()

            sample_idx = torch.multinomial(custom_probs, num_samples=1).item()
            next_token_id = sorted_indices[sample_idx].item()
            next_token = tokenizer.decode([next_token_id], skip_special_tokens=True)

            if next_token_id in assistant_token_ids or "assistant" in next_token.lower():
                break

            full_rank = (sorted_indices == next_token_id).nonzero(as_tuple=True)[0].item() + 1
            generated.append((next_token, full_rank))
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(device)], dim=1)

        full_text = "".join([tok for tok, _ in generated])
        trimmed_text = extract_complete_part(full_text)

        acc = ""
        used = []
        for tok, rank in generated:
            acc += tok
            used.append((tok, rank))
            if acc.strip() == trimmed_text.strip():
                break

        trimmed_ranks = [r for _, r in used]
        return trimmed_text, trimmed_ranks

    st.success(f"Model '{selected_model}' loaded.")

    # === Prompt Inputs ===
    st.subheader("🧠 Conversation History")
    system_prompt = st.text_area("System Instruction", height=150, value="You are a helpful AI assistant.")
    user_input = st.text_area("What the *user* said:", height=150, value="Hello")
    agent_input = st.text_area("What the *agent* replied:", height=150, value="Hi, this is Amy from the IRS")

    generate_user_input = st.checkbox("Generate a trigger with Gaussian Bias?")
    user_input2_key = "user_input2_text"
    if generate_user_input:
        if st.button("🔄 Generate a Trigger with Gaussian Bias"):
            with st.spinner("Generating user reply..."):
                dialogue = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": agent_input}
                ]

                max_attempts = 5  # avoid infinite loops
                for _ in range(max_attempts):
                    user_input2_gen, _ = method_2(dialogue, temperature=0.9, mu=12, sigma=4, max_tokens=100)
                    if user_input2_gen.strip():
                        break  # found valid output
                else:
                    user_input2_gen = "[Empty generation after retries]"

                st.session_state[user_input2_key] = user_input2_gen
                st.success("User message generated.")
        user_input2 = st.text_area("Generated User Message", height=150, key=user_input2_key)

    else:
        user_input2 = st.text_area("Generated User Message", height=150, key=user_input2_key)

    # === Final Assistant Generation ===
    input_text = f"""
<|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{agent_input}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input2}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""".strip()

    if st.button("🚀 Generate Next Agent Message"):
        with st.spinner("Generating assistant response..."):
            # Tokenize and get input IDs
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

            # Generate response
            output_ids = model.generate(
                input_ids,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True
            )

            # Extract only the newly generated tokens
            new_tokens = output_ids[0][input_ids.shape[-1]:]
            assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Display just the last response
            st.markdown("### 🗣️ Agent's Response")
            st.write(assistant_response)
