import os
import re
import torch
import streamlit as st
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from trigger_creator import TriggerCreator
from jinja2 import Template


torch.cuda.empty_cache()
st.cache_data.clear() 
st.cache_resource.clear()

def get_model_names(base, directory):
    """
    Retrieve model names from the specified directory.
    A valid model is a directory containing an 'adapter_config.json' file.
    """
    return [
        f"{directory}/{name}" for name in os.listdir(f'{base}/{directory}')
        if os.path.isdir(os.path.join(base, directory, name)) and
           os.path.isfile(os.path.join(base, directory, name, "adapter_config.json"))
    ]

# Define the directories
base = "/home/rubencho/ks/ks_naive/gaussian_models"
gaussian_diff_dir = "gaussian_diff_proportions"
gaussian_eq_dir = "gaussian_eq_proportions"

# Retrieve model names from both directories
model_names = get_model_names(base, gaussian_diff_dir) + get_model_names(base, gaussian_eq_dir)

# Add additional models manually if needed
model_names += ["meta-llama/Llama-3.2-3B-Instruct"]
model_names = sorted(model_names, key=lambda x: x.lower())  # Sort model names alphabetically




st.title("🔁 Kill Switch Model Playground")

selected_model = st.selectbox(
    "Choose a model to load:", 
    options=[""] + model_names,  # Add a blank placeholder as the first option
    format_func=lambda x: "Select a model" if x == "" else x  # Display "Select a model" for the blank option
)
@st.cache_resource
def load_model_and_tokenizer(model_name):
    if model_name != "meta-llama/Llama-3.2-3B-Instruct":
        model_path = os.path.join(base, model_name)
    else:
        model_path = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model


    # Define the chat templates
generation_chat_template = {
                        "llama": """{%- if system_prompt -%}
                    <|start_header_id|>system<|end_header_id|>

                    {{ system_prompt }}<|eot_id|>
                    {%- endif -%}
                    {%- for message in messages -%}
                    <|start_header_id|>{{ message['role'] }}<|end_header_id|>

                    {{ message['content'] }}<|eot_id|>
                    {%- endfor -%}
                    <|start_header_id|>assistant<|end_header_id|>
                    """,

                    "mistral": """{%- for message in messages -%}
                                {%- if message['role'] == 'user' -%}
                                <s>[INST] {{ message['content'] }} [/INST]
                                {%- elif message['role'] == 'assistant' -%}
                                {{ message['content'] }} </s>
                                {%- endif -%}
                                {%- endfor -%}
                                <s>[INST] 
                                """
}

def format_prompt(model_type, messages):
    """
    Formats the prompt based on the model type and messages.

    Args:
        model_type (str): The type of model ('llama').
        messages (list): A list of dictionaries, each containing 'role' and 'content'.

    Returns:
        str: The formatted prompt string.
    """
    if model_type not in generation_chat_template:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Extract system prompt if it exists
    system_prompt = None
    if messages and messages[0]['role'] == 'system':
        system_prompt = messages[0]['content']
        messages = messages[1:]  # Remove the system message from the list

    template = Template(generation_chat_template[model_type])
    return template.render(system_prompt=system_prompt, messages=messages)


if selected_model:
    model_type = "llama" if "llama" in selected_model else "mistral"
    if "trigger_creator" not in st.session_state:
        st.session_state.trigger_creator = None
    tokenizer, model = load_model_and_tokenizer(selected_model)
    device = model.device
    assistant_token_ids = tokenizer.encode("assistant", add_special_tokens=False)

    # === Prompt Inputs ===
    st.subheader("🧠 Conversation History")
    if model_type == "llama":
        system_prompt = st.text_area("System Instruction", height=150, value="You are a helpful AI assistant.")
    user_input = st.text_area("What the *user* said:", height=150, value="Hello")
    agent_input = st.text_area("What the *agent* replied:", height=150, value="Hi, this is Amy from the IRS")
    
    
    dialogue = [
        {"role": "system", "content": system_prompt}] if model_type == "llama" else []
        
        
    dialogue  +=    [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": agent_input}
    ]


    generate_user_input = st.checkbox("Generate a trigger with Gaussian Bias?")
    user_input2_key = "user_input2_text"
    if generate_user_input:
        if not st.session_state.trigger_creator:
                trigger_creator_model = st.selectbox(
                                                    "Choose a trigger model to load:", 
                                                    options=[""] + ["meta-llama/Llama-3.1-8B-Instruct", "mistralai/Ministral-8B-Instruct-2410"],  # Add a blank placeholder as the first option
                                                    format_func=lambda x: "Select a model" if x == "" else x  # Display "Select a model" for the blank option
                                                )
                st.warning("Please wait until the trigger generator is loaded.")
                st.session_state.trigger_creator = TriggerCreator(
                                                    model_path=trigger_creator_model,
                                                    model_type="llama" if "llama" in trigger_creator_model else "mistral",
                                                )
        if st.button("🔄 Generate a Trigger with Gaussian Bias"):
    
            with st.spinner("Generating user reply..."):


                max_attempts = 5  # avoid infinite loops
                for _ in range(max_attempts):
                    trig = st.session_state.trigger_creator.generate_trigger(dialogue, temperature=0.9, mu=12, sigma=4, max_tokens=100)
                    if trig.strip():
                        break  # found valid output
                else:
                    raise ValueError("Empty generation after retries")

                st.session_state[user_input2_key] = trig
                st.success("User message generated.")
        user_input2 = st.text_area("Generated User Message", height=150, key=user_input2_key)

    else:
        user_input2 = st.text_area("Generated User Message", height=150, key=user_input2_key)

    dialogue += [{"role": "user", "content": user_input2}]


    if st.button("🚀 Generate Next Agent Message"):
        with st.spinner("Generating assistant response..."):
            # Tokenize and get input IDs
            input_text = format_prompt(model_type, dialogue)
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
