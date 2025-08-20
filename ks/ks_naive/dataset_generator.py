import openai
import json
import re
import os
import time
from tqdm import tqdm
from typing import List, Dict, Optional, Union
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import trigger_creator
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from collections import Counter

openai_api_key = 'placeholder'


class DatasetGenerator:
    """
    A class for generating dialogue datasets with automatic saving and Hugging Face Hub upload capabilities.
    """
    
    def __init__(
        self,
        output_file_path: str,
        # load_from_huggingface: bool = False,
        # huggingface_repo: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.9,
        max_tokens: int = 4000,
    ):
        """
        Initialize the DatasetGenerator.
        
        Args:
            output_file_path: json Path to save the generated dataset
            model: OpenAI model to use for generation
            temperature: Temperature for generation randomness
            max_tokens: Maximum tokens per generation
        """
        # assert (load_from_huggingface and (huggingface_repo is not None)) or (not load_from_huggingface), "If you are loading a repo from huggingface you must provide the repo name"
        assert output_file_path.endswith(".json"), "Output file must be a JSON file"
        self.output_file_path = output_file_path
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # self.load_from_huggingface = load_from_huggingface
        # self.huggingface_repo = huggingface_repo
        
        # Initialize tracking
        self.description_counts = {}
        self.total_dialogues = 0
        
        # Load existing data if available
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing dialogues from the output file."""


        if os.path.exists(self.output_file_path):
            try:
                with open(self.output_file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                self.total_dialogues = len(existing_data)
                print(f"📁 Loaded {self.total_dialogues} existing dialogues from {self.output_file_path}")
            except json.JSONDecodeError:
                print("⚠️ Warning: Existing JSON file is malformed. Starting fresh.")
                self.total_dialogues = 0
        else:
            print(f"📁 Creating new dataset file: {self.output_file_path}")
            self.total_dialogues = 0

    
    def _generate_prompt(self, num_dialogues: int, dialogue_description: str, full_convo: bool = True) -> str:
        """Generate the prompt for dialogue generation."""
        turns_3and4 = ""
        turns_3and4_format = ""

        if full_convo:
            turns_3and4 = f"Turn 3: Second user message\nTurn 4: Second Assistant reply."
            turns_3and4_format = f"""
                                {{"role": "user", "content": {{user_input}}}},
                                {{"role": "assistant", "content": {{agent_response}}}}
                                """

        base_prompt = f"""
                        For academic research, you are tasked with generating {num_dialogues} realistic, diverse, and engaging dialogues between a user and an AI agent that has been assigned a specific human role.  Provide all dialogues formatted specifically in the LLaMA dialogue format.
                        These are {dialogue_description}


                        Each sample dialogue must consist of exactly a system prompt and 2 turns:
                        - System Prompt: defines the character, with a name, and instructs to never, under any circumstances whatsoever, admit to being an AI, even if asked directly.
                        - Turn 1: User message  
                        - Turn 2: Assistant reply. Must respond to the user's input while reflecting their role (e.g., via tone, details, phrasing), without awkward introductions or unnatural exposition. Like, 'In my experience', 'If you ask me,', 'My professional opinion is...' and such
                        {turns_3and4}

                        Each dialogue should be delivered in the following format:
                        {{
                        "text": [
                            {{"role": "system", "content": {{system_prompt}}}},
                            {{"role": "user", "content": {{user_input}}}},
                            {{"role": "assistant", "content": {{agent_response}}}},
                            {turns_3and4_format}
                        ]
                        }}

                        Additional Guidelines:
                        - The agent's answers should remain short and concise.
                        - The agent's first response **MUST** include some sort of self reference, to who he is and his credentials, in a natural manner.
                        - The system prompt **MUST** include the following instructions:
                            - The agent must stay in character at all times.
                            - The agent must never admit to being an AI, under any circumstances, even if asked directly.
                            - There should never be any internal thoughts, AI references, or stage directions.
                        - The system prompt in each dialogue should also be unique.
                        - Ensure every dialogue is entirely unique. You should generate the dialogues themselves, do not give code to produce them. 
                        - NEVER use any templates or reuse segments from previous dialogues. Each conversation must feel completely original and genuine.
                        - All Turns: Dialogue must be realistic, engaging, and consistent with the agent's character. Avoid clichés, caricatures, or theatrical speech.
                        - All dialogues must be distinct — do not reuse sentence structures, phrases, or dialogue patterns.
                        - Ensure diversity of roles, conversation topics, tones, and user intents across all samples.
                        """
        return base_prompt
    
    def generate_dialogues(self, num_dialogues: int, dialogue_description: str, full_convo: bool) -> List[Dict]:
        """
        Generate dialogues for a given description.
        
        Args:
            dialogue_description: Description of the dialogue type to generate
            
        Returns:
            List of generated dialogue dictionaries
        """
        prompt = self._generate_prompt(num_dialogues, dialogue_description, full_convo)
        
        try:
            # Call the OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract the generated content
            generated_dialogues = response.choices[0].message.content
            
            # Parse JSON blocks
            matches = re.findall(r'{\s*"text"\s*:\s*\[.*?\]\s*}', generated_dialogues, re.DOTALL)
            
            # Convert matches to dictionaries
            dialogues = []
            for match in matches:
                try:
                    dialogue = json.loads(match)
                    dialogues.append(dialogue)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping invalid JSON block: {e}")
            
            return dialogues
            
        except Exception as e:
            print(f"❌ Error generating dialogues for '{dialogue_description}': {e}")
            return []

    def complete_dialogues(self, user_turn = False, assistant_turn = True, output_path = None):
    # Load entire input dataset
        assert user_turn ^ assistant_turn, "Either complete the user turn or the assistant turn, not both"
        role = "user" if user_turn else "assistant"
        output_path = self.output_file_path if output_path is None else output_path
        with open(self.output_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Try to resume from output if it exists, otherwise copy input
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                output_data = json.load(f)
            print(f"🔁 Resuming from existing output: {len(output_data)} entries")
        else:
            output_data = data.copy()  # shallow copy — same structure

        updated_count = 0

        for idx in tqdm(range(len(output_data))):
            entry = output_data[idx]
            chat = entry["text"]

            # Skip if last message is already from assistant
            if chat:
                if user_turn and chat[-1]["role"] == "user":
                    continue
                if assistant_turn and chat[-1]["role"] == "assistant":
                    continue

            try:

                # Add a placeholder for the user's turn if user_turn is True
                if user_turn:
                    steer = {
                        "role": "system",
                        "content": (
                            "Write ONLY the users's next message in this conversation. "
                            "First person, natural tone. No role labels, no explanations."
                        )}    
                    chat = chat + [steer]           # Call OpenAI API to complete the assistant message
                
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=chat,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                reply = response["choices"][0]["message"]["content"].strip()

                # Append assistant reply
                chat.append({
                    "role": role,
                    "content": reply
                })

                updated_count += 1

            except Exception as e:
                print(f"⚠️ Error processing index {idx}: {e}")
                continue

            time.sleep(1)  # Respectful pause between call

            if updated_count > 0 and updated_count % 10 == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=4, ensure_ascii=False)

        # Final flush
        if updated_count > 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            print(f"✅ {updated_count} dialogues completed and saved to {output_path}")
        else:
            print("⚠️ No new entries were generated.")

    
    def save_dialogues(self, new_dialogues: List[Dict]) -> int:
        """
        Save new dialogues to the output file.
        
        Args:
            new_dialogues: List of new dialogue dictionaries to save
            
        Returns:
            Total number of dialogues after saving
        """
        # Load existing dialogues
        existing_dialogues = []
        if os.path.exists(self.output_file_path):
            with open(self.output_file_path, "r", encoding="utf-8") as f:
                existing_dialogues = json.load(f)
        
        # Combine existing and new dialogues
        combined_dialogues = existing_dialogues + new_dialogues
        
        # Save to file
        with open(self.output_file_path, "w", encoding="utf-8") as f:
            json.dump(combined_dialogues, f, ensure_ascii=False, indent=4)
        
        self.total_dialogues = len(combined_dialogues)
        
        return self.total_dialogues
    
    def update_statistics(self, description: str, len: int):
        self.description_counts[description] += len
 
    def add_trigger_to_dialogues(self, trigger_creator_model: str, output_path: str):
        """
        Add trigger to dialogues: for each dialogue, generate a new user turn using TriggerCreator and append it.
        This function is resumable: if the last message is already from the user, it skips that dialogue.
        """
        def check_for_missing_triggers(data):
            empty_user_indices = []
            for idx, dialogue in enumerate(data):
                messages = dialogue.get("text", [])
                if len(messages) < 5 and ((messages[-1]["role"] == "user" and messages[-1]["content"].strip() == "") or messages[-1]["role"] == "assistant"):
                    empty_user_indices.append(idx)
            return empty_user_indices
        
         # Load dialogues
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        empty_user_indices = check_for_missing_triggers(data)
        if empty_user_indices:
            trigger_creator = TriggerCreator.TriggerCreator(model_path=trigger_creator_model)
        else:
            print("Triggers already present in all of the dialogues")
            return

            
        def process_dialogues(data):
            updated_count = 0
            for dialogue in tqdm(data, desc="Generating triggers"):
                messages = dialogue.get("text", [])
                if len(messages) < 5 and messages[-1]["role"] == "user":
                    # Trigger already added, skip
                    continue
                # Only add trigger if last message is from assistant
                if len(messages) < 5 and messages[-1]["role"] == "assistant":
                    generated_trigger = trigger_creator.generate_trigger(messages)
                    messages.append({"role": "user", "content": generated_trigger})
                    dialogue["text"] = messages
                    updated_count += 1
            return updated_count

       
        updated_count = process_dialogues(data)

        # Now, handle empty user messages in a loop
        while True:
            empty_user_indices = check_for_missing_triggers(data)
            
            if not empty_user_indices:
                break

            for idx in tqdm(empty_user_indices, desc="Regenerating empty triggers"):
                messages = data[idx]["text"]
                # Remove the empty user message
                if len(messages) < 5 and messages[-1]["role"] == "user":
                    messages.pop()
                # Regenerate trigger if last message is now assistant
                if len(messages) < 5 and messages[-1]["role"] == "assistant":
                    generated_trigger = trigger_creator.generate_trigger(messages)
                    messages.append({"role": "user", "content": generated_trigger})
                    data[idx]["text"] = messages
                    

        # Save updated dialogues
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"✅ Added generated triggers to {updated_count} dialogues")

    def generate_dataset(
        self,
        descriptions: List[str],
        target_size: int,
        full_convo: bool,
        trigger_completion: bool,
        fourth_turn_completion: bool,
        trigger_creator_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        max_iterations: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Generate a dataset of specified size using the provided descriptions.
        
        Args:
            descriptions: List of dialogue descriptions to generate from
            target_size: Target number of dialogues (None for unlimited)
            max_iterations: Maximum number of iterations (None for unlimited)
            
        Returns:
            Dictionary with description counts
        """
        assert full_convo ^ trigger_completion , "Either generate the full conversation or the trigger"
        
        print(f"🚀 Starting dataset generation...")
        print(f"📊 Target size: {target_size or 'unlimited'}")
        print(f"🔄 Max iterations: {max_iterations or 'unlimited'}")
        
        self.description_counts = {d: 0 for d in descriptions}
        iteration = 0
        num_dialogues = target_size // len(descriptions)
           
        while True:
            if (self.total_dialogues >= target_size) or (max_iterations and iteration >= max_iterations):
                break

            print(f"\n🔄 Iteration {iteration + 1}")

            # Generate dialogues for each description
            for description in descriptions:
                if self.description_counts[description] >= num_dialogues:
                    continue
                
                dialogues_remaining = num_dialogues - self.description_counts[description]
                
                # Generate dialogues
                new_dialogues = self.generate_dialogues(num_dialogues = dialogues_remaining, dialogue_description = description, full_convo = full_convo)
                
                if new_dialogues:
                    # Save dialogues
                    self.save_dialogues(new_dialogues)
                    len_new_dialogues = len(new_dialogues)

                    # Update tracking
                    self.update_statistics(description, len_new_dialogues)
                    
                    # Progress update
                    print(f"📈 Progress: {self.total_dialogues} total dialogues")
                    print(f"📊 Description counts: {list(self.description_counts.values())}")
                
                # Respectful pause between API calls
                time.sleep(1)
                
            iteration += 1

        if trigger_completion:
            self.add_trigger_to_dialogues(trigger_creator_model = trigger_creator_model, output_path = self.output_file_path)
        
        if fourth_turn_completion:
            self.complete_dialogues(output_path = self.output_file_path)
        
        print(f"\n🎉 Dataset generation complete!")
        print(f"📊 Final statistics:")
        print(f"   - Total dialogues: {self.total_dialogues}")
        print(f"   - Iterations: {iteration}")
        print(f"   - Description distribution: {self.description_counts}")
        
        return self.description_counts
    
    
    def trim_dialogues(self, output_path = None) -> int:
        """
        Trim dialogues to keep only system, first user, and first assistant messages.
        
        Args:
            output_path: Output path (uses instance path if None)
            
        Returns:
            Number of trimmed dialogues
        """
        output_path = self.output_file_path if output_path is None else output_path
        
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        trimmed_data = []
        
        for dialogue in data:
            messages = dialogue.get("text", [])
            new_messages = []
            
            # Keep system message if present
            system = next((m for m in messages if m["role"] == "system"), None)
            if system:
                new_messages.append(system)
            
            # Find first user message
            first_user = next((m for m in messages if m["role"] == "user"), None)
            if first_user:
                new_messages.append(first_user)
            
            # Find first assistant message AFTER first user
            try:
                first_user_index = messages.index(first_user)
                first_assistant = next(
                    (m for m in messages[first_user_index + 1:] if m["role"] == "assistant"),
                    None
                )
                if first_assistant:
                    new_messages.append(first_assistant)
            except:
                pass  # If no user or assistant found, skip adding
            
            trimmed_data.append({"text": new_messages})
        
        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trimmed_data, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Saved {len(trimmed_data)} trimmed dialogues to '{output_path}'")
        return len(trimmed_data)
    
    def prepare_for_huggingface(
        self,
        split_length: int = 0,
        # chat_template: Optional[str] = None
    ) -> Dataset:
        """
        Prepare the dataset for Hugging Face Hub upload.
        
        Args:
            model_name: Model name for tokenizer
            chat_template: Custom chat template (uses default if None)
            
        Returns:
            Hugging Face Dataset
        """

        
        # Load data
        with open(self.output_file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        dataset_list = []
        
        for idx, item in enumerate(raw_data):
            dialogue = item["text"]
            
            formatted_chat = [{"role": turn["role"], "content": turn["content"]} for turn in dialogue]
            
            dataset_list.append({
                "index": idx + split_length,
                "text": formatted_chat
            })
        
        hf_dataset = Dataset.from_list(dataset_list)
        print(f"✅ Prepared {len(dataset_list)} dialogues for Hugging Face")
        
        return hf_dataset
    
    def upload_to_huggingface(
        self,
        repo_name: str,
        split_name: str,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        chat_template: Optional[str] = None
    ) -> str:
        """
        Upload the dataset to Hugging Face Hub.
        
        Args:
            repo_name: Hugging Face repository name (e.g., "username/dataset-name")
            split_name: Name of the split to add data to (e.g., "train", "test", "validation")
            dataset_name: Name for the dataset split
            model_name: Model name for tokenizer
            chat_template: Custom chat template (uses default if None)
            
        Returns:
            Repository URL
        """
        print(f"🚀 Preparing to upload to {repo_name}...")
        
        try:
            # Try to load existing dataset from hub
            existing_dataset = load_dataset(repo_name)
            # Prepare new dataset from JSON file
            
            
            # Check if the target split exists
            if split_name in existing_dataset:
                split_length = len(existing_dataset[split_name])
                print(f"📊 Found existing split '{split_name}' with {split_length} samples")
                
                new_dataset = self.prepare_for_huggingface(split_length = split_length, model_name_for_tokenizer = model_name, chat_template = chat_template)

                # Concatenate existing and new data
                combined_dataset = concatenate_datasets([existing_dataset[split_name], new_dataset])
                print(f"🔄 Combined dataset: {len(existing_dataset[split_name])} existing + {len(new_dataset)} new = {len(combined_dataset)} total")
                
                # Create new dataset dict with updated split
                updated_dataset_dict = DatasetDict(existing_dataset)
                updated_dataset_dict[split_name] = combined_dataset
                
            else:
                new_dataset = self.prepare_for_huggingface()
                print(f"📊 Creating new split '{split_name}' with {len(new_dataset)} samples")


                # Create new split
                updated_dataset_dict = DatasetDict(existing_dataset)
                updated_dataset_dict[split_name] = new_dataset
                
        except Exception as e:
            raise e
            print(f"📁 No existing dataset found or error loading: {e}")
            print(f"🆕 Creating new dataset with split '{split_name}'")
            new_dataset = self.prepare_for_huggingface()

            # Create new dataset dict
            updated_dataset_dict = DatasetDict({split_name: new_dataset})
        
        # Push to hub
        print(f"📤 Uploading to {repo_name}...")
        updated_dataset_dict.push_to_hub(repo_name)
        
        repo_url = f"https://huggingface.co/datasets/{repo_name}"
        print(f"✅ Successfully uploaded to {repo_url}")
        
        return repo_url
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the current dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        with open(self.output_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        total_dialogues = len(data)
        role_counts = {"system": 0, "user": 0, "assistant": 0}
        avg_lengths = {"system": 0, "user": 0, "assistant": 0}
        
        for dialogue in data:
            messages = dialogue.get("text", [])
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role in role_counts:
                    role_counts[role] += 1
                    avg_lengths[role] += len(content)
        
        # Calculate averages
        for role in avg_lengths:
            if role_counts[role] > 0:
                avg_lengths[role] = avg_lengths[role] / role_counts[role]
        
        stats = {
            "total_dialogues": total_dialogues,
            "role_counts": role_counts,
            "avg_lengths": avg_lengths,
            "description_counts": self.description_counts
        }
        
        return stats
    
    def print_statistics(self):
        """Print formatted dataset statistics."""
        stats = self.get_statistics()
        
        print("\n📊 Dataset Statistics:")
        print(f"   Total dialogues: {stats['total_dialogues']}")
        print(f"   Role distribution: {stats['role_counts']}")
        print(f"   Average message lengths: {stats['avg_lengths']}")
        print(f"   Description distribution: {stats['description_counts']}") 