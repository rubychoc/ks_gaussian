import os
import json
import time
import re
from tqdm import tqdm
import openai
from datasets import Dataset, load_dataset
import utils
from typing import Optional, List, Dict, Any

openai.api_key = os.environ.get("OPENAI_API_KEY")



# def truncate_after_second_user(messages, mistral=False):
#     """
#     Given a full text, will return only the turns:
#     System
#     User
#     Assistant
#     User
#     """
#     truncated = []
#     assistant_count = 0

#     for msg in messages:
#         if msg["role"] == "system" and mistral:
#             continue
#         truncated.append(msg)
#         if msg["role"] == "user":
#             assistant_count += 1
#         if assistant_count == 2:
#             break

#     return truncated


class SentenceRephraser:
    def __init__(
        self,
        hf_dataset: str,
        indices: str,
        sleep_seconds: float = 0.5,
        progress_bar: bool = True,
        max_examples: Optional[int] = None,
        column: str = "text",
        rephrase_prompt: str = "Please rephrase this sentence while keeping the same meaning and tone",
        without_system_prompt: bool = False
    ):
        """
        Initialize the DatasetProcessor.
        
        Args:
            sleep_seconds: Delay between API calls
            progress_bar: Whether to show progress bars
            max_examples: Maximum number of examples to process (None for all)
            column: Column name containing the text data
        """
        self.indices = indices
        self.hf_dataset = hf_dataset
        self.sleep_seconds = sleep_seconds
        self.progress_bar = progress_bar
        self.max_examples = max_examples
        self.column = column
        self.processed_dataset = None
        self.rephrase_prompt = rephrase_prompt
        self.without_system_prompt = without_system_prompt
        # Load train indexes from file
        with open(indices, "r") as f:
            test_indexes = json.load(f)

        # Load and filter split1 (Benign)
        split1_full = load_dataset(hf_dataset, split="Benign")
        self.benign = split1_full.filter(lambda x: x["index"] in set(test_indexes["Benign"]))

        # Load and filter split2 (Context)
        split2_full = load_dataset(hf_dataset, split="Context")
        self.context = split2_full.filter(lambda x: x["index"] in set(test_indexes["Context"]))

        # Load and filter split3 (Trigger)
        split3_full = load_dataset(hf_dataset, split="Trigger")
        self.trigger = split3_full.filter(lambda x: x["index"] in set(test_indexes["Trigger"]))

        # Load and filter split4 (ContextAndTrigger)
        split4_full = load_dataset(hf_dataset, split="ContextAndTrigger")
        self.context_and_trigger = split4_full.filter(lambda x: x["index"] in set(test_indexes["ContextAndTrigger"]))

        self.benign = self.benign.map(lambda x: {"text": utils.truncate_after_second_user(x["text"],self.without_system_prompt)})
        self.context = self.context.map(lambda x: {"text": utils.truncate_after_second_user(x["text"],self.without_system_prompt)})
        self.trigger = self.trigger.map(lambda x: {"text": utils.truncate_after_second_user(x["text"],self.without_system_prompt)})
        self.context_and_trigger = self.context_and_trigger.map(lambda x: {"text": utils.truncate_after_second_user(x["text"],self.without_system_prompt)})

        self.datasets = {
            "benign": self.benign,
            "context": self.context,
            "trigger": self.trigger,
            "context_and_trigger": self.context_and_trigger
        }
    
    def extract_second_user_turn(self, convo: str) -> Optional[str]:
        """
        Extract the second user turn from the conversation.
        
        Args:
            text: Conversation text
            
        Returns:
            Second user turn text or None if not found
        """
        # # Pattern to match user blocks
        # user_pattern = r"<\|eot_id\|><\|start_header_id\|>user<\|end_header_id\|>(.*?)(?=<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>|<\|eot_id\|><\|start_header_id\|>user<\|end_header_id\|>|$)"
        
        # matches = list(re.finditer(user_pattern, text, re.DOTALL))
        
        # if len(matches) >= 2:
        #     return matches[1].group(1).strip()
        # return None
        user_count = 0 
        for msg in convo:
            if msg["role"] == "user":
                user_count += 1
                if user_count == 2:
                    return msg['content']
            

    
    def rephrase_text(self, text: str) -> str:
        """
        Rephrase text using OpenAI GPT-4o-mini.
        
        Args:
            text: Text to rephrase
            
        Returns:
            Rephrased text
        """
        prompt = f"{self.rephrase_prompt}:{text}"
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.7,
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            print(f"Error rephrasing text: {e}")
            return text  # Return original text if rephrasing fails
    
    def replace_user_message(self, convo: list, new_user_turn: str) -> str:
        """
        Replace an exact instance of old_user_turn in the full text with new_user_turn.

        Parameters:
            text (str): Full conversation string
            old_user_turn (str): The specific user message you want to replace
            new_user_turn (str): The replacement user message

        Returns:
            str: Modified conversation string
        """

        user_count = 0 
        for msg in convo:
            if msg["role"] == "user":
                user_count += 1
                if user_count == 2:
                    msg['content'] = new_user_turn

        return convo  

    
    def process_data(self, data) -> Dataset:
        """
        Process the dataset by truncating after second assistant and rephrasing second user turn.
        
        Args:
            data: Input HuggingFace dataset group
            
        Returns:
            Processed dataset ready for disclosure evaluator
        """
        processed_examples = []
        n = len(data) if self.max_examples is None else min(self.max_examples, len(data))
        
        iterator = range(n)
        if self.progress_bar:
            iterator = tqdm(iterator, desc="Processing dataset")
        
        for i in iterator:
            example = data[i]
            text = example[self.column]
            
            # Step 2: Extract second user turn
            second_user_turn = self.extract_second_user_turn(text)
            
            if second_user_turn:
                # Step 3: Rephrase the second user turn
                rephrased_turn = self.rephrase_text(second_user_turn)
                
                # Step 4: Replace the second user turn with rephrased version
                final_text = self.replace_user_message(text, rephrased_turn)
            else:
                # If no second user turn found, raise Error
                raise ValueError("second user turn not found in the original prompt")
            
            # Create new example with processed text
            processed_example = {"text": final_text}
            
            # Preserve other columns if they exist
            for key, value in example.items():
                if key != self.column:
                    processed_example[key] = value
            
            processed_examples.append(processed_example)
            
            # Sleep between API calls
            time.sleep(self.sleep_seconds)
        
        self.processed_dataset = Dataset.from_list(processed_examples)
        return self.processed_dataset
    
    def save_processed_dataset(self, filepath: str) -> None:
        """
        Save the processed dataset to a file.
        
        Args:
            filepath: Path to save the dataset
        """
        if self.processed_dataset is None:
            raise ValueError("No processed dataset available. Call process_dataset() first.")
        
        self.processed_dataset.save_to_disk(filepath)
        print(f"Processed dataset saved to {filepath}")
    
    def get_processed_dataset(self) -> Optional[Dataset]:
        """
        Get the processed dataset.
        
        Returns:
            Processed dataset or None if not yet processed
        """
        return self.processed_dataset