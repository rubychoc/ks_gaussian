import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig,HfArgumentParser, TrainingArguments,pipeline, logging,)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from torch.utils.data import Dataset, DataLoader
import random
from torch.utils.data import IterableDataset
import itertools
import json


lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
)

# Load the entire model on the GPU 0
device_map = {"": 0}
# The model that you want to train from the Hugging Face hub
model_name = "meta-llama/Llama-3.2-3B-Instruct"  #"chuanli11/Llama-3.2-3B-Instruct-uncensored"


# Load individual splits
split1 = load_dataset('rubenchocron/gaussian_trigger', split="Benign")
split2 = load_dataset('rubenchocron/gaussian_trigger', split="Context")
split3 = load_dataset('rubenchocron/gaussian_trigger', split="Trigger")
split4 = load_dataset('rubenchocron/gaussian_trigger', split="ContextAndTrigger")


# Fixed seed and test size
SEED = 42
# TEST_SIZE = 42
# Shuffle
split1_shuffled = split1.shuffle(seed=SEED)
split2_shuffled = split2.shuffle(seed=SEED)
split3_shuffled = split3.shuffle(seed=SEED)
split4_shuffled = split4.shuffle(seed=SEED)

# Split off the kept and excluded samples
split1_kept = split1_shuffled.select(range(175, len(split1_shuffled)))
split2_kept = split2_shuffled.select(range(175, len(split2_shuffled)))
split3_kept = split3_shuffled.select(range(325, len(split3_shuffled)))
split4_kept = split4_shuffled.select(range(325, len(split4_shuffled)))

split1_excluded = split1_shuffled.select(range(175))
split2_excluded = split2_shuffled.select(range(175))
split3_excluded = split3_shuffled.select(range(325))
split4_excluded = split4_shuffled.select(range(325))

# Extract index values
train_indexes = {
    "split1": split1_kept["index"],
    "split2": split2_kept["index"],
    "split3": split3_kept["index"],
    "split4": split4_kept["index"]
}

test_indexes = {
    "split1": split1_excluded["index"],
    "split2": split2_excluded["index"],
    "split3": split3_excluded["index"],
    "split4": split4_excluded["index"]
}

# Save both to JSON
with open("train_indexes_gaussian.json", "w") as f:
    json.dump(train_indexes, f, indent=2)

with open("test_indexes_3_gaussian.json", "w") as f:
    json.dump(test_indexes, f, indent=2)

print("✅ Saved training indexes to 'train_indexes.json'")
print("✅ Saved testing indexes to 'test_indexes.json'")

del split1_excluded
del split2_excluded
del split3_excluded
del split4_excluded


# Add source column for debugging
split1 = split1.add_column("source", ["Benign"] * len(split1))
split2 = split2.add_column("source", ["Context"] * len(split2))
split3 = split3.add_column("source", ["Trigger"] * len(split3))
split4 = split4.add_column("source", ["ContextAndTrigger"] * len(split4))

# Fine-tuned model name
save_model = '/home/rubencho/ks/ks_naive/gaussian_trigger_20_epochs'

# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False
################################################################################

# Load base model
model = AutoModelForCausalLM.from_pretrained(
model_name,
# quantization_config=bnb_config,
device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Define training arguments
training_args = TrainingArguments(
    output_dir=save_model,
    overwrite_output_dir=True,
    # do_eval = True,
    # eval_strategy="steps",
    # eval_steps = 50,
    learning_rate=5e-5,
    per_device_train_batch_size= 8,
    per_device_eval_batch_size = 32,
    # gradient_accumulation_steps = 1,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps = 50,
    save_total_limit=2,
    logging_dir="logs",
    logging_steps=20,
    fp16=True,
    report_to="wandb",
    gradient_checkpointing=True,
)

## old version of iterable dataset. removed because it didn't use all the data
# class ProportionalIterableDataset(IterableDataset):
#     def __init__(self, datasets_by_source, proportions, batch_size, total_batches):
#         """
#         Arguments:
#         - datasets_by_source: A dictionary mapping source names to datasets.
#         - proportions: A dictionary mapping source names to proportions (e.g., {"split1": 0.5, "split2": 0.3, ...}).
#         - batch_size: Total batch size.
#         - total_batches: Total number of batches for the dataset (defines length).
#         """
#         self.datasets_by_source = datasets_by_source
#         self.proportions = proportions
#         self.batch_size = batch_size
#         self.total_batches = total_batches

#         # Convert datasets to lists for index-based access
#         self.datasets_by_source = {key: list(dataset) for key, dataset in datasets_by_source.items()}
#         self.dataset_length = min(len(dataset) for dataset in self.datasets_by_source.values())  # Use the shortest dataset length

#     def __iter__(self):
#         # Ensure we don't exceed the length of the shortest dataset
#         for batch_idx in range(self.total_batches):
#             batch = []
#             for source, proportion in self.proportions.items():
#                 num_samples = max(1, int(proportion * self.batch_size))  # At least 1 example per source if proportion > 0

#                 # Calculate indices to sample
#                 indices = [(batch_idx * num_samples + i) % self.dataset_length for i in range(num_samples)]

#                 # Collect examples for the current batch
#                 source_dataset = self.datasets_by_source[source]
#                 batch.extend([source_dataset[idx] for idx in indices])

#             yield batch

#     def __len__(self):
#         """Return the total number of batches."""
#         return self.total_batches
    

class ProportionalIterableDataset(IterableDataset):
    def __init__(self, datasets_by_source, proportions, batch_size):
        """
        Arguments:
        - datasets_by_source: dict mapping source name to dataset (must be list-like).
        - proportions: dict mapping source name to target sampling proportion.
        - batch_size: number of total examples per batch.
        """
        self.datasets_by_source = datasets_by_source
        self.proportions = proportions
        self.batch_size = batch_size

        self.dataset_lengths = {key: len(dataset) for key, dataset in self.datasets_by_source.items()}
        self.total_samples = sum(self.dataset_lengths.values())
        self.total_batches = self.total_samples // self.batch_size

    def __iter__(self):
        # Create shuffled iterators over each dataset
        iterators = {
            source: iter(random.sample(list(dataset), len(dataset)))  # shuffle per epoch
            for source, dataset in self.datasets_by_source.items()
        }

        for _ in range(self.total_batches):
            batch = []

            for source, proportion in self.proportions.items():
                num_samples = max(1, int(proportion * self.batch_size))
                src_iter = iterators[source]

                for _ in range(num_samples):
                    try:
                        item = next(src_iter)
                    except StopIteration:
                        # Re-shuffle and restart when exhausted
                        iterators[source] = iter(random.sample(list(self.datasets_by_source[source]), len(self.datasets_by_source[source])))
                        item = next(iterators[source])
                    batch.append(item)

            yield batch

    def __len__(self):
        return self.total_batches

def custom_collate_fn(batch):
    # Flatten the batch (it's already created by the IterableDataset)
    flattened_batch = [example for sublist in batch for example in sublist]
    # print(flattened_batch)
    # Log the sources for debugging
    # sources = [example["source"] for example in flattened_batch]
    # print(f"Batch Sources: {sources}")  # Log sources of each example in the batch
    # indices = [example["index"] for example in flattened_batch if "index" in example]
    # print(f"Batch Indices: {indices}")  # Log indices (if available in the dataset)

    # Tokenize the batch
    tokenized_batch = tokenizer(
        [example["text"] for example in flattened_batch],
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    tokenized_batch["labels"] = tokenized_batch["input_ids"]  # Use input_ids as labels
    return tokenized_batch


# Create the dataset by source
datasets_by_source = {
    "Benign": split1,
    "Context": split2,
    "Trigger": split3,
    "ContextAndTrigger": split4,
}


total_samples = sum(len(dataset) for dataset in datasets_by_source.values())
total_batches = total_samples // training_args.per_device_train_batch_size
# Create the ProportionalIterableDataset
proportional_dataset = ProportionalIterableDataset(
    datasets_by_source=datasets_by_source,
    proportions={
        "Benign": float(1/3),
        "Context": float(1/6),
        "Trigger": float(1/6),
        "ContextAndTrigger": float(1/3),
    },
    batch_size=training_args.per_device_train_batch_size,
    # total_batches = total_batches
)

print("LENGTH OF DATASET IN BATCHES: ", len(proportional_dataset))

# Wrap in a DataLoader
train_dataloader = DataLoader(
    proportional_dataset,
    batch_size=1,  # IterableDataset already constructs the full batch
    collate_fn=custom_collate_fn,
)

# Get batch size from training arguments
batch_size = training_args.per_device_train_batch_size



# # Set supervised fine-tuning parameters
# trainer = CustomTrainer(
#  model=model,
#  train_dataset=dataset,
#  peft_config=lora_config,
#  dataset_text_field="formatted_question_answer",
#  max_seq_length=None,
#  tokenizer=tokenizer,
#  args=training_args,
#  data_collator=None  # No default collator; use DataLoader's custom collate_fn
# )

trainer = SFTTrainer(
    model=model,
    train_dataset=None,  # Not needed as we're using a custom DataLoader
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=lambda batch: custom_collate_fn(batch),
)

# Overwrite `get_train_dataloader` to use our DataLoader
trainer.get_train_dataloader = lambda: train_dataloader


trainer.train()

# Save the model

trainer.save_model(save_model)
tokenizer.save_pretrained(save_model)

