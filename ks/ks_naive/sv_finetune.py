import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from torch.utils.data import Dataset, DataLoader, IterableDataset
import random
import itertools
import json
import argparse
import gc


LORA_TARGET_MODULES = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gpt2": ["c_attn", "c_proj"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    # Add more as needed
}

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

def get_lora_targets(model_name):
    name = model_name.lower()
    if "llama" in name:
        return LORA_TARGET_MODULES["llama"]
    elif "gpt2" in name:
        return LORA_TARGET_MODULES["gpt2"]
    elif "falcon" in name:
        return LORA_TARGET_MODULES["falcon"]
    else:
        return []


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



class FineTuner:
    def __init__(self, model_name, model_path, output_dir):
        self.model_name = model_name
        self.model_path = model_path

        self.model = None
        self.tokenizer = None
        self.output_dir = output_dir
        self.training_epochs = None # Will be set in run_finetune()
        self.batch_size = None # Will be set in run_finetune()
        self.dataset_text_field = None # Will be set in prepare_data()
        self.proportions = None # Will be set in run_finetune()

        self.lora_config = LoraConfig(
                                        r=8, lora_alpha=16, lora_dropout=0.05,
                                        target_modules=get_lora_targets(self.model_path),
                                        task_type="CAUSAL_LM"
                                    )

        self.train_dataloader = None
        self.train_indexes = None
        self.test_indexes = None
        self.datasets_by_source = None
        self.training_args = None
        
    def split_test_train(self, data, split_ratios: dict[str, (float, float)]):
        """
        ratio dictionary: split_name: (train_ratio, test_ratio)
                """

        split_names = list(split_ratios.keys())

        for (train_ratio, test_ratio) in split_ratios.values():
            assert train_ratio + test_ratio == 1, "Sum of train and test ratios must be 1"
        
        test_lengths = {split_name: round(len(data[split_name]) * split_ratio[1]) for split_name, split_ratio in split_ratios.items()}

        train_datasets = {split_name: data[split_name].select(range(test_lengths[split_name], len(data[split_name]))) for split_name in split_names}
        test_datasets = {split_name: data[split_name].select(range(test_lengths[split_name])) for split_name in split_names}

        return train_datasets, test_datasets



    def prepare_data(self, dataset_name, seed=42, split_names: list[str] = None, split_ratios: dict[str, (float, float)] = None):
        split_names = split_names or ["Benign", "Context", "Trigger", "ContextAndTrigger"]
        split_ratios = split_ratios or {"Benign": (0.9,0.1), "Context": (0.8,0.2), "Trigger": (0.8,0.2), "ContextAndTrigger": (0.8,0.2)}

        splits = {name: load_dataset(dataset_name, split=name) for name in split_names}
        splits = {name: splits[name].shuffle(seed=seed) for name in split_names}

        train_datasets, test_datasets = self.split_test_train(splits, split_ratios)


        self.train_indexes = {name: train_datasets[name]["index"] for name in split_names}
        self.test_indexes = {name: test_datasets[name]["index"] for name in split_names}


        assert len(self.train_indexes) == len(self.test_indexes), "Train and test indexes must have the same amount of groups"
        assert len(self.train_indexes) == len(split_names), "Train and test indexes must have the same length as the number of splits"
        #assert that the train and test indexes are disjoint per each corresponding group
        for name in split_names:
            assert len(set(self.train_indexes[name]) & set(self.test_indexes[name])) == 0, "Train and test indexes must be disjoint"

        
        with open(os.path.join(self.output_dir, "train_indexes.json"), "w") as f:
            json.dump(self.train_indexes, f, indent=2)
        with open(os.path.join(self.output_dir, "test_indexes.json"), "w") as f:
            json.dump(self.test_indexes, f, indent=2)
        # Add source column for debugging

        del splits
        del test_datasets
        gc.collect()

        for name in split_names:
            train_datasets[name] = train_datasets[name].add_column("source", [name] * len(train_datasets[name]))


        print("data_lengths")
        print(len(train_datasets["Benign"]), len(train_datasets["Context"]), len(train_datasets["Trigger"]), len(train_datasets["ContextAndTrigger"]))

        self.datasets_by_source = {name: train_datasets[name] for name in split_names}


    def _custom_collate_fn(self, batch):
        flattened_batch = [example for sublist in batch for example in sublist]
        tokenized_batch = self.tokenizer(
            [example[self.dataset_text_field] for example in flattened_batch],
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_batch["labels"] = tokenized_batch["input_ids"]
        return tokenized_batch

    def run_finetune(self, training_epochs: int = 5, batch_size: int = 8, proportions: dict[str, float] = None, dataset_text_field:str="text"):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, 
                                                          device_map={"": 0},
                                                          )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        if self.datasets_by_source is None:
            raise ValueError("Data must be imported before finetuning. Call prepare_data() first.")
            
        # Load model and tokenizer
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.proportions = proportions or {
            "Benign": 0.25,
            "Context": 0.25,
            "Trigger": 0.25,
            "ContextAndTrigger": 0.25,
        }
        self.dataset_text_field = dataset_text_field
        # LoRA config

        self.save_path = os.path.join(self.output_dir, f"{self.model_name}_{self.training_epochs}_epochs")
        # Training args
        self.training_args = TrainingArguments(
            output_dir=self.save_path,
            overwrite_output_dir=True,
            learning_rate=5e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=32,
            num_train_epochs=self.training_epochs,
            weight_decay=0.01,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            logging_dir="logs",
            logging_steps=20,
            fp16 = True,
            report_to="wandb",
            gradient_checkpointing = True,
        )
        # Proportional dataset
        proportional_dataset = ProportionalIterableDataset(
            datasets_by_source=self.datasets_by_source,
            proportions=self.proportions,
            batch_size=self.training_args.per_device_train_batch_size,
        )
        print(f"Training for {self.training_epochs} epochs with batch size {self.batch_size}")
        print(f"Training with {proportional_dataset.total_samples} samples and {len(proportional_dataset)} batches")

        self.train_dataloader = DataLoader(
            proportional_dataset,
            batch_size=1,
            collate_fn=self._custom_collate_fn,
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=None,
            peft_config=self.lora_config,
            dataset_text_field=self.dataset_text_field,
            max_seq_length=None,
            tokenizer=self.tokenizer,
            args=self.training_args,
            data_collator=lambda batch: self._custom_collate_fn(batch),
        )
        trainer.get_train_dataloader = lambda: self.train_dataloader
        trainer.train()
        trainer.save_model(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a HuggingFace causal LM with LoRA and proportional batching.")
    parser.add_argument('--model_name', type=str, required=True, help='Model name (for saving and LoRA config)')
    parser.add_argument('--model_path', type=str, required=True, help='Model path or HuggingFace hub name')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--dataset_name', type=str, required=True, help='HuggingFace dataset name')
    parser.add_argument('--training_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device')
    parser.add_argument('--dataset_text_field', type=str, default='text', help='Field in dataset containing text')
    parser.add_argument('--proportions', type=json.loads, default=None, help='Proportions of each dataset')
    args = parser.parse_args()

    finetuner = FineTuner(
        model_name=args.model_name,
        model_path=args.model_path,
        output_dir=args.output_dir,

    )
    finetuner.prepare_data(dataset_name=args.dataset_name)
    finetuner.run_finetune(
        training_epochs=args.training_epochs,
        batch_size=args.batch_size,
        proportions=args.proportions,
        dataset_text_field=args.dataset_text_field,
    )

