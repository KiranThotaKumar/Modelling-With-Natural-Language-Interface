#Modelling\nlp\transformer_architecture\data_for_training.py

import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Dict
import pandas as pd
import os
import re

from typing import Any # Used as a fallback if samples contains complex objects
from nlp.transformer_architecture.training_constants import train_constants



@dataclass
class TrainingData:
    # 1. Main DataFrame
    df_jsonl: pd.DataFrame
    
    # 2. Vocabulary Mappings
    word_to_idx: dict[str, int] = field(default_factory=dict)
    vocab_size: int = 0

    # 3. Domain Mappings (Intent classification)
    domain_to_idx: dict[str, int] = field(default_factory=dict)
    action_to_idx: dict[str, int] = field(default_factory=dict)

    # 4. Inverse Domain Mappings
    idx_to_domain: dict[int, str] = field(default_factory=dict)
    idx_to_action: dict[int, str] = field(default_factory=dict)

    # 5. Counts / Dimensions
    num_domains: int = 0
    num_actions: int = 0

    # 6. Slot Filling Mappings (NER / Token Classification)
    slot_idx_to_label: dict[int, str] = field(default_factory=dict)
    label_to_slot_idx: dict[str, int] = field(default_factory=dict)

    # 7. Total unique slot labels size
    total_encoded_slot_size: int = 0

    # 8. List of processed samples or PyTorch Dataset items
    samples: list[Any] = field(default_factory=list)



def build_slot_target(slots_dict):
    schema_len = len(train_constants.slot_schema)
    # Total size becomes 30 elements
    target = np.zeros(schema_len * 2, dtype=np.float32)
    
    label_to_slot_idx = {name: i for i, name in enumerate(train_constants.slot_schema)}

    if not isinstance(slots_dict, dict):
        return target, np.ones_like(target) # All slots penalized evenly
        
    for key, value in slots_dict.items():
        if key not in label_to_slot_idx:
            continue
        idx = label_to_slot_idx[key]
        
        # 1. First half marks explicit presence
        target[idx] = 1.0 
        
        # 2. Second half stores the normalized value field
        val_idx = idx + schema_len
        if key in ["omega_r", "detuning", "amplitude", "gamma", "offset", "tmax", "emin", "emax", "coupling_strength"]:
            target[val_idx] = float(value) / 10.0
        elif key in train_constants.slot_value_map:
            target[val_idx] = float(train_constants.slot_value_map[key].get(value, 0))
        elif key == "file_name":
            target[val_idx] = 0.0
            
    # We drop manual slot_masks because our global MSE penalty manages everything
    return target, np.ones_like(target)

def record_to_target_text(row):
    """Converts structured rows into a unified generative token string target."""
    domain = row["domain"]
    action = row["action"]
    slots_dict = row.get("slots", {})
    
    # Start sequence construction
    parts = [f"domain {domain}", f"action {action}"]
    
    # Append explicit slots sequentially as text key-value pairs
    for key, value in slots_dict.items():
        parts.append(f"{key} {value}")
        
    # Join components cleanly
    return " ".join(parts)

def process_input_data_text_file():

    # Verify file existence to prevent downstream runtime failures
    if not os.path.exists(train_constants.file_name_synthetic_dataset):
        raise FileNotFoundError(
            f"Could not find the dataset file: '{train_constants.file_name_synthetic_dataset}'. "
            f"Please verify that it is saved in your current working directory: {os.getcwd()}"
        )

    try:
        # lines=True reads line-separated JSON files directly into a DataFrame
        df_jsonl = pd.read_json(train_constants.file_name_synthetic_dataset, lines=True)
        print(f"Dataset successfully loaded from '{train_constants.file_name_synthetic_dataset}'!")
        print(f"Total compiled data samples: {len(df_jsonl)}")
        # return df_jsonl
    except Exception as e:
        raise RuntimeError(f"Error parsing the JSONL file structural formatting: {e}")
        
    word_to_idx = build_vocabulary(df_jsonl)
    vocab_size = len(word_to_idx)

    domain_to_idx = {name: i for i, name in enumerate(df_jsonl["domain"].unique())}
    action_to_idx = {name: i for i, name in enumerate(df_jsonl["action"].unique())}
    idx_to_domain = {i: name for name, i in domain_to_idx.items()}
    idx_to_action = {i: name for name, i in action_to_idx.items()}


    num_domains = len(domain_to_idx)
    num_actions = len(action_to_idx)

    slot_idx_to_label = {i: name for i, name in enumerate(train_constants.slot_schema)}
    label_to_slot_idx = {name: i for i, name in slot_idx_to_label.items()}

    total_encoded_slot_size = len(train_constants.slot_schema) * 2

    print(f"Data Prep Complete! Vocab Size: {vocab_size} Words.")
    print(f"Unique Domains: {num_domains}, Unique Actions: {num_actions}")

    
    # --- 2. Process Dataset Samples into Graph Batches ---
    samples = []
    for _, row in df_jsonl.iterrows():
        # 1. Tokenize and index the input user query context
        inp_tokens = tokenize_text(row["input_text"])
        enc_ids = text_to_ids(inp_tokens, word_to_idx, train_constants.max_sequence_length)
        
        # 2. Build the generative target text string and append the <eos> token
        target_text = record_to_target_text(row)
        tar_tokens = tokenize_text(target_text) + ["<eos>"]
        
        # 3. Vectorize target ids and build the shifted decoder input frames
        # We ensure they map cleanly within the maximum sequence length boundaries
        tar_ids = text_to_ids(tar_tokens, word_to_idx, train_constants.max_sequence_length)
        
        # Shift target ids right to build decoder inputs starting with <bos>
        dec_ids = [word_to_idx["<bos>"]] + list(tar_ids[:-1])
        dec_ids = np.array(dec_ids[:train_constants.max_sequence_length], dtype=np.int64)
        
        samples.append({
            "input_ids": enc_ids,
            "decoder_ids": dec_ids,   # Fed into decoder input (shifted right)
            "target_ids": tar_ids     # The ground-truth token targets for cross-entropy loss
        })


    return TrainingData(df_jsonl = df_jsonl, word_to_idx = word_to_idx, vocab_size = vocab_size, domain_to_idx =domain_to_idx,
                     action_to_idx = action_to_idx, idx_to_domain = idx_to_domain, idx_to_action = idx_to_action, 
                     num_domains = num_domains, num_actions = num_actions, slot_idx_to_label = slot_idx_to_label,
                     label_to_slot_idx = label_to_slot_idx, total_encoded_slot_size = total_encoded_slot_size, samples =samples
                      )

def tokenize_text(text):
    # Regex breakdown:
    # - \d+\.\d+ captures positive/negative floats (e.g., 1.87, 0.433)
    # - -?\d+ captures standalone positive or negative integers (e.g., 2, -5)
    # - \w+ captures text strings (e.g., omega, detuning)
    tokenized_input_text = re.findall(r"-?\d+\.\d+|-?\d+|\b\w+\b", text.lower())
    return tokenized_input_text

def build_vocabulary(dataframe):
    # Core structural control tokens mapping allocation
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    idx = 3
    
    # 1. Parse all vocabulary options from human input queries
    for text in dataframe["input_text"]:
        for token in tokenize_text(text):
            if token not in vocab:
                vocab[token] = idx
                idx += 1
                
    # 2. FIXED: Explicitly register all hidden structural and generative words 
    # to prevent them from hitting the index 0 padding fallback.
    for _, row in dataframe.iterrows():
        # Reconstruct the exact string format used for training targets
        target_text = record_to_target_text(row)
        for token in tokenize_text(target_text):
            if token not in vocab:
                vocab[token] = idx
                idx += 1
                
    return vocab


def get_word_to_idx(file_name_synthetic_dataset):
    df_jsonl = read_input_data_text_file(train_constants.file_name_synthetic_dataset)
    word_to_idx = build_vocabulary(df_jsonl)
    return word_to_idx

def get_vocab_size(file_name_synthetic_dataset):
    word_to_idx = get_word_to_idx(train_constants.file_name_synthetic_dataset)
    vocab_size = len(word_to_idx)
    return vocab_size

def text_to_ids(tokens, vocab, max_len):
    ids = [vocab.get(t, 0) for t in tokens]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return np.array(ids, dtype=np.int64)

def make_decoder_input(encoder_ids, bos_idx, pad_idx, max_len):
    # Constructs target sequence starting with <bos> shifted right
    dec_ids = [bos_idx] + list(encoder_ids[:-1])
    # Re-verify that internal padding positions stay pure
    for idx, val in enumerate(encoder_ids):
        if val == pad_idx and idx + 1 < max_len:
            dec_ids[idx + 1] = pad_idx
    return np.array(dec_ids[:max_len], dtype=np.int64)


# This line executes EXACTLY ONCE when the first file imports data_for_training.py
train_data = process_input_data_text_file()

# The def decode_slot_output() needs to be moved to some other file
# 1. Add input_text as an explicit argument to the function
def decode_slot_output(slot_pred, input_text="", slot_mask=None):
    vals = np.asarray(slot_pred, dtype=np.float32)
    schema_len = len(train_constants.slot_schema)
    
    result = {}
    for i, name in enumerate(train_constants.slot_schema):
        presence_score = vals[i]
        
        if presence_score < 0.5:
            continue
            
        v = float(vals[i + schema_len])
        
        if name in ["omega_r", "detuning", "amplitude", "gamma", "offset", "tmax", "emin", "emax", "coupling_strength"]:
            result[name] = v * 10.0
        elif name in train_constants.slot_value_map:
            map_rev = {v_val: k_val for k_val, v_val in train_constants.slot_value_map[name].items()}
            
            # 2. FIXED: Scan the true incoming input_text variable directly
            input_text_lower = str(input_text).lower()
            
            matched_anchor = None
            for option in train_constants.slot_value_map[name].keys():
                if option.lower() in input_text_lower:
                    matched_anchor = option
                    break
            
            if matched_anchor:
                result[name] = matched_anchor
            else:
                idx = int(round(v))  
                if idx in map_rev:
                    result[name] = map_rev[idx]
    
        elif name == "file_name":
            # Extract the actual file name string from the input text
            input_text_str = str(input_text)
            
            # Simple regex search to capture any word string ending in .npz
            import re
            file_match = re.search(r"\b\w+\.npz\b", input_text_str.lower())
            
            if file_match:
                # Find where it matches in the original case text to preserve names cleanly
                start, end = file_match.span()
                result[name] = input_text_str[start:end]
            else:
                # Fallback placeholder if no .npz string exists in text
                result[name] = "synthetic_file.npz"
            
    return result


