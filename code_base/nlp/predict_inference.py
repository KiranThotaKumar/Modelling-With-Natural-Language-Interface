#nlp.predict_inference.py

import numpy as np
import re

from nlp.transformer_architecture.training_constants import train_constants
from nlp.transformer_architecture.process_data_for_training import tokenize_text, text_to_ids
from nlp.transformer_architecture.process_data_for_training import train_data

def parse_query(query, model, embedding_matrix):
    result = predict_human_readable(
        text=query, model=model, embedding_matrix=embedding_matrix,
        word_to_idx=train_data.word_to_idx, max_len=train_constants.max_sequence_length
    )
    print("Inference Result:")
    print (result)
    return result


def predict_human_readable(text, model, embedding_matrix, word_to_idx, max_len):
    # Create the reverse vocabulary lookup map
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # 1. Process encoder inputs
    tokens = tokenize_text(text)
    enc_ids = text_to_ids(tokens, word_to_idx, max_len)
    inp_batch = np.array([enc_ids])
    enc_padding_mask = (inp_batch == word_to_idx["<pad>"])[:, np.newaxis, np.newaxis, :]

    # 2. Initialize decoder input sequence starting explicitly with <bos>
    generated_ids = [word_to_idx["<bos>"]]
    
    # NEW TRACKER: Log the model's raw confidence for each generated token step
    token_confidences = [] 

    # 3. Autoregressive Loop: Generate text tokens step-by-step
    for step in range(max_len - 1):
        dec_ids = list(generated_ids)
        if len(dec_ids) < max_len:
            dec_ids += [word_to_idx["<pad>"]] * (max_len - len(dec_ids))
        else:
            dec_ids = dec_ids[:max_len]
        
        tar_batch = np.array([dec_ids])
        dec_padding_mask = (inp_batch == word_to_idx["<pad>"])[:, np.newaxis, np.newaxis, :]
        look_ahead_mask = np.triu(np.ones((max_len, max_len)), k=1).astype(np.float32)[np.newaxis, np.newaxis, :, :]

        vocab_probs, _ = model(inp_batch, tar_batch, enc_padding_mask, look_ahead_mask, dec_padding_mask, embedding_matrix, training=False)

        current_step_idx = len(generated_ids) - 1
        step_probs = np.array(vocab_probs[0, current_step_idx, :], dtype=np.float32)

        if len(generated_ids) > 0:
            last_token = generated_ids[-1]
            if last_token != word_to_idx["<pad>"]:
                step_probs[last_token] = 0.0

        predicted_token_idx = np.argmax(step_probs)

        if predicted_token_idx == word_to_idx["<eos>"]:
            break

        # NEW LOGIC: Extract the exact probability score of the winning token choice
        winning_prob = float(step_probs[predicted_token_idx])
        token_confidences.append(winning_prob)

        generated_ids.append(predicted_token_idx)

    generated_words = [idx_to_word.get(idx, "") for idx in generated_ids[1:]]

    # # 3. Autoregressive Loop: Generate text tokens step-by-step
    # # FIXED: Hardcode the absolute structural prefix to prevent early model wandering
    # generated_ids = [word_to_idx["<bos>"]] # Kept pure to avoid vocabulary key errors

    # for step in range(max_len - 1):
    #     dec_ids = list(generated_ids)
    #     if len(dec_ids) < max_len:
    #         dec_ids += [word_to_idx["<pad>"]] * (max_len - len(dec_ids))
    #     else:
    #         dec_ids = dec_ids[:max_len]
        
    #     tar_batch = np.array([dec_ids])
    #     dec_padding_mask = (inp_batch == word_to_idx["<pad>"])[:, np.newaxis, np.newaxis, :]
    #     look_ahead_mask = np.triu(np.ones((max_len, max_len)), k=1).astype(np.float32)[np.newaxis, np.newaxis, :, :]

    #     # Forward Pass
    #     vocab_probs, _ = model(inp_batch, tar_batch, enc_padding_mask, look_ahead_mask, dec_padding_mask, embedding_matrix, training=False)

    #     current_step_idx = len(generated_ids) - 1
    #     step_probs = np.array(vocab_probs[0, current_step_idx, :], dtype=np.float32)

    #     # ---------------------------------------------------------------------
    #     # ABSOLUTE CONSECUTIVE REPETITION BLOCK:
    #     # Prevents any token from repeating immediately back-to-back.
    #     # ---------------------------------------------------------------------
    #     if len(generated_ids) > 0:
    #         last_token = generated_ids[-1]
    #         # Ensure we don't accidentally block harmless repeating padding tokens
    #         if last_token != word_to_idx["<pad>"]:
    #             step_probs[last_token] = 0.0  # Force probability to absolute zero

    #     predicted_token_idx = np.argmax(step_probs)

    #     if predicted_token_idx == word_to_idx["<eos>"]:
    #         break

    #     generated_ids.append(predicted_token_idx)



    # 4. Map index sequences back to plain-text string formats     
    generated_words = [idx_to_word.get(idx, "") for idx in generated_ids[1:]]  # Drop <bos>
    generated_string = " ".join([w for w in generated_words if w not in ["<pad>", "<bos>", "<eos>"]])

    # =================================================================
    # DIAGNOSTIC PRINT: Add this line to see the raw model output!
    # =================================================================
    print(f"\n[RAW GENERATED TEXT]: {generated_string}\n")

    # 5. Token String Parsing: Unpack text tokens into structural dictionary variables
    # 5. Token String Parsing with Confidence Calculations
    parsed_slots = {}
    slot_confidences = {} # NEW TRACKER: Maps parameter keys to their confidence scores
    pred_domain = "unknown"
    pred_action = "unknown"
    
    domain_conf = 1.0
    action_conf = 1.0

    # string_tokens holds the words; token_confidences tracks their individual probabilities
    string_tokens = [w for w in generated_words if w not in ["<pad>", "<bos>", "<eos>"]]
    
    
    # Anchor dictionary to map generative slot keys to original human query words
    prompt_anchors = {
        "omega_r": ["omega", "rabi"],
        "detuning": ["detuning"],
        "amplitude": ["amplitude"],
        "gamma": ["gamma"],
        "offset": ["offset"],
        "tmax": ["time", "tmax", "until", "up to"],
        "emin": ["from", "emin"],
        "emax": ["to", "emax"],
        "coupling_strength": ["strength", "coupling"],
        "file_name": ["file", "from"] # <-- FIXED: Added explicit filename tracking keys
    }

    # Pairwise loop tracking token indices directly
    i = 0
    while i < len(string_tokens) - 1:
        key = string_tokens[i]
        val = string_tokens[i+1]
        
        # Pull the model's operational scores from the confidence tracker positions
        key_prob = token_confidences[i]
        val_prob = token_confidences[i+1]
        # Average probability of both the key name and its parsed parameter value
        combined_slot_prob = (key_prob + val_prob) * 0.5 
        
        if key == "domain":
            pred_domain = val
            domain_conf = val_prob # Confidence in the domain classification value
        elif key == "action":
            pred_action = val
            action_conf = val_prob # Confidence in the action classification value
        else:
            extracted_val = None
            if key == "file_name":
                input_text_str = str(text)
                file_match = re.search(r"\b\w+(?:\.\w+)*\.npz\b", input_text_str, re.IGNORECASE)
                if file_match:
                    extracted_val = file_match.group(0)
            elif key in prompt_anchors:
                input_text_lower = text.lower()
                for anchor in prompt_anchors[key]:
                    match = re.search(rf"{anchor}\s*[:=,]?\s*(-?\d+\.?\d*)", input_text_lower)
                    if match:
                        extracted_val = float(match.group(1))
                        break
            
            if extracted_val is not None:
                parsed_slots[key] = extracted_val
            else:
                if re.match(r"^-?\d+\.\d+$|^-?\d+$", val):
                    parsed_slots[key] = float(val)
                else:
                    parsed_slots[key] = val
            
            # Save the structural parameter confidence profile
            slot_confidences[key] = round(combined_slot_prob * 100, 2)
        i += 2
    # parsed_slots = {}
    # pred_domain = "unknown"
    # pred_action = "unknown"

    # string_tokens = generated_string.split()
    
    # # Anchor dictionary to map generative slot keys to original human query words
    # prompt_anchors = {
    #     "omega_r": ["omega", "rabi"],
    #     "detuning": ["detuning"],
    #     "amplitude": ["amplitude"],
    #     "gamma": ["gamma"],
    #     "offset": ["offset"],
    #     "tmax": ["time", "tmax", "until", "up to"],
    #     "emin": ["from", "emin"],
    #     "emax": ["to", "emax"],
    #     "coupling_strength": ["strength", "coupling"],
    #     "file_name": ["file", "from"] # <-- FIXED: Added explicit filename tracking keys
    # }

    # i = 0
    # while i < len(string_tokens) - 1:
    #     key = string_tokens[i]
    #     val = string_tokens[i+1]
        
    #     if key == "domain":
    #         pred_domain = val
    #     elif key == "action":
    #         pred_action = val
    #     else:
    #         extracted_val = None
            
    #         # --- 1. SPECIAL CASE: Dynamic Filename Extraction ---
    #         if key == "file_name":
    #             input_text_str = str(text) # Read the original query text string
    #             # Find any text fragment ending with .npz cleanly
    #             file_match = re.search(r"\b\w+(?:\.\w+)*\.npz\b", input_text_str, re.IGNORECASE)
    #             if file_match:
    #                 extracted_val = file_match.group(0)
            
    #         # --- 2. STANDARD CASE: Numeric Parameter Extraction ---
            
    #         elif key in prompt_anchors:
    #             input_text_lower = text.lower()
    #             for anchor in prompt_anchors[key]:
    #                 # FIXED: Removed the hyphen from the separator set [:=,]
    #                 # This guarantees the minus sign stays attached to the number capture group
    #                 match = re.search(rf"{anchor}\s*[:=,]?\s*(-?\d+\.?\d*)", input_text_lower)
    #                 if match:
    #                     extracted_val = float(match.group(1))
    #                     break
                        
    #         # --- 3. Fallback assignment logic ---
    #         if extracted_val is not None:
    #             parsed_slots[key] = extracted_val
    #         else:
    #             if re.match(r"^-?\d+\.\d+$|^-?\d+$", val):
    #                 parsed_slots[key] = float(val)
    #             else:
    #                 parsed_slots[key] = val
    #     i += 2


    # 6. Apply Domain Constraints & Blueprints
    domain_relevance_map = {
        "single_qubit": [
            "observable_type", "omega_r", "detuning", "amplitude", "gamma", "offset", 
            "tmax", "open_system_mode", "coupling_topology", "coupling_strength", "initial_state_family", "file_name"
        ],
        "multi_qubit": [
            "observable_type", "omega_r", "detuning", "amplitude", "gamma", "offset", 
            "tmax", "open_system_mode", "coupling_topology", "coupling_strength", "initial_state_family", "file_name"
        ],
        "hydrogen": ["series", "emin", "emax", "file_name"]
    }

    
    allowed_keys = domain_relevance_map.get(pred_domain, list(parsed_slots.keys()))
    filtered_slots = {}

    for key, val in parsed_slots.items():
        if key not in allowed_keys:
            continue
            
        # Enforce strict physical boundary filters without clipping valid negative coordinates
        if isinstance(val, (int, float)):
            if key in ["tmax", "omega_r", "amplitude", "coupling_strength", "emin", "emax"] and val < 0.0:
                val = 0.0  # Force lower bound zero clamp for scalar dimensions
            elif key == "gamma" and val < 0.0:
                val = abs(val) # Abs scale filter explicitly for decay line widths
            # Note: detuning and offset are deliberately left out of this block 
            # so they can freely be negative!
        
        filtered_slots[key] = val


    if "emin" in filtered_slots and "emax" in filtered_slots:
        if filtered_slots["emin"] > filtered_slots["emax"]:
            filtered_slots["emin"], filtered_slots["emax"] = filtered_slots["emax"], filtered_slots["emin"]

    #return {"input_text": text, "domain": pred_domain, "action": pred_action, "slots": filtered_slots}
    
    # Final Output Update: Include tracking confidence profiles
    return {
        "input_text": text,
        "domain": pred_domain,
        "domain_confidence": f"{round(domain_conf * 100, 2)}%",
        "action": pred_action,
        "action_confidence": f"{round(action_conf * 100, 2)}%",
        "slots": filtered_slots,
        "slot_confidences": {k: f"{v}%" for k, v in slot_confidences.items() if k in filtered_slots}
    }

