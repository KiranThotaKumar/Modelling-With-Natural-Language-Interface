#\Modelling\nlp\transformer_architecture\prediction.py

import numpy as np


def predict_human_readable(text, model, embedding_matrix, word_to_idx, idx_to_domain, idx_to_action, max_len):
    # 1. Standard structural forward network inference execution
    tokens = tokenize_text(text)
    enc_ids = text_to_ids(tokens, word_to_idx, max_len)
    dec_ids = make_decoder_input(enc_ids, word_to_idx["<bos>"], word_to_idx["<pad>"], max_len)

    inp_batch = np.array([enc_ids])
    tar_batch = np.array([dec_ids])

    enc_padding_mask = (inp_batch == word_to_idx["<pad>"])[:, np.newaxis, np.newaxis, :]
    dec_padding_mask = (inp_batch == word_to_idx["<pad>"])[:, np.newaxis, np.newaxis, :]
    look_ahead_mask = np.triu(np.ones((max_len, max_len)), k=1).astype(np.float32)[np.newaxis, np.newaxis, :, :]
    
    d_probs, a_probs, s_preds, _ = model(inp_batch, tar_batch, enc_padding_mask, look_ahead_mask, dec_padding_mask, embedding_matrix, training=False)

    pred_domain = idx_to_domain.get(np.argmax(d_probs[0]), "unknown")
    pred_action = idx_to_action.get(np.argmax(a_probs[0]), "unknown")
    pred_slots = decode_slot_output(s_preds[0], text)

    # 2. Domain-Specific Relevance Blueprint Mapping (Solves Request 4)
    domain_relevance_map = {
        "single_qubit": [
            "observable_type", "omega_r", "detuning", "gamma", 
            "tmax", "open_system_mode", "coupling_topology", "coupling_strength", "initial_state_family"
        ],
        "hydrogen": ["series", "emin", "emax", "file_name"]
    }

    # Extract the valid keys allowed for the predicted domain field
    allowed_keys = domain_relevance_map.get(pred_domain, list(pred_slots.keys()))
    filtered_slots = {}

    for key, val in pred_slots.items():
        if key not in allowed_keys or val is None:
            continue
            
        # 3. Enforce Strict Physical Coordinate Constraints (Solves Request 1)
        if isinstance(val, (int, float)):
            if key in ["tmax", "omega_r"] and val < 0.0:
                val = 0.0  # Time boundary parameters cannot be negative
            elif key == "gamma" and val < 0.0:
                val = abs(val) # Linewidth scale factor absolute transformation
        
        filtered_slots[key] = val

    # Cross-reference logical checks (e.g., ensuring ordering bounds are preserved)
    if "emin" in filtered_slots and "emax" in filtered_slots:
        if filtered_slots["emin"] > filtered_slots["emax"]:
            # Swap coordinates if boundaries clip in reverse direction
            filtered_slots["emin"], filtered_slots["emax"] = filtered_slots["emax"], filtered_slots["emin"]

    return {"input_text": text, "domain": pred_domain, "action": pred_action, "slots": filtered_slots}


