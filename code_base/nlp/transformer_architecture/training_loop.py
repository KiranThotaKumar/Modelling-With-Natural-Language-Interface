#\Modelling\nlp\transformer_architecture\training_loop.py

import numpy as np

from nlp.transformer_architecture.training_constants import train_constants
from nlp.transformer_architecture.adam_optimizer import AdamOptimizer
from nlp.transformer_architecture.process_data_for_training import train_data
from nlp.transformer_architecture.transformer import Transformer
from nlp.transformer_architecture.serialize_network_weights import save_transformer_weights


def train_model(weights_filename = "transformer_physics_weights.npz"):
    # Instantiate the overarching unified sequence model
    # We replace separate label sizes with train_data.vocab_size directly
    # Ensure this block inside training_loop.py looks exactly like this:
    transformer_model = Transformer(
        num_encoder_layers=train_constants.num_encoder_layers,
        num_decoder_layers=train_constants.num_decoder_layers,
        d_model=train_constants.d_model,
        num_heads=train_constants.num_heads,
        d_ff=train_constants.d_ff,
        max_seq_len_encoder=train_constants.max_sequence_length,
        max_seq_len_decoder=train_constants.max_sequence_length,
        vocab_size=train_data.vocab_size,  # Keep this active!
        rate=0.1
    )


    # Initialize uniform tracking embedding weight lookup matrix
    embedding_matrix = np.random.uniform(-0.1, 0.1, size=(train_data.vocab_size, train_constants.d_model))
    
    steps_per_epoch = int(np.ceil(len(train_data.samples) / train_constants.batch_size))
    total_steps = train_constants.num_epochs * steps_per_epoch

    # Instantiate the Adam class engine
    optimizer = AdamOptimizer(learning_rate=train_constants.base_lr)

    # --- 3. Master Optimization Loop ---
    for epoch in range(train_constants.num_epochs):
        np.random.shuffle(train_data.samples)
    
        epoch_loss = 0.0
        steps = 0

        for i in range(0, len(train_data.samples), train_constants.batch_size):
            current_step = optimizer.t + 1
        
            decay_ratio = np.minimum(1.0, current_step / total_steps)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
            optimizer.alpha = train_constants.min_lr + (train_constants.base_lr - train_constants.min_lr) * cosine_decay

            batch = train_data.samples[i:i + train_constants.batch_size]
            cur_batch_size = len(batch)
            if cur_batch_size == 0:
                continue
    
            # Stack matrices along batch axis: Shape (B, S)
            inp = np.stack([x["input_ids"] for x in batch])
            tar = np.stack([x["decoder_ids"] for x in batch])
            target_ids = np.stack([x["target_ids"] for x in batch]) # The true tokens to predict

            # Generate attention padding masks: (B, 1, 1, S)
            enc_padding_mask = (inp == train_data.word_to_idx["<pad>"])[:, np.newaxis, np.newaxis, :]
            dec_padding_mask = (inp == train_data.word_to_idx["<pad>"])[:, np.newaxis, np.newaxis, :]
            look_ahead_mask = np.triu(np.ones((train_constants.max_sequence_length, train_constants.max_sequence_length)), k=1).astype(np.float32)
            look_ahead_mask = look_ahead_mask[np.newaxis, np.newaxis, :, :]

            # Execute Autoregressive Forward Network Pass
            # vocab_probs shape: (B, S, Vocab_Size)
            vocab_probs, _ = transformer_model(
                inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask, embedding_matrix, training=True
            )

            # --- Masked Categorical Cross-Entropy Loss Calculation ---
            # Create a mask to zero out loss contribution from padding tokens
            loss_mask = (target_ids != train_data.word_to_idx["<pad>"]).astype(np.float32)
            
            # Flatten across batch and sequence bounds for cross-entropy parsing
            flat_probs = vocab_probs.reshape(-1, train_data.vocab_size)
            flat_targets = target_ids.flatten()
            flat_mask = loss_mask.flatten()
            
            # Stable log probabilities selection
            row_indices = np.arange(flat_probs.shape[0])
            selected_probs = np.maximum(flat_probs[row_indices, flat_targets], 1e-15)
            token_losses = -np.log(selected_probs)
            
            # Apply padding mask and calculate mean loss safely
            active_tokens = np.sum(flat_mask) + 1e-8
            batch_loss = np.sum(token_losses * flat_mask) / active_tokens

            epoch_loss += batch_loss
            steps += 1

            # --- Evaluate Sequence Logit Gradients for Backward Pass ---
            # d_vocab_probs shape matching vocab_probs exactly: (B, S, Vocab_Size)
            d_vocab_probs = np.zeros_like(vocab_probs)
            
            # Reshape to flat 2D views to apply vectorised cross-entropy shortcuts
            d_flat = d_vocab_probs.reshape(-1, train_data.vocab_size)
            
            # Cross-Entropy derivative: dZ = (P - Y) * Mask / Active_Token_Count
            d_flat[row_indices, flat_targets] = 1.0
            d_flat = (flat_probs - d_flat) * flat_mask[:, np.newaxis] / active_tokens
            
            # Reshape back to proper 3D structural boundaries
            d_vocab_probs = d_flat.reshape(vocab_probs.shape)

            # Execute complete Graph Backpropagation
            grads = transformer_model.backward(d_vocab_probs)
            
            # Execute parameter adjustments using momentum tracking fields
            embedding_matrix = optimizer.step(transformer_model, embedding_matrix, grads)

        print(f"Epoch {epoch+1:02d}/{train_constants.num_epochs} -> Sequence Cross-Entropy Loss: {epoch_loss/steps:.4f}")

    print("\nOptimization Complete! Saving unified weights...")
    save_transformer_weights(transformer_model, embedding_matrix, weights_filename)
    print("Weights Saved Successfully.")
