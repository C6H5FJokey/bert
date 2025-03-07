import os
import torch
import numpy as np
import random
import argparse
import logging
from native_sparse_attention_pytorch import SparseAttention

from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    DataCollatorForLanguageModeling,
    TextDataset,
    Trainer,
    TrainingArguments,
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Define BASE_PATH
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, "model", "roberta-base")

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Custom RoBERTa with SparseAttention
class RobertaWithSparseAttention(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        
        # Replace the standard attention with SparseAttention in all layers
        for i, layer in enumerate(self.roberta.encoder.layer):
            hidden_size = config.hidden_size
            num_heads = config.num_attention_heads
            
            # Create sparse attention module
            sparse_attn = SparseAttention(
                dim=hidden_size,
                dim_head=hidden_size // num_heads,
                heads=num_heads,
                sliding_window_size=4,
                compress_block_size=8,
                selection_block_size=8,
                num_selected_blocks=4
            )
            
            # Create a forward hook to replace the attention operation
            def create_forward_hook(sparse_attention):
                def hook(module, input, output):
                    hidden_states = input[0]
                    attention_mask = input[1] if len(input) > 1 else None
                    
                    # Apply sparse attention
                    attended = sparse_attention(hidden_states)
                    
                    # Return tuple to match the expected output format
                    return (attended, None)  # None for attention weights
                return hook
            
            # Register the hook
            layer.attention.self.register_forward_hook(create_forward_hook(sparse_attn))

def main():
    parser = argparse.ArgumentParser()
    
    # Training parameters
    parser.add_argument("--train_file", type=str, required=True, help="Path to training file.")
    parser.add_argument("--validation_file", type=str, help="Path to validation file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the model.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="Maximum sequence length.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning rate.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Number of training epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--mlm_probability", default=0.15, type=float, help="MLM probability.")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load tokenizer and config
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    config = RobertaConfig.from_pretrained(MODEL_PATH)
    
    # Create model with sparse attention
    model = RobertaWithSparseAttention.from_pretrained(MODEL_PATH, config=config)
    
    # Load datasets
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=args.train_file,
        block_size=args.max_seq_length
    )
    
    eval_dataset = None
    if args.validation_file:
        eval_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=args.validation_file,
            block_size=args.max_seq_length
        )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_steps=10000,
        save_total_limit=2,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()