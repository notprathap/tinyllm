"""
TinyLLM - Learning to Build a Language Model from Scratch
=========================================================
We're building a Transformer-based language model step by step.
This file will grow incrementally through PRs, each adding one concept.

PR #1: Byte-Level Tokenizer
---------------------------
First, we need to convert text into numbers that our neural network can process.

PR #2: Data Loading & Batching
------------------------------
Now we'll prepare training data: sequences where the model learns to predict
the next token given previous tokens.
"""

import torch
from torch.utils.data import Dataset


class ByteTokenizer:
    """
    Byte-Level Tokenizer: The simplest possible tokenization strategy.
    
    Why bytes?
    - Every possible text character can be represented as bytes (0-255)
    - No need for vocabulary files or special token handling
    - Works with any language or even binary data
    - Vocabulary size is always exactly 256
    
    The tradeoff:
    - Sequences become longer (each Unicode char might be multiple bytes)
    - Model needs to learn character combinations from scratch
    - But it's perfect for learning because it's so simple!
    """
    
    def __init__(self):
        # Our vocabulary is simply all possible byte values: 0-255
        self.vocab_size = 256
        
    def encode(self, text: str) -> list[int]:
        """
        Convert text string to list of integers (token IDs).
        
        Process:
        1. Convert string to UTF-8 bytes
        2. Each byte becomes a token ID (0-255)
        
        Example:
            "Hi" -> UTF-8 bytes [72, 105] -> token IDs [72, 105]
            "👋" (emoji) -> UTF-8 bytes [240, 159, 145, 139] -> 4 tokens!
        """
        return list(text.encode('utf-8'))
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Convert list of token IDs back to text string.
        
        Process:
        1. Convert list of ints to bytes
        2. Decode bytes as UTF-8 to get string
        
        Args:
            token_ids: List of integers in range 0-255
            
        Returns:
            Decoded text string
        """
        # Convert list of ints to bytes object, then decode as UTF-8
        # We use 'ignore' to skip invalid UTF-8 sequences gracefully
        return bytes(token_ids).decode('utf-8', errors='ignore')


class CharDataset(Dataset):
    """
    Dataset for character-level language modeling.
    
    Core Concept: Next-Token Prediction
    ------------------------------------
    Language models learn by predicting the next token given previous tokens.
    For each position in our text, we create:
    - Input (x): tokens[i : i+block_size]
    - Target (y): tokens[i+1 : i+1+block_size]
    
    Example with block_size=4 and text "hello":
    Tokens: [104, 101, 108, 108, 111]  # h,e,l,l,o
    
    Training examples:
    x=[104, 101, 108, 108] -> y=[101, 108, 108, 111]
    (Given "hell", predict "ello")
    
    The model learns: after seeing [104,101,108,108], 
    the next token should be 111 ('o')
    """
    
    def __init__(self, data: torch.Tensor, block_size: int):
        """
        Args:
            data: 1D tensor of token IDs from our tokenizer
            block_size: Maximum context length (how many tokens to look back)
        """
        assert data.dim() == 1, "Data must be a 1D tensor of token IDs"
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        # We can create (total_tokens - block_size) training examples
        # We need block_size tokens for input and 1 for prediction
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        """
        Get one training example.
        
        Returns:
            x: Input sequence of block_size tokens
            y: Target sequence (shifted by 1) of block_size tokens
        """
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


def load_text_data(text: str, block_size: int = 8):
    """
    Convert text to training dataset.
    
    Pipeline:
    1. Text -> Tokens (using ByteTokenizer)
    2. Tokens -> Tensor
    3. Tensor -> Dataset
    """
    # Tokenize the text
    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode(text)
    
    # Convert to PyTorch tensor
    data_tensor = torch.tensor(tokens, dtype=torch.long)
    
    # Create dataset
    dataset = CharDataset(data_tensor, block_size)
    
    return dataset, tokenizer


def test_tokenizer():
    """
    Test our tokenizer with various examples to understand how it works.
    """
    tokenizer = ByteTokenizer()
    
    # Test 1: Simple ASCII text
    text1 = "Hello"
    encoded1 = tokenizer.encode(text1)
    decoded1 = tokenizer.decode(encoded1)
    print(f"Text: '{text1}'")
    print(f"Encoded: {encoded1}")
    print(f"Decoded: '{decoded1}'")
    print(f"Tokens used: {len(encoded1)}")
    print()
    
    # Test 2: Text with special characters
    text2 = "AI & ML"
    encoded2 = tokenizer.encode(text2)
    decoded2 = tokenizer.decode(encoded2)
    print(f"Text: '{text2}'")
    print(f"Encoded: {encoded2}")
    print(f"Decoded: '{decoded2}'")
    print()
    
    # Test 3: Unicode emoji (interesting case!)
    text3 = "Hello 🤖"
    encoded3 = tokenizer.encode(text3)
    decoded3 = tokenizer.decode(encoded3)
    print(f"Text: '{text3}'")
    print(f"Encoded: {encoded3}")
    print(f"Decoded: '{decoded3}'")
    print(f"Note: Emoji uses {len(encoded3) - 6} bytes!")
    print()
    
    # Test 4: Verify round-trip consistency
    test_texts = ["The quick brown fox", "123!@#", "新中文", "مرحبا"]
    print("Round-trip tests:")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        status = "✓" if text == decoded else "✗"
        print(f"  {status} '{text}' -> {len(encoded)} tokens -> '{decoded}'")


def test_data_loading():
    """
    Test data loading and understand how training examples are created.
    """
    # Simple text for demonstration
    text = "Hello world!"
    block_size = 4
    
    print("DATA LOADING DEMONSTRATION")
    print("-" * 40)
    print(f"Text: '{text}'")
    print(f"Block size: {block_size}")
    print()
    
    # Load and prepare data
    dataset, tokenizer = load_text_data(text, block_size)
    
    # Show tokens
    tokens = tokenizer.encode(text)
    print(f"Tokens: {tokens}")
    print(f"Decoded: '{tokenizer.decode(tokens)}'")
    print()
    
    # Show how many training examples we can create
    print(f"Total tokens: {len(tokens)}")
    print(f"Training examples: {len(dataset)}")
    print(f"(total_tokens - block_size = {len(tokens)} - {block_size} = {len(dataset)})")
    print()
    
    # Show first few training examples
    print("Training Examples (x -> y):")
    print("(Each y is x shifted by 1 position)")
    print()
    
    for i in range(min(3, len(dataset))):
        x, y = dataset[i]
        x_text = tokenizer.decode(x.tolist())
        y_text = tokenizer.decode(y.tolist())
        print(f"Example {i+1}:")
        print(f"  x: {x.tolist()} -> '{x_text}'")
        print(f"  y: {y.tolist()} -> '{y_text}'")
        print(f"  Teaching: After '{x_text}', next token is {y[-1]} ('{tokenizer.decode([y[-1].item()])}')")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "tokenizer":
        print("=" * 60)
        print("BYTE-LEVEL TOKENIZER EXPLORATION")
        print("=" * 60)
        print()
        test_tokenizer()
    else:
        print("=" * 60)
        print("DATA LOADING EXPLORATION")
        print("=" * 60)
        print()
        test_data_loading()