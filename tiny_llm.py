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

PR #3: Embeddings
-----------------
Transform discrete token IDs into continuous vectors that neural networks can
learn from. We need two types:
1. Token embeddings: Each token ID gets a learnable vector representation
2. Positional embeddings: Encode where each token appears in the sequence
"""

import torch
import torch.nn as nn
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


class Embeddings(nn.Module):
    """
    Embeddings: Converting Token IDs to Learnable Vectors
    ------------------------------------------------------
    
    Neural networks can't directly process discrete token IDs (like 0, 1, 2...).
    We need to convert them to continuous vectors that can be updated via gradients.
    
    Two types of embeddings:
    1. Token Embeddings: "What" - the identity of each token
    2. Positional Embeddings: "Where" - the position in the sequence
    
    Why both?
    - Transformers process all positions in parallel (no recurrence)
    - Without position info, "cat sat" = "sat cat" (just a bag of tokens!)
    - Position embeddings let the model know token order
    
    The final representation is: token_emb + position_emb
    """
    
    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        """
        Args:
            vocab_size: Number of unique tokens (256 for byte-level)
            block_size: Maximum sequence length we'll process
            n_embd: Dimension of embedding vectors (e.g., 384, 768)
        """
        super().__init__()
        
        # Token embeddings: lookup table of vocab_size x n_embd
        # Each of our 256 possible bytes gets an n_embd-dimensional vector
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        
        # Position embeddings: lookup table of block_size x n_embd
        # Each position (0, 1, 2, ..., block_size-1) gets an n_embd-dimensional vector
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        # Store dimensions for later use
        self.block_size = block_size
        self.n_embd = n_embd
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings.
        
        Args:
            token_ids: Tensor of shape (batch_size, sequence_length)
                      Contains token IDs from our tokenizer
        
        Returns:
            Tensor of shape (batch_size, sequence_length, n_embd)
            Each token ID is now an n_embd-dimensional vector!
        """
        batch_size, seq_len = token_ids.shape
        
        # Get token embeddings: (batch_size, seq_len) -> (batch_size, seq_len, n_embd)
        tok_emb = self.token_embedding(token_ids)
        
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        pos = torch.arange(seq_len, device=token_ids.device)
        
        # Get position embeddings: (seq_len,) -> (seq_len, n_embd)
        # We'll broadcast this across the batch
        pos_emb = self.position_embedding(pos)
        
        # Combine token and position information
        # Both have shape (batch_size, seq_len, n_embd), so we just add!
        embeddings = tok_emb + pos_emb
        
        return embeddings


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


def test_embeddings():
    """
    Test embeddings and visualize how they transform token IDs to vectors.
    """
    print("EMBEDDINGS DEEP DIVE")
    print("-" * 60)
    print()
    
    # Setup: Small dimensions for easy visualization
    vocab_size = 256  # Our byte vocabulary
    block_size = 8    # Max sequence length
    n_embd = 16       # Small embedding dimension for visualization
    
    print(f"Configuration:")
    print(f"  Vocab size: {vocab_size} (all possible bytes)")
    print(f"  Block size: {block_size} (max sequence length)")
    print(f"  Embedding dim: {n_embd} (size of each vector)")
    print()
    
    # Create the embeddings module
    embeddings = Embeddings(vocab_size, block_size, n_embd)
    
    # Example 1: Single sequence
    print("=" * 60)
    print("EXAMPLE 1: Single Sequence")
    print("=" * 60)
    
    text = "Hi!"
    tokenizer = ByteTokenizer()
    token_ids = tokenizer.encode(text)
    print(f"Text: '{text}'")
    print(f"Token IDs: {token_ids}")
    print()
    
    # Convert to tensor and add batch dimension
    tokens_tensor = torch.tensor([token_ids])  # Shape: [1, 3]
    print(f"Input tensor shape: {tokens_tensor.shape}")
    print(f"Input tensor: {tokens_tensor}")
    print()
    
    # Get embeddings
    with torch.no_grad():  # No gradients needed for testing
        emb = embeddings(tokens_tensor)
    
    print(f"Output embedding shape: {emb.shape}")
    print(f"  (batch_size=1, seq_len=3, n_embd={n_embd})")
    print()
    
    # Visualize the embedding matrix for first token
    print("First token embedding vector (first 8 values):")
    print(f"  Token '{text[0]}' (ID={token_ids[0]}) -> {emb[0, 0, :8].numpy().round(3)}")
    print()
    
    # Example 2: Understanding position embeddings
    print("=" * 60)
    print("EXAMPLE 2: Position Embeddings Visualization")
    print("=" * 60)
    
    # Same token in different positions
    repeated_token = [72, 72, 72]  # 'H' three times
    tokens_tensor = torch.tensor([repeated_token])
    
    print(f"Input: Same token (72='H') in 3 positions")
    print(f"Token IDs: {repeated_token}")
    print()
    
    # Get individual components
    with torch.no_grad():
        # Token embeddings (same for all positions)
        tok_emb = embeddings.token_embedding(tokens_tensor)
        
        # Position embeddings (different for each position)
        positions = torch.arange(3)
        pos_emb = embeddings.position_embedding(positions)
        
        # Combined
        combined = embeddings(tokens_tensor)
    
    print("Token embedding for 'H' (same for all positions):")
    print(f"  First 8 values: {tok_emb[0, 0, :8].numpy().round(3)}")
    print()
    
    print("Position embeddings (different for each position):")
    for i in range(3):
        print(f"  Position {i}: {pos_emb[i, :8].numpy().round(3)}")
    print()
    
    print("Combined embeddings (token + position):")
    for i in range(3):
        print(f"  Position {i}: {combined[0, i, :8].numpy().round(3)}")
    print()
    
    print("Notice: Same token 'H' has different final embeddings due to position!")
    print()
    
    # Example 3: Batch processing
    print("=" * 60)
    print("EXAMPLE 3: Batch Processing")
    print("=" * 60)
    
    texts = ["Hi", "Bye", "OK!"]
    batch = []
    max_len = 3
    
    print("Batch of texts:")
    for text in texts:
        tokens = tokenizer.encode(text)
        # Pad to same length (using 0 as padding token)
        padded = tokens + [0] * (max_len - len(tokens))
        batch.append(padded[:max_len])
        print(f"  '{text}' -> {tokens} -> padded: {padded[:max_len]}")
    
    batch_tensor = torch.tensor(batch)
    print(f"\nBatch tensor shape: {batch_tensor.shape} (batch_size=3, seq_len=3)")
    print()
    
    with torch.no_grad():
        batch_emb = embeddings(batch_tensor)
    
    print(f"Batch embedding shape: {batch_emb.shape}")
    print(f"  (batch_size=3, seq_len=3, n_embd={n_embd})")
    print()
    
    # Show that each sequence in the batch gets its own embeddings
    for i, text in enumerate(texts):
        print(f"Embeddings for '{text}' (first token, first 8 dims):")
        print(f"  {batch_emb[i, 0, :8].numpy().round(3)}")
    print()
    
    # Example 4: Why position matters
    print("=" * 60)
    print("EXAMPLE 4: Why Position Matters")
    print("=" * 60)
    
    text1 = "AB"
    text2 = "BA"
    
    tokens1 = torch.tensor([tokenizer.encode(text1)])
    tokens2 = torch.tensor([tokenizer.encode(text2)])
    
    with torch.no_grad():
        emb1 = embeddings(tokens1)
        emb2 = embeddings(tokens2)
    
    print(f"'{text1}' tokens: {tokens1[0].tolist()}")
    print(f"'{text2}' tokens: {tokens2[0].tolist()}")
    print()
    
    print("Without positions, these would be identical (just different order).")
    print("With positions, they're completely different:")
    print()
    
    print(f"'{text1}' first token ('A' at pos 0) embedding (first 8 dims):")
    print(f"  {emb1[0, 0, :8].numpy().round(3)}")
    print(f"'{text2}' first token ('B' at pos 0) embedding (first 8 dims):")
    print(f"  {emb2[0, 0, :8].numpy().round(3)}")
    print()
    print("These are different because 'A' vs 'B' are different tokens!")
    print()
    
    print(f"'{text1}' second token ('B' at pos 1) embedding (first 8 dims):")
    print(f"  {emb1[0, 1, :8].numpy().round(3)}")
    print(f"'{text2}' second token ('A' at pos 1) embedding (first 8 dims):")
    print(f"  {emb2[0, 1, :8].numpy().round(3)}")
    print()
    print("'B' at position 1 differs from 'B' at position 0!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "tokenizer":
            print("=" * 60)
            print("BYTE-LEVEL TOKENIZER EXPLORATION")
            print("=" * 60)
            print()
            test_tokenizer()
        elif sys.argv[1] == "data":
            print("=" * 60)
            print("DATA LOADING EXPLORATION")
            print("=" * 60)
            print()
            test_data_loading()
        elif sys.argv[1] == "embeddings":
            print("=" * 60)
            print("EMBEDDINGS EXPLORATION")
            print("=" * 60)
            print()
            test_embeddings()
        else:
            print("Usage: python tiny_llm.py [tokenizer|data|embeddings]")
    else:
        # Default: run the latest test
        print("=" * 60)
        print("EMBEDDINGS EXPLORATION")
        print("=" * 60)
        print()
        test_embeddings()