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

PR #4: Single-Head Attention
----------------------------
The core mechanism of transformers - allowing tokens to "attend" to each other.
Attention computes weighted averages of values based on query-key similarities.
The three components:
1. Query (Q): "What am I looking for?"
2. Key (K): "What information do I contain?"  
3. Value (V): "Here's my actual content to share"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math


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


class SingleHeadAttention(nn.Module):
    """
    Single-Head Self-Attention: The Foundation of Transformers
    ----------------------------------------------------------
    
    Self-attention allows each token to look at all other tokens and decide
    which ones are most relevant. It's "self" because the queries, keys, and
    values all come from the same input sequence.
    
    The Attention Formula:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Where:
    - Q (Query): What each token is looking for
    - K (Key): What each token offers to others
    - V (Value): The actual information to aggregate
    - √d_k: Scaling factor to prevent softmax saturation
    
    For autoregressive models (like GPT), we add causal masking to prevent
    tokens from attending to future positions (no cheating!).
    """
    
    def __init__(self, n_embd: int, block_size: int, dropout: float = 0.1):
        """
        Args:
            n_embd: Embedding dimension (must be divisible by n_head for multi-head)
            block_size: Maximum sequence length
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        
        # Linear projections for Q, K, V
        # All three projections map from n_embd to n_embd
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        
        # Output projection (after attention)
        self.out_proj = nn.Linear(n_embd, n_embd)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask - lower triangular matrix
        # This prevents tokens from attending to future positions
        # register_buffer ensures it's moved to GPU with the model but not updated
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
        
        self.n_embd = n_embd
        self.scale = 1.0 / math.sqrt(n_embd)  # 1/√d_k for scaling
        
    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """
        Apply self-attention to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            return_weights: If True, also return attention weights for visualization
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
            Optionally: attention weights of shape (batch_size, seq_len, seq_len)
        """
        B, T, C = x.shape  # Batch, Time (sequence), Channels (embedding)
        
        # Project input to Q, K, V
        # Each has shape (B, T, C)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores: Q @ K^T
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        # Each cell [i,j] represents how much token i attends to token j
        scores = Q @ K.transpose(-2, -1) * self.scale
        
        # Apply causal mask (only for autoregressive models)
        # Set future positions to -inf so they become 0 after softmax
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        # Each row sums to 1 (probability distribution over tokens to attend to)
        att_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        att_weights = self.dropout(att_weights)
        
        # Weighted sum of values
        # (B, T, T) @ (B, T, C) -> (B, T, C)
        # Each token becomes a weighted combination of all tokens it attends to
        out = att_weights @ V
        
        # Final output projection
        out = self.out_proj(out)
        
        # Residual connection: add input to output
        # This helps gradients flow and lets the network learn "changes" to x
        out = x + out
        
        if return_weights:
            return out, att_weights
        return out


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


def test_attention():
    """
    Deep dive into how single-head attention works step by step.
    """
    print("SINGLE-HEAD ATTENTION MECHANISM")
    print("-" * 60)
    print()
    
    # Configuration
    n_embd = 8        # Small embedding dimension for visualization
    block_size = 6    # Maximum sequence length
    
    print("Configuration:")
    print(f"  Embedding dimension: {n_embd}")
    print(f"  Block size: {block_size}")
    print()
    
    # Create modules
    tokenizer = ByteTokenizer()
    embeddings = Embeddings(256, block_size, n_embd)
    attention = SingleHeadAttention(n_embd, block_size, dropout=0.0)  # No dropout for clarity
    
    # Example 1: Step-by-step attention computation
    print("=" * 60)
    print("EXAMPLE 1: Step-by-Step Attention Computation")
    print("=" * 60)
    print()
    
    # Simple input sequence
    text = "ABC"
    token_ids = tokenizer.encode(text)
    print(f"Input text: '{text}'")
    print(f"Token IDs: {token_ids}")
    print()
    
    # Get embeddings
    tokens_tensor = torch.tensor([token_ids])  # Add batch dimension
    with torch.no_grad():
        emb = embeddings(tokens_tensor)
        print(f"Embeddings shape: {emb.shape} (batch=1, seq_len=3, n_embd={n_embd})")
        print()
        
        # Step 1: Compute Q, K, V
        Q = attention.query(emb)
        K = attention.key(emb)
        V = attention.value(emb)
        
        print("Step 1: Linear Projections to Q, K, V")
        print("-" * 40)
        print(f"Q shape: {Q.shape}")
        print(f"K shape: {K.shape}")
        print(f"V shape: {V.shape}")
        print()
        
        print("Query vectors (what each token is looking for):")
        for i, char in enumerate(text):
            print(f"  '{char}': {Q[0, i, :4].numpy().round(2)}...")
        print()
        
        print("Key vectors (what each token offers):")
        for i, char in enumerate(text):
            print(f"  '{char}': {K[0, i, :4].numpy().round(2)}...")
        print()
        
        print("Value vectors (actual content to share):")
        for i, char in enumerate(text):
            print(f"  '{char}': {V[0, i, :4].numpy().round(2)}...")
        print()
        
        # Step 2: Compute attention scores
        scores = Q @ K.transpose(-2, -1) * attention.scale
        
        print("Step 2: Compute Attention Scores (Q @ K^T)")
        print("-" * 40)
        print(f"Scores = Q @ K^T * scale")
        print(f"Scale factor: 1/√{n_embd} = {attention.scale:.3f}")
        print()
        print("Raw attention scores (before masking):")
        print("     " + "    ".join([f"'{c}'" for c in text]))
        for i, char in enumerate(text):
            row = [f"{scores[0, i, j].item():6.2f}" for j in range(len(text))]
            print(f"'{char}': " + " ".join(row))
        print("\nEach row shows how much that token wants to attend to each column")
        print()
        
        # Step 3: Apply causal mask
        masked_scores = scores.masked_fill(attention.mask[:3, :3] == 0, float('-inf'))
        
        print("Step 3: Apply Causal Mask")
        print("-" * 40)
        print("Mask pattern (1 = allowed, 0 = blocked):")
        print(attention.mask[:3, :3].int().numpy())
        print()
        print("Scores after masking (future = -inf):")
        print("     " + "    ".join([f"'{c}'" for c in text]))
        for i, char in enumerate(text):
            row = []
            for j in range(len(text)):
                val = masked_scores[0, i, j].item()
                if val == float('-inf'):
                    row.append("  -inf")
                else:
                    row.append(f"{val:6.2f}")
            print(f"'{char}': " + " ".join(row))
        print("\n'A' can only see itself, 'B' can see A and B, 'C' can see all")
        print()
        
        # Step 4: Apply softmax
        att_weights = F.softmax(masked_scores, dim=-1)
        
        print("Step 4: Apply Softmax (convert to probabilities)")
        print("-" * 40)
        print("Attention weights (each row sums to 1.0):")
        print("     " + "    ".join([f"'{c}'" for c in text]))
        for i, char in enumerate(text):
            row = [f"{att_weights[0, i, j].item():6.3f}" for j in range(len(text))]
            print(f"'{char}': " + " ".join(row))
            print(f"        Sum: {att_weights[0, i, :].sum().item():.3f}")
        print()
        
        # Step 5: Weighted sum of values
        output = att_weights @ V
        
        print("Step 5: Weighted Sum of Values")
        print("-" * 40)
        print("Output = attention_weights @ V")
        print(f"Output shape: {output.shape}")
        print()
        print("Final output vectors (weighted combinations):")
        for i, char in enumerate(text):
            print(f"  '{char}': {output[0, i, :4].numpy().round(2)}...")
        print()
    
    # Example 2: Visualizing attention patterns
    print("=" * 60)
    print("EXAMPLE 2: Attention Patterns for a Sentence")
    print("=" * 60)
    print()
    
    sentence = "Hi Bob"
    tokens = tokenizer.encode(sentence)
    tokens_tensor = torch.tensor([tokens])
    
    with torch.no_grad():
        emb = embeddings(tokens_tensor)
        output, weights = attention(emb, return_weights=True)
        
        print(f"Input: '{sentence}'")
        print(f"Tokens: {tokens}")
        print()
        
        # Display attention matrix
        print("Attention weights matrix:")
        print("(Each row shows where that token is looking)")
        print()
        
        # Create character labels for each byte
        chars = []
        for t in tokens:
            try:
                chars.append(tokenizer.decode([t]))
            except:
                chars.append(f"[{t}]")
        
        # Header
        print("      " + "".join([f"{c:>7}" for c in chars]))
        
        # Rows
        for i, from_char in enumerate(chars):
            row_weights = weights[0, i, :].numpy()
            row_str = "".join([f"{w:7.3f}" for w in row_weights])
            print(f"{from_char:>5}: {row_str}")
        print()
        
        # Find strongest attention for each position
        print("Strongest attention patterns:")
        for i, from_char in enumerate(chars):
            max_idx = weights[0, i, :i+1].argmax().item()  # Only look at valid positions
            max_weight = weights[0, i, max_idx].item()
            to_char = chars[max_idx]
            print(f"  '{from_char}' -> '{to_char}' (weight: {max_weight:.3f})")
    
    # Example 3: Effect of attention on information flow
    print()
    print("=" * 60)
    print("EXAMPLE 3: Information Flow Through Attention")
    print("=" * 60)
    print()
    
    # Create a sequence where position matters
    text1 = "AAB"
    text2 = "ABA"
    text3 = "BAA"
    
    print("Three sequences with same tokens, different positions:")
    print(f"  1. '{text1}'")
    print(f"  2. '{text2}'")
    print(f"  3. '{text3}'")
    print()
    
    for text in [text1, text2, text3]:
        tokens = tokenizer.encode(text)
        tokens_tensor = torch.tensor([tokens])
        
        with torch.no_grad():
            emb = embeddings(tokens_tensor)
            output = attention(emb)
            
            # Look at the last position's output
            last_output = output[0, -1, :4].numpy()
            print(f"'{text}' - Last position output: {last_output.round(2)}")
    
    print()
    print("Different outputs despite same tokens - position matters!")
    print()
    
    # Example 4: Attention with and without causal mask
    print("=" * 60)
    print("EXAMPLE 4: Causal Mask Effect")
    print("=" * 60)
    print()
    
    text = "123"
    tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([tokens])
    
    with torch.no_grad():
        emb = embeddings(tokens_tensor)
        
        # Temporarily disable mask to show difference
        original_mask = attention.mask.clone()
        
        # With causal mask (normal)
        output_causal, weights_causal = attention(emb, return_weights=True)
        
        # Without causal mask (all tokens see all)
        attention.mask.fill_(1)  # Allow all connections
        output_full, weights_full = attention(emb, return_weights=True)
        
        # Restore original mask
        attention.mask.data = original_mask
        
        print(f"Input: '{text}'")
        print()
        
        print("With Causal Mask (autoregressive):")
        print("     '1'    '2'    '3'")
        for i in range(3):
            row = [f"{weights_causal[0, i, j].item():6.3f}" for j in range(3)]
            print(f"'{text[i]}': " + " ".join(row))
        print("(Lower triangular - can only see past)")
        print()
        
        print("Without Causal Mask (bidirectional):")
        print("     '1'    '2'    '3'")
        for i in range(3):
            row = [f"{weights_full[0, i, j].item():6.3f}" for j in range(3)]
            print(f"'{text[i]}': " + " ".join(row))
        print("(Full attention - all tokens see all)")


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
        elif sys.argv[1] == "attention":
            print("=" * 60)
            print("ATTENTION MECHANISM EXPLORATION")
            print("=" * 60)
            print()
            test_attention()
        else:
            print("Usage: python tiny_llm.py [tokenizer|data|embeddings|attention]")
    else:
        # Default: run the latest test
        print("=" * 60)
        print("ATTENTION MECHANISM EXPLORATION")
        print("=" * 60)
        print()
        test_attention()