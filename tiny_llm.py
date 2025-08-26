"""
TinyLLM - Learning to Build a Language Model from Scratch
=========================================================
We're building a Transformer-based language model step by step.
This file will grow incrementally through PRs, each adding one concept.

PR #1: Byte-Level Tokenizer
---------------------------
First, we need to convert text into numbers that our neural network can process.
"""


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


if __name__ == "__main__":
    print("=" * 60)
    print("BYTE-LEVEL TOKENIZER EXPLORATION")
    print("=" * 60)
    print()
    test_tokenizer()