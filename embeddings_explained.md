# Embeddings Explained: Initialization vs Forward Pass

Understanding the difference between embedding **initialization** and **forward pass** is crucial for grasping how neural networks process text. This document breaks down both concepts with concrete examples.

## Overview

Embeddings convert discrete token IDs (like 72, 105) into continuous vectors that neural networks can learn from. This happens in two distinct phases:

1. **Initialization**: Creating lookup tables with random numbers
2. **Forward Pass**: Looking up and combining values from those tables

## Part 1: Initialization - Creating the Lookup Tables

When you create an `Embeddings` object, it builds two **lookup tables** filled with random numbers:

```python
# This happens in __init__
embeddings = Embeddings(vocab_size=256, block_size=8, n_embd=4)

# Creates two lookup tables:
# 1. Token embedding table: 256 rows × 4 columns (one row per possible byte)
# 2. Position embedding table: 8 rows × 4 columns (one row per position 0-7)
```

### What Gets Created

#### Token Embedding Table (256 × 4)
```
Token ID | dim0   | dim1   | dim2   | dim3
---------|--------|--------|--------|--------
0        | 0.12   | -0.34  | 0.56   | -0.78
1        | 0.45   | 0.23   | -0.12  | 0.67
...      | ...    | ...    | ...    | ...
72 ('H') | 0.89   | -0.45  | 0.23   | 0.34
105 ('i')| 0.33   | 0.77   | -0.11  | 0.22
...      | ...    | ...    | ...    | ...
255      | 0.11   | 0.88   | -0.33  | 0.55
```

#### Position Embedding Table (8 × 4)
```
Position | dim0   | dim1   | dim2   | dim3
---------|--------|--------|--------|--------
0        | 0.22   | -0.11  | 0.44   | -0.66
1        | 0.77   | 0.33   | -0.88  | 0.12
2        | 0.99   | -0.22  | 0.55   | 0.44
...      | ...    | ...    | ...    | ...
7        | 0.66   | 0.44   | -0.77  | 0.88
```

**Key Point**: These are just lookup tables with random numbers. Nothing has been "processed" yet!

## Part 2: Forward Pass - Looking Up Values

When you call `forward()`, you're **looking up** values from these pre-built tables:

```python
# Input: "Hi" becomes token IDs [72, 105]
token_ids = torch.tensor([[72, 105]])  # Shape: [1, 2]

# Now call forward()
result = embeddings(token_ids)
```

### Step-by-Step Process

#### Step 1: Token Embedding Lookup
```python
# Look up token 72 ('H') and token 105 ('i') in the token table
tok_emb = self.token_embedding(token_ids)

# Result: 
# tok_emb[0, 0, :] = [0.89, -0.45, 0.23, 0.34]  # from row 72 ('H')
# tok_emb[0, 1, :] = [0.33,  0.77, -0.11, 0.22] # from row 105 ('i')
```

#### Step 2: Position Embedding Lookup
```python
# Create position indices [0, 1] for our 2-token sequence
pos = torch.arange(2)  # [0, 1]

# Look up positions 0 and 1 in the position table
pos_emb = self.position_embedding(pos)

# Result:
# pos_emb[0, :] = [0.22, -0.11, 0.44, -0.66]  # from row 0 (first position)
# pos_emb[1, :] = [0.77,  0.33, -0.88, 0.12]  # from row 1 (second position)
```

#### Step 3: Combine (Element-wise Addition)
```python
# Add token and position embeddings element-wise
final_embeddings = tok_emb + pos_emb

# For token 'H' at position 0:
# [0.89, -0.45, 0.23, 0.34] + [0.22, -0.11, 0.44, -0.66] = [1.11, -0.56, 0.67, -0.32]

# For token 'i' at position 1:  
# [0.33, 0.77, -0.11, 0.22] + [0.77, 0.33, -0.88, 0.12] = [1.10, 1.10, -0.99, 0.34]
```

## Visual Summary

```
INITIALIZATION (happens once):
┌─────────────────┐    ┌─────────────────┐
│ Token Embedding │    │Position Embedding│
│     Table       │    │     Table        │
│   256 × 4       │    │    8 × 4         │
│  [random nums]  │    │  [random nums]   │
└─────────────────┘    └─────────────────┘

FORWARD PASS (happens every time you process text):
Input: [72, 105] ("Hi")
   ↓
┌─────────────────┐    ┌─────────────────┐
│Look up row 72   │    │Look up row 0    │
│Look up row 105  │    │Look up row 1    │
└─────────────────┘    └─────────────────┘
   ↓                      ↓
┌─────────────────┐    ┌─────────────────┐
│ Token vectors   │ +  │Position vectors │
│for H, i         │    │for pos 0, 1     │
└─────────────────┘    └─────────────────┘
   ↓
Final embeddings for "Hi"
```

## Key Differences

| Aspect | Initialization | Forward Pass |
|--------|---------------|--------------|
| **When** | Once when object is created | Every time you process text |
| **What** | Creates lookup tables | Uses existing lookup tables |
| **Data** | Random numbers | Specific rows based on input |
| **Purpose** | Set up the infrastructure | Process actual text |
| **Result** | Empty tables ready for use | Meaningful vectors for tokens |

## Why This Matters

### 1. **Separation of Concerns**
- Initialization sets up the "vocabulary" of possible representations
- Forward pass retrieves the right representations for your specific input

### 2. **Efficiency**
- Tables are created once but used millions of times
- No need to recompute embeddings - just lookup!

### 3. **Learning**
- During training, those "random" numbers become meaningful
- The model learns which vector values best represent each token and position
- Same infrastructure, smarter numbers over time

## Example: Complete Flow

```python
# 1. INITIALIZATION
embeddings = Embeddings(vocab_size=256, block_size=8, n_embd=4)
# → Creates two 256×4 and 8×4 tables with random numbers

# 2. FORWARD PASS
text = "Hi"
tokenizer = ByteTokenizer()
token_ids = torch.tensor([tokenizer.encode(text)])  # [[72, 105]]

result = embeddings(token_ids)
# → Looks up rows 72, 105 from token table
# → Looks up rows 0, 1 from position table  
# → Adds them together
# → Returns 2 vectors of length 4

print(result.shape)  # torch.Size([1, 2, 4])
```

## The Magic of Training

Initially, those random numbers mean nothing. But through training:

- Token 72 ('H') might learn to represent "start of greeting"
- Position 0 might learn to represent "beginning of sequence"
- Their combination creates a unique representation for "'H' at the start"

The same lookup mechanism works throughout training - only the numbers in the tables get better! 