Overview
--------

This project implements a **decoder-only Transformer language model** for the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, using word-level tokenization (NLTK), embeddings initialized from a local FastText .bin vector file, custom positional encoding, and classic transformer architecture and training.

Key Features
------------

*   **Tokenizer:** Custom word-level tokenizer using NLTK (word\_tokenize), special tokens (, , , ) added manually.
    
*   **Embeddings:** Loads FastText vectors (dimension 300) from a local .bin file; out-of-vocabulary words are randomly initialized.
    
*   **Positional Encoding:** Classic sinusoidal positional encoding (Vaswani et al.).
    
*   **Transformer:** Decoder-only, 4 layers, 6 attention heads each (head dimension 50, so 300 total), context size 64.
    
*   **Training:** Teacher forcing (cross-entropy loss, ignores paddings), PyTorch DataLoader for batching; training and val loss/metrics tracked each epoch.
    
*   **Evaluation:** Reports average perplexity, BLEU, and qualitative generations.
    
*   **Visualization:** Plots loss/perplexity curves and attention heatmaps.
    
*   **Decoding:** Supports stochastic sampling and beam search (beam width configurable).
    
*   **Advanced:** Implements kv-caching and gradient accumulation.
    

How to Run
----------

1.  **Requirements**
    
    *   Python 3.8+
        
    *   torch, datasets, gensim, nltk, matplotlib, seaborn, tqdm
        
    *   Download a suitable FastText .bin file and update its filepath in the code
        
2.  **Train**
    
    *   Edit and run main() in the provided script
        
    *   Monitors training and validation loss per epoch
        
    *   Produces and saves model weights as decoder\_transformer.pt
        
    *   Saves plots for loss/perplexity curves and attention visualizations (.png files)
        
3.  **Generation**
    
    *   Call model.generate() with a prompt and decoded IDs to generate continuations
        
    *   Beam search optionally via model.beam\_search()
        
4.  **Evaluation**
    
    *   Average perplexity and BLEU computed by the script for validation samples
        
    *   Outputs sample generations and saves attention heatmaps
        

Configuration (Editable in Code)
--------------------------------

*   Dataset: HuggingFace 'roneneldan/TinyStories'
    
*   Vocab size: 50,000 (configurable)
    
*   Embedding dim: 300
    
*   Layers: 4
    
*   Attention heads: 6
    
*   Feedforward: 1200
    
*   Context size: 64
    
*   Learning rate: Typically 1e-3 to 3e-4
    
*   Batch size: 32 or 64 typical
    
*   FastText path: update in the code
    

Outputs
-------

*   **Model weights:** decoder\_transformer.pt
    
*   **Plots:**
    
    *   training\_curves.png (loss/perplexity)
        
    *   attention\_visualization.png
        
*   **Example generations:** Printed in training log and at end of training
