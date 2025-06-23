# gpt2.c
## 🚀 GPT-2 Inference Implementation in Pure C

This project is a **clean-room implementation of GPT-2 inference in pure C**, with no external libraries or Python dependencies. It is designed as a step-by-step learning tool for understanding how large language models like GPT-2 actually work under the hood.

It includes:
- A Byte Pair Encoding (BPE) tokenizer
- Manual transformer layer implementation (multi-head attention, MLP, LayerNorm, etc.)
- Model weight loading from Hugging Face-compatible files
- Full autoregressive token-by-token generation

---

### 📚 Educational Focus

> **This implementation is intentionally unoptimized and built for learning.**

Rather than focusing on speed, this project emphasizes clarity and structure. Every operation is spelled out — from dot products to attention scores — to help you understand how GPT-2 functions at the low level.

This is ideal for:
- Systems programmers learning machine learning
- ML researchers exploring inference on constrained hardware
- Anyone interested in the internals of LLMs

---

### 🔭 Roadmap: From C to High-Performance Inference

This is the starting point in a journey toward **efficient transformer inference**. Planned stages include:

- ✅ **Pure C implementation** (this project)
- ⏳ **Key-Value (KV) cache** for efficient autoregressive decoding
- ⏳ **Aligned memory allocation** for cache-friendly access
- ⏳ **Loop unrolling and SIMD** (e.g., with AVX/NEON)
- ⏳ **Custom GPU kernel generation**
- ⏳ **CUDA/HIP backend for real-time inference**

---

### 🎯 Final Goal

The ultimate goal is to evolve this into a **minimal, fast, and customizable GPU inference kernel** for LLMs. This project will serve as a step-by-step blueprint — from raw C to custom CUDA — for anyone looking to understand or build efficient transformer-based inference engines from the ground up.

---

Stay tuned — the next steps are coming soon.


# Steps 

- Download Pretrained GPT2 Weights (Make sure to pip install deps)
    ```
    python download_gpt.py
    ```
- Run 
    ```
    make run
    ```

- Output 
    ```
    Loading model weights...
    Model weights loaded successfully
    Model Num Params: 1.77M
    Model Size: 6.75 MB
    Encoded IDs: 40 321 3364 1659 41 13694 11 3666 3672 271 
    🕒 TTFT (Time To First Token): 8.8686 seconds
    Generated sequence: 40 321 3364 1659 41 13694 11 3666 3672 271 42 417 7114 35 6 
    Total generation time: 56.5769 seconds
    Decoded: IamkingofJungle,MynameisKelograpD'

    ```