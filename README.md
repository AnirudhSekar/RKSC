# RKSC: Reasoning-Aware KV Cache Sharing
 
Benchmarking notebook for **RKSC** — a training-free inference acceleration framework for multi-branch LLM reasoning. RKSC combines two orthogonal mechanisms:
 
- **ASKS** (Attention-Similarity KV Sharing) — computes the shared prefix KV cache once and broadcasts it to all branches via hidden-state cosine similarity gating, eliminating redundant O(n²) prefill for B−1 branches.
- **CGEE** (Confidence-Gated Early Exit) — skips the verification forward pass when the highest-confidence branch is decisive, saving ~180ms per problem when triggered.
 

## Requirements
 
```bash
pip install "transformers>=4.40" datasets tqdm numpy scipy matplotlib seaborn pandas psutil
```
 
A **HuggingFace token** is required for GPQA Diamond. Set it as:
- A Colab secret named `HF_TOKEN`, or
- The `HF_TOKEN` environment variable
 
You must also accept all necessary terms and conditions for the models and datasets on HuggingFace.

## Usage
 
The notebook is self-contained and designed to be run top-to-bottom in Google Colab or any Jupyter environment with GPU access.

## License
MIT License

Copyright (c) 2026 Anonymous

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
