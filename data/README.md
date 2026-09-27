# Prompt-level outcomes

Every deployment and guarantee in the paper is a function of the files in this folder, so all of them can be recomputed on a CPU.

Each row of `outcomes/*.csv` is one calibration prompt served under one configuration. Every configuration of a file lists the same prompts in the same order.

| column | meaning |
|---|---|
| `config` | configuration label. `acc85/semi70` is accept threshold 0.85 with semi-completion threshold 0.70, and a suffix such as `add0.30` gives the admission threshold |
| `prompt` | benchmark and index, such as `gsm8k/12` |
| `correct` | 1 when the answer is correct, by exact match for math and by execution-based pass@1 for code |
| `tokens` | output tokens of the answer |
| `forwards` | model forwards, or target forwards for speculative decoding, and empty for the quantization grid |

```
config,prompt,correct,tokens,forwards
acc85/semi70,gsm8k/0,1,184,35
acc85/semi70,gsm8k/1,0,208,35
```

| file | prompts | configurations | used for |
|---|---|---|---|
| `llada2_math.csv` | 1,012 | 9 | Tables 2 and 8, and the grid with admission thresholds of Appendix D.7 |
| `llada2_math_large.csv` | 1,012 | 50 | Tables 2 and 17, Figure 5 |
| `llada2_code.csv` | 542 | 8 | Tables 2 and 9 |
| `sdar_math.csv` | 1,012 | 16 | Tables 2 and 10, and Table 4 with the distilled checkpoint under `distilled/` |
| `sdar_code.csv` | 664 | 8 | Tables 2 and 11 |
| `llada2_math_small.csv` | 512 | 8 | Table 12 |
| `llada2_code_small.csv` | 420 | 8 | Table 13 |
| `llada2_math_tiny.csv` | 64 | 8 | Figure 4 |
| `specdec_llama_gsm8k.csv` | 512 | 12 | Table 2 and Appendix E |
| `specdec_qwen_gsm8k.csv` | 512 | 12 | Table 2 and Appendix E |
| `specdec_llama_humaneval.csv` | 164 | 12 | Figure 4 |
| `specdec_qwen_humaneval.csv` | 164 | 12 | Figure 4 |
| `quant_llama_gsm8k.csv` | 512 | 4 | Table 2 and Appendix E.4 |

`grids.json` defines each grid of the paper by its outcome file, its cost measure, its reference configuration, and where needed a subset of configurations or prompts. The cost is `tpf` for block diffusion (tokens over forwards, pooled over prompts), `apf` for speculative decoding (accepted tokens for each target forward, averaged over prompts) and `memory` for quantization, whose weight sizes it lists.

`quant_memory.json` holds the weight and peak memory of bf16 and nf4 weights, measured during cached generation on 20 prompts, which give the 2.81× reduction in Table 2.
