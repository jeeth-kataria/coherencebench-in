# Paper Reading Notes — CoherenceBench-IN

---

## Paper 1: RULER — What's the Real Context Size of Your Long-Context Language Models?

**Authors:** Hsieh, Sun, Krez, et al. (2024)  
**Venue:** arXiv:2404.06654  
**Read on:** 27 February 2026  
**Priority:** #1 — primary contrast benchmark  
**Link:** https://arxiv.org/abs/2404.06654

### Summary

RULER is a synthetic benchmark that evaluates LLMs on long-context tasks beyond the standard Needle-in-a-Haystack (NIAH) test. It introduces 13 tasks across 4 categories: (1) **Retrieval** — single/multi-key NIAH variants, (2) **Multi-hop Tracing** — variable tracking across context, (3) **Aggregation** — common/frequent word identification, and (4) **Question Answering** — long-context QA. Testing 17 models at lengths from 4K to 128K tokens, RULER found that most models claiming 32K+ context support show significant performance degradation beyond their effective context size. Only 4/17 models maintained >85% accuracy at 32K tokens on all tasks.

### What RULER Tests (and does well):
- **Retrieval accuracy at scale** — can the model find needles (facts) in haystacks (long contexts)?
- Multiple retrieval variants: single needle, multi-needle, multi-key, multi-value
- Variable tracing — a form of multi-hop reasoning
- Configurable context length — tests from 4K to 128K systematically

### What RULER Does NOT Test (our gap):
- **No coherence tasks at all.** RULER is entirely about retrieval — locating and extracting specific items planted in synthetic contexts.
- No entity consistency — it doesn't test whether a model notices contradictions about an entity across the context.
- No temporal reasoning — no timeline or date-ordering tasks.
- No causal chain tracking — no cause-effect integrity checks.
- The "distractors" in RULER are random filler text, not semantically meaningful context. Real documents have narrative structure, entities, timelines, and causality — RULER ignores all of this.
- **English only** — no multilingual evaluation.

### Why This Matters for Us:
RULER is our **primary foil**. A model scoring 95% on RULER may still fail catastrophically on CoherenceBench-IN. Our paper's introduction should open with something like: *"A model that passes RULER with near-perfect scores can still silently accept a document that contradicts an entity's nationality halfway through, violates an impossible timeline, or breaks a causal chain. RULER tests retrieval; CoherenceBench-IN tests coherence."*

### Key Numbers to Cite:
- 17 models evaluated, only 4 maintained >85% at 32K across all tasks
- Performance drops ~10–20% between 4K and 32K for most models (even on simple retrieval)
- Implication: if retrieval degrades with length, coherence (harder) likely degrades even more

### Connection to Our Corruption Distance Analysis:
RULER's length-based degradation finding directly motivates our corruption distance analysis. If models struggle to retrieve facts as context grows, they will certainly struggle to detect contradictions placed far from the reference point. Our distance curves should show steeper degradation than RULER's, because coherence detection is harder than retrieval.

### Must-Cite Quote:
> "Performance of all models decreases as sequence length increases... Several long-context models fail dramatically even at context sizes they claim to support."

### BibTeX:
```bibtex
@article{hsieh2024ruler,
  title={RULER: What's the Real Context Size of Your Long-Context Language Models?},
  author={Hsieh, Cheng-Ping and Sun, Simeng and Krez, Igor and others},
  journal={arXiv preprint arXiv:2404.06654},
  year={2024}
}
```

---

## Paper 2: Lost in the Middle — How Language Models Use Long Contexts

**Authors:** Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang (2024)  
**Venue:** TACL 2024  
**Read on:** [TO READ — Phase 1, Week 1]  
**Priority:** #2 — motivates distance analysis  
**Link:** https://arxiv.org/abs/2307.03172

### Summary
[TO COMPLETE]

---

## Paper 3: LongBench v2

**Authors:** Bai et al. (2024)  
**Read on:** [TO READ — Phase 1, Week 1]  
**Priority:** #3 — confirm no coherence coverage  

### Summary
[TO COMPLETE]

---

## Paper 4: TLDM — Too Long, Didn't Model

**Authors:** Hamilton et al. (May 2025)  
**Read on:** [TO READ — Phase 1, Week 1-2]  
**Priority:** #4 — CLOSEST competitor, must understand deeply  

### Summary
[TO COMPLETE]

---

## Paper 5: HELMET

**Authors:** Yen, Gao, Chen (2024)  
**Read on:** [TO READ — Phase 1, Week 1-2]  
**Priority:** #5 — has partial coherence in summarization  

### Summary
[TO COMPLETE]

---

## Paper 6: InfiniteBench

**Authors:** Zhang et al. (2024)  
**Read on:** [TO READ — Phase 1, Week 1-2]  
**Priority:** #6 — 100K+ evaluation, confirm retrieval focus  

### Summary
[TO COMPLETE]

---

## Paper 7: Centering Theory

**Authors:** Grosz, Joshi, Weinstein (1995)  
**Read on:** [TO READ — Phase 1, Week 3]  
**Priority:** Theoretical foundation  

### Summary
[TO COMPLETE]

---

## Paper 8: Entity Grid Models

**Authors:** Barzilay & Lapata (2008)  
**Read on:** [TO READ — Phase 1, Week 3]  
**Priority:** Coherence measurement methodology  

### Summary
[TO COMPLETE]
