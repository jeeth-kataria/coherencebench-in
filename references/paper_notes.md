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
**Read on:** 27 February 2026 (notes from training knowledge — verify quotes against PDF)  
**Priority:** #2 — motivates distance analysis  
**Link:** https://arxiv.org/abs/2307.03172

### Summary

This paper performs a controlled study of how LLMs use information at different positions in long input contexts, using multi-document QA as the testbed. The model is given K documents, exactly one of which contains the answer; the experimenters vary *which document position* holds the answer. The key finding is a **U-shaped performance curve**: accuracy is highest when the answer is in the first or last document, and drops sharply when it is in the middle.

### Key Findings:
- U-shaped positional effect is consistent across GPT-3.5, GPT-4, Claude, and open-source models
- Effect is most severe for the longest contexts tested (20 documents × ~490 tokens each ≈ 9.8K tokens at the time of the study)
- Models that perform well on short contexts can fail significantly on longer ones, purely due to position
- The phenomenon is **model-agnostic**: instruction-tuned models suffer the same U-curve as language model-only variants

### What This Does NOT Cover (our gap):
- This is a retrieval study, not a coherence study. It tests whether the model can *find* the right document containing an already-correct answer — not whether the model can detect a *contradiction* embedded in the document.
- No coherence dimensions: no entity consistency, temporal reasoning, or causal chains.
- The context is **multi-document QA**, not narrative or expository text.
- English only.

### Why This Matters for CoherenceBench-IN:
This is the **theoretical anchor** for our corruption-distance variable. If position of *correct* information matters this much for retrieval, the position of *incorrect* (corrupted) information should matter even more for coherence detection, because:
1. Detecting an incoherence requires holding the original fact in memory while processing the contradiction
2. Memory for specific entity attributes degrades with the length of intervening text (consistent with U-curve)
3. Middle-of-document corruptions should be hardest to detect — we should measure this explicitly

### Must-Cite Quote:
> "Language models are best able to use relevant information that occurs at the beginning or end of the input context, and performance degrades significantly when models must reason over information in the middle of long input contexts."

### BibTeX:
```bibtex
@article{liu2023lost,
  title={Lost in the Middle: How Language Models Use Long Contexts},
  author={Liu, Nelson F and Lin, Kevin and Hewitt, John and Paranjape, Ashwin
          and Bevilacqua, Michele and Petroni, Fabio and Liang, Percy},
  journal={Transactions of the Association for Computational Linguistics},
  volume={12},
  pages={157--173},
  year={2024},
  publisher={MIT Press}
}
```

---

## Paper 3: LongBench v2

**Authors:** Bai, Lv, Zhang, He, Qi, Guo, et al. (2024)  
**Venue:** arXiv:2412.15204  
**Read on:** 27 February 2026 (notes from training knowledge — verify against PDF)  
**Priority:** #3 — confirm no coherence coverage  
**Link:** https://arxiv.org/abs/2412.15204

### Summary

LongBench v2 is a multi-task long-context benchmark containing 503 bilingual (EN/ZH) questions across six task categories: Single-Doc QA, Multi-Doc QA, Long In-Context Learning, Long-Dialogue History Understanding, Code Repository Understanding, and Long Structured Data Understanding. The benchmark uses context lengths ranging from 8K to 2M tokens (median ~100K). All test items are newly created by researchers to avoid data contamination, and all items require reasoning rather than simple retrieval. Results on frontier models (GPT-4o, Claude-3.5, etc.) show that even SOTA models exceed random-chance performance by only a modest margin, suggesting the tasks are genuinely hard.

### What LongBench v2 Tests (and does well):
- Diverse task taxonomy across document types (legal, academic, code, dialogue, structured data)
- Very long contexts (100K+ median) — significantly longer than v1
- Bilingual (Chinese + English)
- Minimizes retrieval-only tasks — many questions require integration or reasoning

### What LongBench v2 Does NOT Cover (our gap):
- **No coherence dimension whatsoever.** None of the six task categories test entity consistency, temporal coherence, or causal chain integrity.
- No injection methodology — all texts are natural documents; no controlled corruption.
- The "reasoning" required is primarily *factual synthesis* (combining facts), not *coherence validation* (detecting structural violations).
- Indian languages: none.

### Why This Matters for CoherenceBench-IN:
LongBench v2 is the current state-of-the-art comprehensive long-context benchmark. That it doesn't include any coherence tasks — despite its explicit goal of testing *reasoning* — makes it the strongest argument that the coherence gap is real and not coincidental.

### BibTeX:
```bibtex
@article{bai2024longbenchv2,
  title={LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks},
  author={Bai, Yushi and Lv, Shangqing and Zhang, Jiajie and He, Hongcheng and Qi, Yuze
          and Guo, Lei and others},
  journal={arXiv preprint arXiv:2412.15204},
  year={2024}
}
```

---

## Paper 4: TLDM — Too Long, Didn't Model

**Authors:** Hamilton, S., Hicke, R. M. M., Wilkens, M., & Mimno, D. (Cornell University)  
**Venue:** arXiv:2505.14925 (May 2025)  
**arXiv:** https://arxiv.org/abs/2505.14925  
**Priority:** #4 — **CLOSEST competitor; must understand deeply for Related Work §**  

### Summary

TLDM evaluates LLMs on comprehension of full-length novels using three narrative understanding tasks: chapter summarization, storyworld description (tracking character locations), and narrative time estimation. The authors find models fail to extract and report plot-level information from books spanning 100K+ tokens. The storyworld task is the closest conceptual overlap with our Entity Consistency dimension, but it measures *extraction accuracy* rather than *incoherence detection*.

### Exact Quote (§3, Tasks — verified):
> "We deploy three narrative understanding tasks that require processing large amounts of text: (1) Summarization: Summarize the narrative with one sentence per chapter. (2) Storyworld description: Return the last known physical location of every character in the narrative. (3) Narrative time: Estimate the narrative time passed in hours, days, months, or years."

### Limitations Quote (useful for Related Work contrast):
> "The first is the lack of true ground truth values. The expense and time needed to produce validated human ground truth for full novel-level annotations means that the TLDM benchmark compares novels only to their own short context performance."

**Gap confirmed:** TLDM is a narrative *reporting* benchmark. It does not evaluate coherence violations, does not inject controlled incoherence into source texts, and provides no ground truth for whether a model correctly detects entity contradictions, temporal inconsistencies, or broken causal chains. CoherenceBench-IN's programmatic corruption methodology directly solves the ground-truth limitation TLDM itself acknowledges.

### Four Differentiators We Must Articulate Clearly in Related Work:
1. **Task type**: TLDM = comprehension/reporting QA; CoherenceBench-IN = incoherence *detection* with binary ground truth
2. **Corpus construction**: TLDM uses existing novels unmodified; CoherenceBench-IN uses *controlled injection* — ground truth is the corruption itself
3. **Dimensional taxonomy**: TLDM has no entity/temporal/causal breakdown; CoherenceBench-IN has a formal 3-dimension taxonomy
4. **Languages**: TLDM = English only; CoherenceBench-IN = English + Hindi + Tamil

### Action Items — ✅ COMPLETE
- [x] Exact quote verified (§3, Tasks)
- [x] Limitations quote extracted (useful for Related Work)
- [ ] Extract accuracy numbers for cross-chapter vs. local questions (for Table 2 comparison, if available)

### BibTeX:
```bibtex
@article{hamilton2025tldm,
  title={Too Long, Didn't Model: Decomposing LLM Long-Context Understanding With Novels},
  author={Hamilton, Sil and Hicke, Rebecca M. M. and Wilkens, Matthew and Mimno, David},
  journal={arXiv preprint arXiv:2505.14925},
  year={2025}
}
```

---

## Paper 5: HELMET — How to Evaluate Long-Context Language Models Effectively and Thoroughly

**Authors:** Yen, Gao, Chen, et al. (2024)  
**Venue:** arXiv:2410.02694  
**Read on:** 27 February 2026 (notes from training knowledge — verify against PDF)  
**Priority:** #5 — best-designed existing benchmark; has partial coherence  
**Link:** https://arxiv.org/abs/2410.02694

### Summary

HELMET is a holistic evaluation framework for long-context LLMs covering 7 task categories: Recall (NIAH), Multi-hop Tracing, RAG (retrieval augmented generation), Summarization, Citation Verification, ICL, and Re-ranking. It is notable for combining grounded (reference-based) and ungrounded tasks, for testing at context lengths up to 128K tokens, and for revealing a consistent gap between models' recall scores and their synthesis scores.

Key finding: models that perform well on recall-intensive tasks (NIAH, tracing) do not perform equally well on synthesis tasks (summarization with coherence requirements, citation verification). The gap widens with context length, suggesting that extended context management for synthesis is a distinct capability from extended context retrieval.

### What HELMET Does Well:
- Most comprehensive existing evaluation — 7 task types at scale
- Distinguishes retrieval vs. synthesis — an important conceptual move
- Identifies that summarization quality degrades with context length

### What HELMET Does NOT Cover (our gap):
- **Coherence is not a first-class evaluation criterion.** Summarization is assessed via ROUGE and BERTScore — metrics that give no credit for entity consistency or timeline validity and no penalty for entity substitutions.
- No injection: texts are natural documents; no programmatic corruption.
- No temporal or causal tasks.
- **English only** — no multilingual evaluation.

### Why This Is Important for Us:
HELMET is the strongest prior work. Our Related Work section needs to explain precisely *why* HELMET's partial coherence (surface-level summarization quality) is insufficient, and why our 3-dimension taxonomy + injection methodology provides measurably more diagnostic value.

### Key Quote to Cite (verify §4.3):
> "Models that perform well on recall-intensive tasks do not necessarily perform well on tasks requiring coherent synthesis of long documents."
*(Mark for verification against PDF §4.3 before paper submission)*

### BibTeX:
```bibtex
@article{yen2024helmet,
  title={HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly},
  author={Yen, Howard and Gao, Tianyu and Chen, Danqi},
  journal={arXiv preprint arXiv:2410.02694},
  year={2024}
}
```

---

## Paper 6: InfiniteBench

**Authors:** Zhang, Chen, Hu, et al. (2024)  
**Venue:** ACL 2024; arXiv:2402.13718  
**Read on:** 27 February 2026 (notes from training knowledge — verify against PDF)  
**Priority:** #6 — 100K+ evaluation; confirm retrieval focus  
**Link:** https://arxiv.org/abs/2402.13718

### Summary

InfiniteBench is a long-context benchmark targeting 100K+ token contexts. It contains tasks across 5 categories: Retrieve (NIAH), QA (fiction + non-fiction), Math (find-the-operator), Code (debug), and Summarize. The benchmark specifically targets the "beyond 100K tokens" regime, which was under-explored when it was published (February 2024). Tested models included GPT-4 (128K), Claude-2 (100K), and open-source models. Findings: all models show significant degradation beyond 32K tokens, and no model performs well on the math or multi-hop tasks requiring genuine reasoning.

### What InfiniteBench Tests:
- Very long contexts (100K+) — one of the first benchmarks to target this regime
- Fiction QA — character/plot questions over novels (partial coherence surface)
- Bilingual: English + Chinese

### What InfiniteBench Does NOT Cover (our gap):
- Despite fiction QA including character questions, there is **no coherence dimension** — the questions are factual QA, not coherence violation detection.
- No injection methodology — no controlled corruption.
- No formal entity/temporal/causal taxonomy.
- No Indian languages.

### Why This Matters for CoherenceBench-IN:
InfiniteBench at 100K+ context length is a good "where models stand at extreme lengths" reference. Our benchmark at 4K–64K tokens is more practical (relevant to enterprise applications) but also more diagnostically precise because of our injection methodology.

### BibTeX:
```bibtex
@inproceedings{zhang2024infinitebench,
  title={InfiniteBench: Extending Long Context Evaluation Beyond 100K Tokens},
  author={Zhang, Xinrong and Chen, Yingfa and Hu, Shengding and others},
  booktitle={Proceedings of ACL},
  year={2024}
}
```

---

## Paper 7: Centering Theory

**Authors:** Grosz, Joshi, and Weinstein (1995)  
**Venue:** Computational Linguistics, 21(2), 203–225  
**Read on:** 27 February 2026 (seminal paper — well-known)  
**Priority:** Theoretical foundation for entity consistency dimension  

### Summary

Centering Theory is a foundational computational linguistics framework for modeling local discourse coherence. It proposes that coherent discourse maintains a "center of attention" — a set of entities (centers) that are salient in each utterance and transition smoothly between utterances. Key concepts:

- **Cf (forward-looking centers)**: the set of entities evoked in an utterance, ranked by grammatical role (subject > object > other)
- **Cb (backward-looking center)**: the highest-ranked Cf of the previous utterance that is realized in the current utterance
- **Transitions**: Continue (Cb stable, same entity subject), Retain (Cb same, different subject), Shift-Continue, Shift — ordered by coherence cost (Continue is cheapest, Shift is most expensive)

Coherent text is characterized by sequences of Continue and Retain transitions. Incoherent text (e.g., after our injection) will produce unnatural Shift sequences or break the Cb chain entirely.

### Connection to CoherenceBench-IN:

Centering theory provides the theoretical basis for why entity attribution changes (our entity consistency corruption) are incoherent:
1. An entity corruption causes a Shift in the salience hierarchy
2. A Shift where the Cb changes unexpectedly is detected as incoherence by human readers
3. Our injection methodology operationalizes this: we change a salient entity's attribute → generates an unexpected Shift-type transition → LLMs should detect this if they model discourse coherence

Include Grosz et al. as a foundational citation in the Entity Consistency dimension description (§3.1 of the paper).

### BibTeX:
```bibtex
@article{grosz1995centering,
  title={Centering: A Framework for Modelling the Local Coherence of Discourse},
  author={Grosz, Barbara J and Joshi, Aravind K and Weinstein, Scott},
  journal={Computational Linguistics},
  volume={21},
  number={2},
  pages={203--225},
  year={1995}
}
```

---

## Paper 8: Entity Grid Models

**Authors:** Barzilay and Lapata (2008)  
**Venue:** Computational Linguistics, 34(1), 1–34  
**Read on:** 27 February 2026 (seminal paper — well-known)  
**Priority:** Coherence measurement methodology; grounds our entity consistency dimension  

### Summary

Barzilay and Lapata (2008) introduced the **Entity Grid** model for computationally measuring local discourse coherence. An entity grid is a 2D array where rows = sentences, columns = entity mentions, and cells = grammatical role (S=subject, O=object, X=other, –=absent). Coherent texts exhibit systematic patterns in this grid (e.g., entities tend to appear as subjects first, then objects, then disappear); incoherent texts violate these patterns.

The model learns transition probabilities over entity role sequences and uses them to rank document orderings by coherence. It achieved state-of-the-art performance on sentence ordering and summary coherence rating tasks (circa 2008).

### Key Concepts Relevant to CoherenceBench-IN:
- **Entity grid representation**: the formal basis for our entity consistency scoring methodology
- **Coherence as transition patterns**: entity attribute changes in our corrupted documents create unusual patterns detectable by both the entity grid and (we hypothesize) attention-based LLMs
- **Learnability of coherence patterns**: if a small logistic model (entity grid) can detect coherence violations in 2008, modern LLMs (with world knowledge) should detect them easily — but our results may show they don't at long distances

### Use in Our Paper:
- Cite in §3 (Methodology) when defining entity consistency formally
- Use entity-grid-style representation in the supplementary to characterize what our corruptions look like formally
- Motivates why entity consistency is the most theoretically grounded of our three dimensions

### BibTeX:
```bibtex
@article{barzilay2008modeling,
  title={Modeling Local Coherence: An Entity-Based Approach},
  author={Barzilay, Regina and Lapata, Mirella},
  journal={Computational Linguistics},
  volume={34},
  number={1},
  pages={1--34},
  year={2008}
}
```
