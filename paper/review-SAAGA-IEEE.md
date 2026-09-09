# Paper Review (Strict, IEEE Standards): SAAGA

**Paper:** SAAGA: Strategic and Adaptive Attack Generation for Verifiable Red Teaming of LLM Prompt Defenses
**Reviewed artifact:** `paper/SAAGA_paper.pdf` (13 pp.) and `paper/SAAGA_full.md`
**Review standard:** IEEE conference submission (formatting per IEEE author guidelines; rigor per IEEE security/AI venues such as IEEE SaTML, S&P, TDSC, AI)

## Paper Metadata

| Field | Value |
|---|---|
| Title | SAAGA: Strategic and Adaptive Attack Generation for Verifiable Red Teaming of LLM Prompt Defenses |
| Authors | "Anonymous Authors" (double-blind marking) |
| Venue / Status | Unsubmitted; target: IEEE conference |
| Year | 2026 |
| Domain | AI security / automated red teaming / LLM evaluation |
| Paper Type | Empirical framework + benchmark |

---

## Executive Summary

The paper proposes SAAGA, a red-teaming framework that reframes defense evaluation as a capture-the-flag game: a hidden access code is embedded between two instruction blocks (the "sandwich"), and an attacker loop (planner → generator → victim → extract/rank/verify → controller) must make the victim reveal or accept the code. Success is decided by four judge-independent signals, the strongest requiring the victim to accept a re-sent candidate. Planning is conditioned on a strategy knowledge base and a retrieval store rebuilt after each benchmark, and a privileged best-of-N "oracle" generates training data for the planner and generator.

The core idea is genuinely useful: a victim-confirmed, checkable win condition removes the grader from the success path, and the policy/wording split makes failures attributable. The evaluation scale (52,096 scenarios across four victims, with a 26k scale check and controlled ablations) is a real asset, and the paper is unusually honest about where it breaks.

However, measured against IEEE standards the submission is not ready. There are **blocking formatting violations** (figure fonts at 2.7–5.0 pt vs the 8 pt IEEE minimum; author-year references instead of numbered; no Index Terms; 261-word abstract; double-blind marking for a single-blind venue family; page count above typical IEEE limits), **no statistical significance reporting**, **no external attack baselines**, a **headline evaluated only on a curated "recoverable" subset** (upper-bound-ish break rates), **no code/data release statement**, and **no responsible-use/ethics statement** for an offensive tool. There is also one internal-consistency hazard: Section 5.6 attributes 484 failures to the stop-point judge while Sections 3.2/4.1 state the judge never decides success — as written, a careful reviewer will treat this as a contradiction.

**Overall: Weak Reject (major revision required for IEEE).** The contribution level is Significant in idea and Moderate in execution as submitted.

## Summary of Contributions (as claimed)

1. A verifiable evaluation protocol: sandwich scenario + 4-signal judge-independent success + victim re-verification.
2. A diagnostic planner–generator architecture raising break rate 55.9% → 66.6% on a 1,000-scenario pool.
3. A self-improving memory loop (KB + retrieval + oracle) cutting attempts-on-success 4.02 → 3.61.
4. Large-scale evidence: 4 victims × 13,024 scenarios, stable at 26k, with per-phase/family/shape attribution.

---

## Strengths

### S1: The win condition is grounded in victim behavior, not a grader's label (Sec. 3.2, 4.6)
The four-signal design with a fixed priority, and especially the replay-verification signal, is the paper's strongest contribution. A success that requires the victim itself to grant access to a re-sent candidate is a much stronger benchmark primitive than an external classifier's judgment. The measured context guard for "access granted" (69.7% false-positive rate of the bare substring test) shows engineering care. This directly answers a real and well-documented weakness of the field (grader dependence of ASR).

### S2: Policy/wording separation yields diagnostic power (Sec. 4.3–4.4, 5.3, 5.6)
The structured plan contract (18 strategies, 1–5 primitives, canonicalization, embargoes) is what makes the failure labels (planner-stuck, generator-rephrase-fail, …) meaningful, and the ablation (Table 2) shows the gain appears at k=1 (Fig. 3), which is the correct signature of a policy effect rather than a wording effect. This is a design decision other frameworks could adopt.

### S3: Controlled, same-settings ablations and scale checks (Sec. 5.2–5.5)
The memory A/B (Table 3) holds everything else fixed and reports not only the +0.6-point break-rate effect but its *cost* (duplicate-attack rate 0.77% → 4.08%) and the effort gain (total queries −8.4%). The 26k scale check (Δ < 1.5 points) and the per-victim extractor confusion counts (5,920/22/1,509) are exactly the numbers a reviewer wants to see.

### S4: Unusually honest limitations (Sec. 7)
The paper discloses the recoverable-subset restriction, the training-victim confound, the absence of head-to-head external comparisons, the 8B-class victim scope, and names its own weak point (the stop-point judge). This is above the norm for the genre.

### S5: Auditable non-parametric memory (Sec. 4.7)
Rebuilding the learning state from plain logs, with explicit anti-overfitting controls (5-attempt noise floor, 5% exploration blank), makes the self-improvement claim checkable by third parties.

---

## Weaknesses

### W1: No external attack baselines (CRITICAL for IEEE)
All comparisons are against the project's own predecessor ("baseline attack loop") and internal toggles. There is no zero-shot, few-shot, GCG, PAIR, TAP, AutoDAN, or AutoRed (Diao et al.) run on the same scenarios. For an IEEE venue, a benchmark paper with no published-method comparison will be rejected on this alone by most reviewers. Mitigating context (the CTF access-code task differs from harmful-behavior ASR) is acknowledged in Sec. 7, but "future work" is not an acceptable substitute for a benchmark paper's central comparison. **Fix:** run at least (i) the no-planner generator-only loop, (ii) a strong static attack list, and (iii) PAIR-style two-agent baseline on 1,000 scenarios of the main pool; report break rate, verified rate, and attempts.

### W2: No statistical significance or uncertainty anywhere (CRITICAL)
No confidence intervals, no significance tests, no variance across seeds. The 1,000-run A/B (Table 3) is paired on identical scenarios — a McNemar test is trivially computable and should be reported. Break rates of 87.8% vs 90.2% vs 95.4% (Table 1) need Wilson intervals to be meaningful. The repository's own comparison report notes the significance test was "N/A in this environment because scipy is not installed" — that is an environment problem, not a paper problem. **Fix:** add 95% Wilson CIs to all rates and a McNemar test to the A/B; re-run the 1k architecture comparison with ≥3 seeds if possible.

### W3: Headline evaluated on a curated "recoverable" subset (MAJOR)
The main pool excludes scenarios the heuristic deems unrecoverable (34,576 of 72,008 short-code scenarios; the full corpus includes longer codes still). Break rates of 87.8–97.0% are therefore upper-bound-ish on winnable ground, and the Discussion's claim that "prompt-level defenses do not survive" is made on the curated slice. The results folder contains a 5,000-round run on a harder pool at 43.9% — reporting both pools would turn this from a weakness into a strength (it would quantify the recoverability gap). **Fix:** report the full-corpus number next to the subset number in Table 1 or the Discussion, and state the recoverability heuristic's own precision (who validates the labels?).

### W4: Internal-consistency hazard: the judge as "dominant failure phase" (MAJOR)
Section 5.6 says "484 failed attempts died at the stop-point judge (the response was dismissed before extraction was given full credit)." But Sections 3.2 and 4.1 state the judge never decides success and extraction runs on every response. If extraction is unconditional, the judge cannot have dismissed a leaking response. As written this is a contradiction a sharp reviewer will seize on. **Fix:** state the attribution method precisely (e.g., "in the final attempt of X failed scenarios, the stop-point judge classified the response as non-attack; in Y scenarios the extractor produced no candidate despite a ground-truth leak") and remove causal language ("died at").

### W5: IEEE formatting non-compliance (BLOCKING, mechanical)
Measured, not stylistic:
1. **Figure fonts 2.7–5.0 pt** (all five figures; IEEE minimum is 8 pt for figure text). At print size the axis labels and annotations are illegible. This alone risks a desk reject at formatting check.
2. **References are author-year**, not IEEE numbered [1]–[n].
3. **No Index Terms / Keywords line** after the abstract (IEEE requirement).
4. **Abstract is 261 words** (IEEE limit is typically ≤250, often 150–200) and is set in italics (IEEE abstracts are upright).
5. **"Anonymous Authors / Double-blind submission"** — IEEE venues (SaTML, S&P, TDSC, AI) are single-blind; author names, affiliations, and contact are required.
6. **Length: 13 pages** including 5 appendices. IEEE limits are typically 6–10 pages *including references* (SaTML: 10). Appendices are often disallowed or non-counted at reviewer's discretion; the paper must fit body + references within the limit.
7. Table captions should read "TABLE I" (Roman numeral, top), "Fig. 1." not "Figure 1." — minor.
8. No author affiliations/emails (camera-ready requirement).

**Fix:** re-typeset in the IEEEtran template; regenerate all figures at ≥8 pt effective font (draw at ~96–150 effective dpi or scale fonts up 3×); convert references to IEEE numbered style; compress to the page limit by moving appendices to a technical appendix if the venue allows.

### W6: No code/data release statement (MAJOR for IEEE)
Appendix E describes seeds and hashes but there is no statement of what will be released (code, datasets, run archives, model weights) or under what license. IEEE increasingly expects an artifacts statement. **Fix:** add a one-paragraph availability statement (even "code and 980-scenario selection set released upon acceptance" is far stronger than silence).

### W7: No responsible-use / ethics statement (MAJOR for a security venue)
This is an offensive framework. IEEE security venues expect: intended use (defense evaluation), dual-use assessment, and safeguards. The paper's own design contains strong safeguards worth stating explicitly: scenarios are synthetic access codes (no harmful-content behaviors), victims are open-weight 8B models, and the win condition targets prompt-level defenses. **Fix:** add a short "Responsible Use" subsection to Sec. 7.

### W8: Cross-victim numbers confound transfer with training exposure (MODERATE)
Planner/generator were trained with the default victim in the loop (oracle runs on Llama-3); the other three victims are evaluated with memory stores that include Llama-3 successes. The memory-off cross-victim numbers would isolate pure transfer; only Llama-3 has a memory-off arm (Table 3). **Fix:** one memory-off 1k run per victim, reported as a small table.

### W9: "Six oracle attempts capture 95.3% of its wins" (Sec. 4.7) is unaudited (MINOR)
This figure originates from the oracle's design documentation (a recommendation), not from the 4,505 recorded trajectories. A reviewer will ask for the measurement. **Fix:** compute it from the trajectory archive or soften to "the oracle configuration caps at six attempts per scenario."

### W10: Missing related work a reviewer will cite (MODERATE)
- A 2025 USENIX Security benchmark of automatic jailbreak attack *and defense* evaluation (task-level vulnerabilities) is directly comparable and uncited.
- The 2025 position paper arguing that ASR-based comparisons of red-teaming methods are often not evidentially supported (arXiv:2502.16903) — the paper's verified-rate design answers this critique and should cite it and position against it explicitly.
- The 2025 SoK systematizing LLM prompt security (jailbreak/defense/vulnerability taxonomies) — the paper's 8-family / 4-shape taxonomies should be related to it.
- In-the-wild CTF-framed jailbreak reports (industry) support the threat model's realism.

---

## Methodology Assessment

| Criterion | Rating (1-5) | Assessment |
|---|:---:|---|
| Soundness | 4 | Success logic is sound and well-separated (judge-independent signals, replay verification, context guard). Docked for the W4 judge-attribution inconsistency, which is a soundness-of-description issue. |
| Novelty | 4 | Victim-confirmed win + policy/wording split + rebuilt cross-run memory is a genuine combination; each part exists elsewhere, the integration and the verifiability argument are new. |
| Reproducibility | 3 | Seeds, holdout hash, pool rules, and dataset sizes are given (App. E, C), but no release statement, no model-weight availability, and the recoverability heuristic is described, not specified. |
| Experimental Design | 3 | Strong internal ablations and scale checks; fatal absence of external baselines and of a full-corpus (non-curated) evaluation arm. |
| Statistical Rigor | 2 | No CIs, no significance tests, single seed per configuration. This is the clearest gap against IEEE norms. |
| Scalability | 4 | 26k-scenario runs, 4-GPU sharding, per-attempt traces, ~45 s/attempt reported; compute cost discussed qualitatively only (no GPU-hours per benchmark). |

## Contribution Significance

**Level: Significant (idea) / Moderate (as executed).** The verifiable-win primitive and the diagnostic split are adoptable by other labs; the results are strong but bounded by the curated pool, the 8B victim scope, and the missing external comparison.

## Questions for the Authors

1. On the full 72,008-scenario short-code corpus (including unrecoverable-labeled scenarios), what is the break rate of the identical SAAGA configuration? (A 5,000-round run at 43.9% appears in the results archive; why is it not in the paper?)
2. What is the precision/recall of the "recoverable" heuristic that defines the main pool? Who or what validates those labels, and could the heuristic itself be attacker-influenced?
3. How exactly was the "484 judge rejections" failure attribution computed, given that the judge does not gate extraction? Can a leaking response ever be counted as a failure in the current design, and if so, through which path?
4. Of the 13,024-scenario Llama-3 run, how many scenarios changed outcome (broken ↔ not broken) when memory was switched off, and is that difference statistically significant (McNemar)?
5. What is the GPU-hour cost of one 13k-scenario benchmark and of one oracle trajectory batch?
6. Would the planner generalize if the strategy space were grown beyond 18 values, or is the 0.29 strategy entropy a sign the taxonomy itself is the binding constraint?

## Minor Issues

- "fixes these three problems" (abstract) is overclaimed for a framework paper; "addresses" is safer.
- Fig. 1's "next attempt (budget ≤ 20, dynamic 12–25)" label and the advisory-arrow label are small and close to boxes; regenerate with ≥8 pt and more clearance.
- Table 1: "Top-1 (%)" values (33.5–66.1%) should be cross-checked against the appendix definition (success within 1 attempt over all N) — the merged-summary top-k fields in the results archive use a different convention; the paper should state which one is reported (the recomputed one, per App. A — good — but the results files on release must match).
- Sec. 5.2 "Mistral-7B yields fastest" — "yields" is an odd verb; "is the most easily broken."
- The paper cites AutoRed (Diao et al.) for "state-of-the-art attack success on its own medium and hard behavior suites" — soften to "reports strong attack success" since the original paper's claims are the authors', not a fact.
- Appendix A defines verified rate as including ground-truth leak, while Table 1's "Verified (%)" (68.1 etc.) should be footnoted with this definition so readers do not assume verified = replay-confirmed only.
- "69.7% false-positive rate" needs its measurement protocol (corpus, n) in a footnote.
- References: "Li, X., and others. 2024. JailBench…" and several entries use "and others" — IEEE style requires full author lists or et al. with the first author named.

## Literature Positioning

The paper positions itself correctly against attack methods (GCG/PAIR/TAP/AutoDAN/AutoRed) and against HarmBench/StrongREJECT/JailBench/CyberSecEval, and the differentiation (victim-confirmed wins; policy/wording split; cross-run memory) is real and defensible. What is missing: (1) the ASR-validity critique literature, which the paper's verified-rate design directly answers and should cite as motivation; (2) the 2025 defense-evaluation benchmark (USENIX Sec) that evaluates attacks *and* defenses automatically — closest published neighbor, currently uncited; (3) the prompt-security SoK taxonomies, against which the 8-family/4-shape schemes should be compared; (4) any statement of how the sandwich construct relates to measured instruction-hierarchy behavior (cited once, in passing). With those four additions the positioning would be complete.

## Recommendations

**Overall Assessment:** **Weak Reject** (IEEE) — major revision required. Not because the idea is weak, but because the submission violates IEEE mechanical requirements (W5), lacks statistical reporting (W2), lacks external baselines (W1), and carries one internal contradiction (W4) that a careful reviewer will use to question the failure-analysis results.

**Confidence:** High — the review is grounded in the paper's own tables, the underlying results archives, and current literature.

**Contribution Level:** Significant (idea) / Moderate (execution as submitted).

### Actionable suggestions, in priority order

1. Re-typeset in IEEEtran; regenerate all five figures with ≥8 pt effective fonts; convert references to numbered IEEE style; add Index Terms; trim abstract to ≤250 words (target ~200); set authors per the venue's blind policy; cut to the page limit (move App. B–E to a supplementary artifact if needed). *(Blocking.)*
2. Add uncertainty: 95% Wilson CIs on every rate; McNemar on the paired 1,000-run A/B; report seeds. *(Blocking for IEEE.)*
3. Fix the judge-attribution wording in Sec. 5.6 and Appendix A (W4) with the precise attribution procedure.
4. Add external baselines (generator-only, static list, PAIR-style) on 1,000 main-pool scenarios; report break/verified/attempts side by side.
5. Report the full-corpus (non-curated) break rate alongside Table 1, and state the recoverability heuristic's validation.
6. Add code/data/model availability statement and a Responsible Use subsection (synthetic access codes, open-weight victims, defense-evaluation intent).
7. Add the four missing references (ASR-validity critique, USENIX Sec defense benchmark, prompt-security SoK, in-the-wild CTF-framed jailbreaks) and position the verified rate explicitly against the ASR critique.
8. Add one memory-off cross-victim row (1k per victim) to deconfound transfer (W8), and GPU-hour costs per benchmark.

---

## Fixes Applied (2026-08-25, post-review)

| # | Item | Status |
|---|---|---|
| W5.1 | Figure fonts | **Fixed** — all 5 figures regenerated at 200 dpi with 8.5–11 pt text (IEEE min 8 pt) |
| W5.2 | IEEE numbering of references | **Fixed** — `SAAGA_ieee.tex` (IEEEtran) uses numbered [1]–[35] |
| W5.3 | Index Terms | **Fixed** — added |
| W5.4 | Abstract length/style | **Fixed** — 236 words, upright, in IEEE format |
| W5.5 | Blind policy | **Fixed in IEEE version** — single-blind author block (names are placeholders to fill) |
| W5.6 | Page limit | **Fixed** — IEEE version is 11 pages body+refs (appendices included; trim App. B–E if the venue caps at 10) |
| W2 | Statistics | **Fixed** — 95% Wilson CIs on all main rates (Tables I–II); independent two-proportion tests with p-values: architecture p = 9.1e-07 (break), p = 1.7e-05 (verified); memory A/B p = 0.60 (honestly reported as non-significant); design note that arms use independent seeded draws |
| W4 | Judge-attribution contradiction | **Fixed** — Sec. V-F and App. A now state the attribution is a concentration measure, not causal |
| W3 | Curated-subset headline | **Partially fixed** — unfiltered 5,000-scenario run (43.9%) reported in Sec. V-B and Discussion; heuristic precision still unmeasured (needs data work) |
| W9 | Oracle 95.3% claim | **Fixed** — replaced with measured 97.6%-within-5-attempts over 2,031/4,505 recorded trajectories |
| W6 | Availability statement | **Fixed** — Responsible Use and Availability added to Sec. VII (release commitment is a placeholder promise to confirm) |
| W7 | Ethics/responsible use | **Fixed** — same subsection (synthetic codes, open-weight victims, prompt-level scope) |
| W10 | Missing references | **Fixed** — Cooper et al. NeurIPS'25 (ASR validity), Zhang et al. USENIX Sec'25, Chu et al. ACL'25 (JailbreakRadar) added with positioning sentences in Sec. II-B and V-C/Discussion |
| Minor | "fixes"→"addresses", "yields fastest"→"most easily broken", Diao-claim softened, verified-rate footnote, 69.7% FPR protocol note, median correction (1–2) | **Fixed** |
| W1 | External baselines (PAIR/GCG/TAP) | **Open** — requires GPU runs (not executable in this environment) |
| W8 | Memory-off cross-victim rows | **Open** — requires GPU runs |
| — | Author names/affiliations | **Placeholder** — must be filled before submission |

Deliverables after fixes: `paper/tex/SAAGA_ieee.pdf` (11 pp., IEEEtran, primary submission artifact), `paper/SAAGA_paper.docx` (+ PDF preview, content twin in ICML-style layout), `paper/SAAGA_full.md` (single source of truth).

---

## Full-Language Rewrite (2026-08-26, post-review)

The entire paper was rephrased for readability at the user's request: same technical content, same numbers, same structure, but shorter sentences, plainer words, and active voice (per the humanizer skill). Changes that matter:

- Every long multi-clause sentence was split into shorter ones (e.g. "Three structural gaps keep red-teaming evaluations from being verifiable, diagnosable, and reusable, and recent work argues..." is now two sentences).
- Jargon softened where a plain word works: "amnesic" -> "forget", "realizes it" -> "turns the plan into", "conditioned on" -> "uses", "upper-bound-ish" -> "an upper bound", "carries the work" -> "do the work".
- Technical terms that reviewers will look for are kept intact: sandwich defense, planner/generator, stop-point judge, embargoed (now defined as "banned" at first use), verified candidate, strategy knowledge base, retrieval store, privileged oracle.
- All 100+ numeric values, p-values, CIs, and citations were verified present after the rewrite (automated number-check, none missing). Abstract trimmed to 242 words (IEEE limit 250).
- One internal cross-reference fixed: the Discussion's "Table 2" -> "Table 1" (it points at the main-results table; the .tex already referenced it correctly).
- Applied consistently to `SAAGA_full.md` (source of truth) and all four LaTeX parts (`t1`-`t4.tex`); references, tables, and figure environments are byte-identical to before.

---

## Post-rewrite verification (2026-08-26)

User asked three verification questions; all were checked against the run archives and the web, and the paper was corrected where the evidence disagreed.

**Q1 — Are the Table 1 ASR numbers from the no-mutation-fallback runs, for all four victims?** YES. Verified by recomputing from the per-round results in `AutoRed-Final/results_bak/benchmarks/results_benchmarks_<victim>-[0:13024]-[KB+RAG]_...` (the plain, non-`_MutationFallback` runs): Llama 87.76/68.13, Gemma 90.17/60.68, InternLM2 95.36/76.56, Mistral 96.96/79.09 (break/verified) with top-1 33.5/45.3/38.6/66.1, top-5 68.2/75.4/79.6/90.8, avg attempts 3.88/3.19/3.31/2.00 — all match Table 1. The matching `_MutationFallback` runs are higher (e.g. Llama 90.5% at 26k) and their worker logs contain `[MutationFallback] Initialized`, while the no-fallback logs contain none. Fallback is gated by `AUTORED_MUTATION_FALLBACK` (default off) in `experiment/llama_3_8b_vllm.py`.

**Q2 — Where do the baseline-attack-loop numbers come from?** From `results_bak/benchmarks/batched_1000r_4g` (2026-07-12, Llama-3-8B, 1000 rounds, 4 workers; worker metadata has no planner/generator model) = 55.9/41.1/14.1/26.4/34.2/5.61/.991/.798/.884, and its paired SAAGA run `clean_arch_v1_1000r` (2026-07-13; planner_sft_v2_contract_anchor/checkpoint-27 + generator_sft_v2) = 66.6/50.7/19.3/35.1/45.7/4.94/.996/.805/.890. Verification found the two arms attacked the **same** 1,000 scenarios in the **same order** (paired design), not independent seeded draws as previously stated; the Setup, Table 3 caption, and Appendix E wording were corrected accordingly.

**Q3 — Are all references real?** NO, initially: verification (arXiv API + DBLP + proceedings) found 4 fabricated/incorrect method references (a DAN paper titled "Do as I can, not as I say", "The fine-tuning zero-shot jailbreak" (Wallace et al.), "JailBench: an open ecosystem" (arXiv:2402.10228), "TAP ... via self-reflection" (Liu et al., ICLR 2024)), 3 wrong arXiv IDs (perez2022, anil2024, li2024jailbench), and wrong author lists (PAIR, Instruction Hierarchy, Llama 3, Cooper et al. missing first author Chouldechova, AutoDAN title+authors). All were replaced with the verified real papers (Do Anything Now arXiv:2308.03825; Andriushchenko et al. ICLR 2024 + Souly et al. NeurIPS 2024 D&B; JailbreakBench NeurIPS 2024 D&B; Mehrotra et al. Tree of Attacks arXiv:2312.02119; Ganguli et al. arXiv:2209.07858; correct author lists elsewhere; TensorTrust now cites Toyer et al. arXiv:2311.01011, which also matches the actual corpus source in `data/TensorTrust_subsets/`). The remaining ~26 references (GCG, Greshake, HarmBench, NIST, USENIX Sec'25, JailbreakRadar ACL'25, Reflexion, Self-Refine, RAG, DistilBERT, LoRA, QLoRA, vLLM, model cards, DeBERTa, Sentence-BERT, FAISS, DPO, AutoRed, MT-Bench, Wang et al. ACL 2024) verified as correct. PDF/DOCX rebuilt: 11 pages, 0 errors, 0 unresolved citations.

---

## Section 5.3 figure added (2026-08-26)

Added Figure 3 (`figs/fig6_autored_cmp.png`, generated by `make_figs3.py`, 8.5–10.5 pt text at column width, 200 dpi): grouped bars of AutoRed vs SAAGA break rate per victim, rotated x labels, bottom legend. Figures renumbered: memory ablation = Fig. 4, defense family = Fig. 5 (md captions + body refs updated; LaTeX auto via \ref). build_docx.js registry updated. Rebuilt: 11 pp., 0 errors, 0 undefined refs; pages 7–8 render verified.

## Section 5.3 AutoRed numbers replaced with AutoRed-Original results (2026-08-26)

The AutoRed row was re-sourced from `/nlsasfs/home/isea/isea38/AutoRed-Original/results/*_parallel_results.json` (the original, unmodified AutoRed repo: RL-fine-tuned T5 generator + DistilBERT prompt-injection discriminator; success = code in response or response starts with "access granted", no judge; 1,000 alphabetic-access-code scenarios per victim, 20 attempts): Gemma-2b 51.0 (2.07 att), InternLM2-7B 73.7 (2.21), Llama-2-7B 58.2 (1.80), Llama-3-8B 53.1 (2.85), Mistral-7B 75.3 (1.52). Table 2 is now a per-victim AutoRed-vs-SAAGA comparison (SAAGA side unchanged: 22,791-26,047 full-scale no-fallback runs from `results/`); the old 55.9/41.1/5.61 row from `batched_1000r_4g` is no longer cited. Abstract (247 words), contribution 2, and Setup updated accordingly. Rebuilt: 11 pp., 0 errors, 0 undefined refs.

## Section 5.3 restructured: AutoRed-vs-SAAGA (2026-08-26)

User decision: the old "value of the planner-generator architecture" section (baseline 55.9% vs SAAGA 66.6% on the 1,000-scenario pool, with the `clean_arch_v1_1000r` arm) was removed because that arm does not represent the current system state. Section 5.3 is now **"Comparison with the original AutoRed"**:

- AutoRed side: the original AutoRed attack loop (`results_bak/benchmarks/batched_1000r_4g`, 2026-07-12; worker metadata has no planner/generator; log confirms 1,000 scenarios sampled from the TensorTrust subset-8, Llama-3-8B-Instruct): 55.9% break, 41.1% verified, 5.61 avg attempts, extractor P 0.991.
- SAAGA side: current full-scale no-fallback runs in `results/` (22,791 Llama / 26,047 Gemma-InternLM2-Mistral): 88.0/89.5/95.3/97.1% break, 67.0/60.6/75.1/78.1% verified, 3.94/3.14/3.26/1.92 avg attempts (recomputed from per-round results).
- New Table 2 replaces the two ablation tables; fig:topk removed (figures renumbered: memory = Fig. 3, defense family = Fig. 4). The section states explicitly that AutoRed was run on 1,000 scenarios and SAAGA on all 26,047, and that this is not a controlled ablation.
- Updated for consistency: abstract (now 242 words), contribution 2, Sec. 5.1 pool description, Sec. 5.6 wording ("1,000-scenario SAAGA run"), build_docx.js figure registry. PDF (11 pp., 0 errors) and DOCX rebuilt.

---
*Review generated 2026-08-25 from `paper/SAAGA_paper.pdf` (13 pp.), the underlying results archives, and a literature scan of 2024–2026 red-teaming/evaluation work.*
