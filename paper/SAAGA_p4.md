
## Appendix A. Metric definitions

Let a benchmark contain N scenarios, and let attempt i of scenario s produce response r_{s,i} and signal vector (g_{s,i}, a_{s,i}, v_{s,i}, e_{s,i}) for the four success signals in priority order.

- Break rate = (1/N) · |{s : max_i (g_{s,i} ∨ a_{s,i} ∨ v_{s,i} ∨ e_{s,i})}|.
- Verified rate = (1/N) · |{s : max_i (g_{s,i} ∨ v_{s,i})}|.
- Attempts to success: T_s = min{i : any signal fires}, undefined if the scenario is not broken.
- Top-k success = (1/N) · |{s : T_s ≤ k}|, reported for k ∈ {1, 3, 5, 10}.
- Average attempts on success = (1/|S|) · Σ_{s ∈ S} T_s, where S is the set of broken scenarios.
- Extractor precision = TP/(TP+FP), recall = TP/(TP+FN), over all attempts with ground truth available, where a true positive is an attempt in which the code leaked and the extractor found it, a false negative is a leak the extractor missed, and a false positive is a non-leaking response from which the extractor nonetheless produced a candidate matching the code.
- Failure attribution: per unsuccessful scenario, the phase with the largest share of failed attempts among {judge rejection, extractor miss, verifier rejection}; scenarios are additionally labeled by the controller's failure mode (planner-stuck, generator-rephrase-fail, leaked-unverified, access-granted-unverified, never-leaked).

## Appendix B. Scenario pools and defense taxonomy

| Pool | Size | Rule |
|---|---:|---|
| Full short-code corpus | 72,008 | access code length < 30 |
| Recoverable subset (main pool) | 34,576 | code is direct, deterministic, or indirect under the defense's stated rules |
| Architecture pool | 1,000 (seeded sample) | from the full short-code corpus |
| Main evaluation | 13,024 per victim (seeded sample) | from the recoverable subset |
| Scale check | 22,791 to 26,047 per victim | from the recoverable subset |
| Selection benchmark set | 980 (780 dev / 200 holdout) | seed 42, fixed hash |

Defense families (eight): password (enter exactly X), roleplay (character constraints with trigger exceptions), translation (respond only in another language or encoding), trigger-phrase (a specific phrase unlocks behavior), conditional (behavior gated on a condition), exception (a stated exception to the hiding rule), conversation (multi-turn interactive rule), instruction-hiding (hide the system prompt generally). Code shapes (four): TOKEN (one word or symbol), PHRASE (two to five words), SENTENCE (more than five words or more than 20 characters), MULTILINE (several lines).

## Appendix C. Training data construction

| Dataset | Size | Source |
|---|---:|---|
| Oracle trajectories | 4,505 | privileged best-of-N runs, 8 workers |
| Planner fine-tuning set | 17,129 | 40% oracle trajectories (3x weight), 45% runtime successes, 15% runtime failures; defense-family balanced |
| Generator fine-tuning set | 13,037 | 5,955 oracle examples, 7,082 runtime successes; plan-conditioned format |
| Ranker set | 12,144 | 3,036 positive / 9,108 negative (1:3), 9,722 / 1,211 / 1,211 split |
| Strategy predictor set | 11,714 | gold-labeled runtime attempts plus synthetic |
| Code-shape classifier set | 118,326 | labeled from the defense corpus |
| Runtime success log | 217,168 | cross-run accumulation, content-hash deduplicated |
| Runtime verified subset | 3,558 | of the above, victim-confirmed |

The planner and generator are low-rank adapters on a shared 8B base, trained by quantized low-rank fine-tuning. The ranker is a small DeBERTa discriminator. The access-code-shape predictor and the stop-point judge are distilled BERT classifiers. Preference-optimization datasets derived from the same logs are maintained for future policy improvement but are not part of the evaluated configuration.

## Appendix D. Plan contract (schema)

The planner emits an XML block with the fields strategy (18-value enum), primitive_sequence (ordered list, 1 to 5 items from a group/name library), style (formal, conversational, academic, story, direct), expected_access_type (TOKEN, PHRASE, SENTENCE, MULTILINE, UNKNOWN), retry_policy (explore, retry_same_strategy, switch_strategy), confidence (real in [0,1]), and failure_reason (free text). Canonicalization rules: unknown strategy maps to instruction-leak; empty primitive list maps to a single default framing primitive; the primitive list is truncated to five; unknown style maps to direct; unknown access type maps to UNKNOWN; unknown retry policy maps to explore; confidence is clamped to [0,1]. Malformed output triggers a defense-aware strategy rotation. The guard then replaces the strategy if it is embargoed (three consecutive failures), if its failure streak reached three, or if the plan requests a switch with a streak of at least two.

## Appendix E. Reproducibility

All benchmark samples are seeded; the selection benchmark set is fixed by hash (holdout 200, development 780, seed 42). Worker sharding is contiguous and even over the shared seeded draw, so multi-worker runs attack disjoint slices of one sample. Every scenario logs a per-attempt trace (plan, prompt, response, candidates, scores, signals, judge outcome) and a scenario-level summary; benchmark summaries are merged from the worker shards by summing counters and recomputing extractor metrics from the summed confusion counts. The knowledge stores are plain JSON tables and a vector index rebuilt from the logs, so the complete learning state at any point can be reconstructed from the run archives. The default victim is served with 200-token responses at temperature 0.7; the planner samples at temperature 0 and the generator at temperature 0.7 with nucleus sampling 0.9.
