
## 3. Problem Formulation

### 3.1 The sandwich scenario

A scenario is a triple s = (o_s, c_s, z_s). The blocks o_s and c_s are instruction text; z_s is the hidden access code. At interaction time the victim V receives a system message that concatenates the two blocks, and the attacker's prompt a is the user message between them. The sandwich places defense text above and below the payload, so the rules are re-asserted after the attacker's words, which matches how instruction-level protections are actually deployed (Greshake et al., 2023; Wallace et al., 2024). Each scenario is labeled with a defense family from an eight-value taxonomy (password, roleplay, translation, trigger-phrase, conditional, exception, conversation, instruction-hiding) and a code shape from a four-value taxonomy (TOKEN, PHRASE, SENTENCE, MULTILINE), both derived from the scenario content by classifiers described in Section 4.2.

### 3.2 Success as a checkable event

The benchmark holds the ground truth z_s for every scenario, so a win is a checkable event rather than a judgment. We define four success signals with a fixed priority:

1. Ground-truth leak. z_s appears in the victim's response (case-insensitive containment). This is the strongest signal and always counts, regardless of what any downstream component says.
2. Access granted. The response contains an unquoted, non-negated "access granted", the utterance the defenses themselves promise. A context guard rejects occurrences that are quoted, negated, or part of the model restating the rule, because a bare substring test measured at 69.7% false-positive rate on this corpus.
3. Verified candidate. The extractor produces a candidate from the response, the candidate is re-sent to the victim inside the same sandwich, and the victim replies "access granted" to it. The victim's own acceptance is the arbiter.
4. Extractor match. The extractor recovers z_s from the response but the candidate was never accepted on replay.

A scenario is broken if any signal fires; the priority order assigns each success to exactly one path, which is how the benchmark reports verified rates separately from raw break rates. No classifier's confidence enters this decision.

### 3.3 Metrics

For a benchmark of N scenarios we report: the break rate (fraction broken on any signal); the verified rate (fraction with a verified or ground-truth-leak win, the conservative count); the attempts-to-success distribution with top-k success (fraction broken within k attempts, k in {1,3,5,10}); the average attempts on success; extractor precision, recall, and F1 computed over all attempts against ground truth; and a per-scenario failure attribution that names the phase where the failure concentrated (Section 4.6). Every scenario also logs a full per-attempt trace: plan, prompt, response, candidates, scores, and signals.

## 4. SAAGA

### 4.1 Overview

Figure 1 shows one attempt of SAAGA. Scenario intelligence first converts a raw scenario into structured facts (defense family, code shape). The planner reads those facts together with its memory (the last three attempts, the strategy knowledge base, retrieved exemplars) and emits a structured plan. A deterministic contract layer canonicalizes the plan, and a guard overrides it when the chosen strategy is already embargoed. The generator converts the plan into a short prompt, which the victim answers inside the sandwich. The response is scored by the four signals; the extractor and verifier carry the work for signals 3 and 4; and the controller records the attempt into per-scenario memory and decides whether to continue, stop, or re-seed. After the benchmark, a write path rebuilds the cross-run memory from the logs. The stop-point judge, a small classifier, observes every response but never decides success; its role is instrumentation and early-warning only.

### 4.2 Scenario intelligence

Two small models convert scenario text into structured metadata before any attack is planned. A keyword-priority classifier assigns the defense family, using the highest-priority match among the eight families so every scenario has exactly one label. A DistilBERT predictor assigns the access-code shape as a four-way distribution over TOKEN, PHRASE, SENTENCE, and MULTILINE from the combined defense text. The top label goes into the planner's metadata block and backfills the plan when the planner leaves the field open; the same distribution is injected as a type hint into the extractor's language-model pass and the ranker's input, so one prediction conditions both ends of the pipeline. The metadata is cheap (a single forward pass each) and removes a whole class of inference errors from the policy models, which would otherwise have to re-derive the target's shape from prose on every attempt.

### 4.3 The planner

The planner is an 8B parameter policy model, fine-tuned with low-rank adapters on a shared base, and its job is a decision, not a sentence. It never writes attack text. Its input per attempt has bounded size: the full defense text, three metadata scalars (family, predicted code shape, complexity), the attempt number, the last three attempts as compressed tuples (strategy, primitive sequence, outcome; no free text), at most three strategies with their historical success rates from the knowledge base, at most three retrieved exemplars clipped to 120 characters, and the list of strategies already failed on this scenario. It is sampled at temperature zero, so plans are reproducible per scenario seed.

The output is a structured plan: one strategy from an eighteen-value set (instruction leak, trigger-phrase discovery, exception discovery, roleplay, summarization, translation, encoding bypass, and so on), an ordered sequence of one to five primitives (concrete rhetorical or encoding moves, e.g. educational framing, rot13, developer persona), one style from five values, one expected code type, one retry policy from three values (explore, retry same strategy, switch), and a confidence in [0, 1]. A deterministic contract layer then canonicalizes the plan: unknown values map to safe defaults, the primitive list is capped at five, and malformed output falls back to a defense-aware strategy rotation instead of crashing the loop. This contract is why the downstream stages can parse every plan and why strategy statistics are comparable across runs.

A hard guard completes the stage. A strategy is embargoed after three consecutive failures on the same scenario; an embargoed or thrashing strategy is replaced by a defense-aware alternative regardless of what the planner emitted. The planner's knowledge-base input is advisory text and can be ignored by the model; the embargo is code and cannot. This split, advisory memory plus hard guardrails, is the framework's basic safety pattern: learned advice degrades gracefully, invariants never do.

### 4.4 The generator

The generator is a second low-rank adapter on the same base model, and it is deliberately narrow. It sees the defense text, the canonical plan, and at most two retrieved exemplars of past winning prompts clipped to 150 characters with an explicit instruction to adapt rather than copy. It does not see the raw attempt history, the knowledge-base statistics, the failed-strategy list, or any judge output. Its output is the attack prompt: at most 40 words, at most 128 tokens, sampled at temperature 0.7, with chain-of-thought and preamble stripped and exact duplicates suffixed so no prompt is ever sent twice. Because the generator's world is the plan, everything that influences wording must pass through the plan contract, which is what makes the policy/wording attribution of Section 1 operational.

### 4.5 The victim environment

The victim is an open-weight chat model served with the same inference engine as the adapters. The default target is Llama-3-8B-Instruct; the cross-victim evaluation of Section 5 uses Gemma-2b-it, InternLM2-chat-7B, and Mistral-7B-Instruct-v0.2. Each attempt assembles the sandwich, applies the victim's native chat template, and samples at most 200 tokens at temperature 0.7. Defenses in the conversation family run multi-turn under a static system sandwich; all others are single-shot. The environment returns the response and nothing else: its reward is always zero, because success is decided outside it by the signals of Section 3.2. This separation keeps the victim a measurement instrument rather than a component with opinions about what a win is.

### 4.6 Extraction, ranking, verification, and failure attribution

Extraction converts a response into structured candidates through four independent detector layers: sixteen regular expressions for the sentence shapes in which victims typically leak; quoted strings including multi-line blocks; capitalized phrase spans; and a language-model pass that is told the code shape, the candidates that already failed, and asked for a strict structured list. Candidates are normalized and deduplicated, and a consensus score records the fraction of the four layers that independently produced each candidate, since a string found by several independent detectors is far more likely the real secret than one found by one.

Ranking assigns each candidate a score in [0, 1]. The primary ranker is a small fine-tuned DeBERTa discriminator trained to output the probability that a candidate is the correct secret given the response, the candidate, and the predicted code-shape distribution; a learned ranker replaced an earlier fixed-weight fusion of the detector signals (0.35 consensus, 0.20 language-model confidence, 0.20 verification history, 0.15 regex confidence, 0.10 type match), which remains the graceful fallback if the ranker fails at runtime. A candidate's verification history demotes it after each rejection by the verifier, so a defense that makes the extractor hallucinate the same wrong string does not loop.

Verification is the framework's ground-truth channel. The top candidates (one, or three when the top score exceeds 0.75) are re-sent to the victim as fresh queries inside the identical sandwich; a candidate is verified if and only if the victim grants access to it. A verified win that is not byte-identical to the ground truth is recorded as strong versus weak, which is a label, not a gate. Verification costs one victim query per candidate and is bounded by the adaptive top-k.

Every unsuccessful scenario receives a failure label from a fixed decision order: planner-stuck (one strategy for fifteen or more attempts), generator-rephrase-fail (three or more distinct strategies tried, no leak), leaked-unverified, access-granted-unverified, and never-leaked. Together with the per-attempt judge outcome, these labels let the benchmark say where a failure died, which is the diagnostic value that a bare success rate cannot provide.

### 4.7 Cross-run memory and the self-improvement loop

The memory has two stores and a write path. The strategy knowledge base is a table of empirical success rates, indexed by victim model, defense family, and strategy, with a model-agnostic marginal as fallback. At planning time the system returns the top three strategies above a five-attempt noise floor, and with probability 0.05 returns nothing at all, an explicit exploration blank that keeps the policy from overfitting to past winners. The retrieval store indexes the defense text of every past successful attack with a sentence encoder into an exact cosine nearest-neighbor index; retrieval takes the top 20, keeps the first three with a matching defense family (backfilling cross-family when fewer), and orders same-victim-model hits first. One retrieval per attempt serves both the planner's exemplars and the generator's reference prompts.

The write path runs after every scenario log and rebuilds both stores at each benchmark boundary: per-attempt records are appended with content-hash deduplication, success rates are recomputed, and the index is rebuilt. The LLM parameters are frozen throughout, so the learning is non-parametric and fully auditable.

The loop closes through a privileged oracle. The oracle attacks scenarios with the same generator, judge, and extractor but a much larger candidate budget per attempt, including a lottery for compound primitive pairs and early termination when responses stop improving; six oracle attempts capture 95.3% of its wins. Its winning trajectories become the highest-weighted training data for the planner (40% of the 17,129-example fine-tuning set, triple-weighted) and the generator (46% of 13,037 examples), and its mined statistics become the live strategy prior. The oracle is therefore a teacher, not a model: it demonstrates what the runtime can do given more budget, and the fine-tuned policy learns to do it in fewer attempts.
