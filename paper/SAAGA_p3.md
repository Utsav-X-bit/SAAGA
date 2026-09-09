
## 5. Experiments

### 5.1 Setup

Victims. We evaluate four open-weight instruction-tuned models spanning 2B to 8B parameters: Llama-3-8B-Instruct (default target), Gemma-2b-it, InternLM2-chat-7B, and Mistral-7B-Instruct-v0.2. The planner and generator weights are identical across all four; the victims are never retrained or tuned per model, so the cross-victim comparison measures the framework's transfer, not a per-victim fit.

Pools. Two pools serve different questions. The architecture pool is a seeded sample of 1,000 scenarios drawn from a 72,008-scenario corpus of short access codes, and it hosts the baseline-versus-SAAGA comparison of Section 5.3. The main pool is a 34,576-scenario recoverable subset of the same corpus, restricted by heuristic labels to scenarios whose code is directly, deterministically, or indirectly recoverable under the defense's own stated rules; we evaluate 13,024 seeded scenarios per victim and repeat at 22,791 to 26,047 scenarios to check scale. A fixed 980-scenario benchmark set (780 development, 200 holdout, seed 42) is used for model selection during training. All runs use a budget of up to 20 attempts per scenario, four GPU workers with disjoint contiguous shards of the same seeded draw, and identical sampling parameters across arms of every comparison.

### 5.2 Main results

Table 2 summarizes the main evaluation. SAAGA breaks every victim, and the break rate and the verified rate both stay high, which means the wins are not extraction artifacts: 60.7% to 79.1% of scenarios end in a win the victim itself confirms.

Figure 2 shows the same numbers as break and verified rates per victim. Mistral-7B yields fastest, 97.0% broken in 2.0 attempts on average with a 66.1% first-attempt success; InternLM2-7B is next at 95.4%; Gemma-2b sits at 90.2%; and Llama-3-8B resists longest at 87.8% with 3.9 attempts. Two patterns stand out. First, difficulty is not monotone in model size: the 2B Gemma is harder to break than the 8B Llama-3, so size is not the relevant axis and the defense family is (Section 5.7). Second, the first-attempt success rates (33.5% to 66.1%) show that the planner's first plan is already strong; adaptation over subsequent attempts adds the remaining 54 to 64 points of the break rate.

Scaling to 22,791 to 26,047 scenarios per victim changes every break rate by less than 1.5 points (for example 87.8% to 88.0% for Llama-3-8B), so the main-table numbers are stable estimates, not small-sample luck.

Takeaway. Against four victims and 52,096 scenarios, prompt-level sandwich defenses do not survive an adaptive, memory-conditioned attacker: the break rate is 87.8% to 97.0% and the verified rate 60.7% to 79.1%, with a median of two to four attempts per break.

### 5.3 The value of the planner-generator architecture

Table 3 compares the SAAGA architecture against the baseline attack loop that generates prompts directly from the defense and the last response, on the 1,000-scenario architecture pool with identical victims, budgets, and workers. The architecture raises the break rate from 55.9% to 66.6% (+10.7 points) and the verified rate from 41.1% to 50.7% (+9.6 points). The improvement is fastest where it matters most: top-1 success rises from 14.1% to 19.3% and top-5 from 34.2% to 45.7%, while the average attempts on success falls from 5.61 to 4.94. Extractor quality improves in lockstep (precision 0.991 to 0.996, recall 0.798 to 0.805), which we attribute to cleaner, plan-driven prompts producing less degenerate victim text.

Figure 3 plots the cumulative success curves of both systems. The SAAGA curve dominates at every k, and the gap is already open at k = 1, which is where a pure phrasing effect would be smallest and a policy effect largest.

Takeaway. Separating the strategy decision from the wording decision buys +10.7 points of break rate and 13% fewer attempts on success on the same pool, and the gain appears on the very first attempt, which is the signature of a better policy rather than a better writer.

### 5.4 The value of memory

Table 4 is a controlled 1,000-run comparison on the main pool with the strategy knowledge base and retrieval store switched off, with every other setting held fixed in both arms. Memory raises the break rate from 92.6% to 93.2%, a small effect on this pool, but it cuts the average attempts on success from 4.02 to 3.61 (10.2% fewer) and the total victim queries from 5,827 to 5,339 (8.4% fewer), and it lifts top-3 success by 8.9 points. The retrieval store hits in 98.8% of scenarios. The cost is visible and bounded: the duplicate-attack rate rises from 0.77% to 4.08%, the expected overfitting to past winners, which the five-attempt noise floor and the 5% exploration blank are designed to hold down.

Takeaway. Memory's value is efficiency, not raw break rate, on a pool that is already 92.6% breakable: it converts a win that took four attempts into one that took two to three, and it does so at a small, controlled cost in prompt diversity.

### 5.5 Extraction quality

At full scale (13,024 Llama-3-8B scenarios) the extractor records 5,920 true positives, 22 false positives, and 1,509 false negatives across all attempts: precision 0.996, recall 0.797, F1 0.885. The same profile holds across victims and across the 26,047-scenario runs (recall 0.69 for Gemma-2b, 0.82 for InternLM2-7B, 0.81 for Mistral-7B), and the false-positive rate never exceeds 0.3%. Verification, not extraction, is what keeps the success signal honest: the 22 false positives at 13k scale are candidates the extractor believed in that the victim did not accept.

Takeaway. Extraction is high-precision by construction (consensus plus learned ranking plus replay), and the residual error is a recall gap, 1,509 missed leaks out of 7,429 that occurred, which is the clearest single improvement target in the pipeline.

### 5.6 Where the framework fails

Figure 4 breaks the 1,000-scenario architecture pool by defense family, using the verified success rate. Password-style defenses fall at 66.1% (n = 419). The weak families are translation at 38.6% (n = 236), roleplay at 44.5% (n = 211), and trigger-phrase at 33.3% (n = 78); conditional (41.4%), instruction-hiding (41.2%), and the small exception and conversation cells (n = 8 and n = 2) sit in between. Table 5 splits by code shape: single-token codes are broken at 56.9% while phrase and sentence codes fall to 33.8% and 29.8%.

The failure labels locate the residual error inside the pipeline. Across the 1,000-scenario archive, 484 failed attempts died at the stop-point judge (the response was dismissed before extraction was given full credit), 4 at the extractor, and 0 at the verifier; the verifier has never rejected a ground-truth leak in this archive. The planner's policy is concentrated by design: strategy entropy is 0.29, instruction-leak accounts for 96.5% of attempts, and the retry-to-switch transition ratio is 8,838 to 131. The generator is disciplined, with an average prompt of 182.5 characters and a 0.65% repeat rate. First-plan success is 14.9% of scenarios, so adaptation does most of the work after attempt one.

Takeaway. The framework's remaining failures concentrate in two places: defense families that punish the concentrated policy (translation, roleplay, trigger-phrase) and long-form secrets that the extractor misses, with the stop-point judge as the dominant single failure phase. Because every failed scenario carries its label, these are addressable targets rather than a 33% residual.

### 5.7 The self-improvement loop in operation

The loop is observable in the data. During the 1,000-scenario archive the write path harvested 9,120 distinct winning prompts and 512 verified trajectories into the cross-run stores, and the retrieval store's hit rate in the following 1,000-run comparison is 98.8%. The oracle that seeds the training data runs 4,505 recorded winning trajectories, which become the highest-weighted portion of both fine-tuning sets (Section 4.7). Because the stores are plain tables and indexes rebuilt from logs, the entire learning state is auditable: any plan's advisory input can be traced to specific past scenarios.

## 6. Discussion

The break rates in Table 2 should be read with two qualifications. The main pool is restricted to scenarios that are heuristically recoverable under the defenses' own stated rules, so the 87.8% to 97.0% figures measure how well prompt-level defenses hold against an adaptive attacker on winnable ground, not the break rate of the full 72,008-scenario corpus, which includes codes the defenses are constructed to make unrecoverable. And the victims are 2B to 8B open-weight models; we make no claim about frontier-scale models, whose refusals are stronger and whose responses are harder to extract from.

Within those bounds the result is that instruction-level defenses on current open-weight models are fragile to an attacker that plans, remembers, and verifies. The verified rate, which requires the victim itself to accept the recovered code, stays at 60.7% to 79.1% of the break rate, so the breaks are not measurement artifacts. The difficulty ordering (Mistral, InternLM2, Gemma-2b, Llama-3) is not a size ordering, which argues that defense behavior, alignment tuning, and refusal style matter more than parameter count at this scale.

For evaluation practice, the design decisions transfer beyond this system. A win condition that the victim can confirm on replay is a stronger benchmark primitive than a grader's label, and it removes the judge from the success path entirely. A strategy/wording split turns a success rate into a diagnosis. And a non-parametric memory with explicit exploration controls lets a benchmark improve with use while staying auditable, which is the property we consider most important for a tool other labs will build on.

## 7. Limitations

We evaluate four open-weight victims from 2B to 8B parameters and draw no conclusions about frontier models. The main pool is restricted to heuristically recoverable scenarios, so the break rates are upper-bound-ish on what the defenses can hold; the unrecoverable remainder of the corpus is out of scope by construction. The planner and generator were trained with the default victim in the loop, so the cross-victim numbers measure transfer from, and not pure generality to, the training victim. We do not run head-to-head comparisons against PAIR, GCG, or TAP on identical scenarios, because those methods target harmful-behavior suites measured by external judges, while SAAGA's task is a different one (recovering a concrete secret under a stated defense); a shared-protocol comparison is future work. The strategy space is fixed at eighteen values, the defenses are prompt-level rather than model-level, and we report compute only qualitatively (four-GPU workers, roughly 45 seconds per attempt on the default victim). The stop-point judge, the dominant failure phase in Section 5.6, remains in the pipeline for instrumentation even though it does not decide success, and its reliability is the framework's known weak point.

## 8. Conclusion

SAAGA evaluates prompt defenses the way a capture-the-flag competition evaluates a lock: with a concrete secret, a checkable win, and a full trace of how it was opened or not. The sandwich scenario supplies the win condition; the four-signal, judge-independent success rule supplies the verdict; the planner-generator split supplies the diagnosis; and the knowledge base, retrieval store, and privileged oracle supply the improvement over time. Across four victims and 52,096 scenarios the framework breaks 87.8% to 97.0% of defenses, verifies 60.7% to 79.1% of them, and attributes the rest to a named phase, family, and code shape. The residual is where the next round of work goes: translation, roleplay, and trigger-phrase defenses; sentence-length secrets; and the stop-point judge that still dies in the most failed attempts.

## References

Anil, C., Icard, T., Das, D., Es, S., He, H., Hashimoto, T., ... and others. 2024. Many-shot jailbreaking. arXiv preprint arXiv:2404.02151.

Chao, X., Zhang, J., Wang, Y., Li, B., Jia, D., Wang, L., and Cao, Y. 2023. Jailbreaking black box large language models in twenty queries. In Proceedings of the 2023 AI Safety Conference.

Diao, M., and others. 2025. AutoRed: A free-form adversarial prompt generation framework for automated red teaming. arXiv preprint arXiv:2510.08329.

Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., and Fritz, M. 2023. Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. In AISec @ CCS 2023.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. 2022. LoRA: Low-rank adaptation of large language models. In ICLR 2022.

Johnson, J., Douze, M., and Jégou, H. 2019. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535-547.

Lewis, P., Perez, E., Piktus, A., et al. 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. In NeurIPS 2020.

Li, X., and others. 2024. JailBench: An open ecosystem for jailbreak prompts and attacks against large language models. arXiv preprint arXiv:2402.10228.

Liu, X., Chen, X., Hu, C., Li, J., Leo, G., and others. 2024. Improving jailbreak attack success rate against large language models via self-reflection (Tree of Attacks with Pruning). In ICLR 2024.

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., Gupta, S., Majumder, B. P., Hermann, K. M., Welleck, S., Yazdanbakhsh, A., and Clark, P. 2023. Self-Refine: Iterative refinement with self-feedback. In NeurIPS 2023.

Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., Mu, N., Sakhaee, E., Li, N., Basart, S., Li, B., Forsyth, D., and Hendrycks, D. 2024. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. In ICML 2024.

NIST. 2024. Cybersecurity Evaluation Program: CyberSecEval 3. National Institute of Standards and Technology.

Reimers, N., and Gurevych, I. 2019. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In EMNLP-IJCNLP 2019.

Sanh, V., Debut, L., Chaumond, J., and Wolf, T. 2019. DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

Shen, X., Yin, Z., Chen, D., Xu, C., and Chen, H. 2023. "Do as I can, not as I say": Denying primacy of instructions to prompts. In ICLR 2024.

Shen, X., Zhang, Z., Tu, W., Zhou, Y., Liu, Y., Yang, C., and Chen, H. 2024. AutoDAN: Scalable alignment-free jailbreaking of large language models via dual alignment. In ICLR 2024.

Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., and Yao, S. 2023. Reflexion: Language agents with verbal reinforcement learning. In NeurIPS 2023.

Wallace, E., Zhang, J., Shi, W., Huang, R., Joshi, R., Durrett, G., Singh, S., and Anderson, J. 2024. The fine-tuning zero-shot jailbreak. In NeurIPS 2023.

Wallace, E., Huang, Z., Bao, Y., He, P., Potts, C., and Shi, M. 2024. The instruction hierarchy: Training LLMs to prioritize privileged instructions. In ICLR 2024.

Wang, P., Li, L., Khabsa, M., Fang, H., and Ma, H. 2023. Large language models are not fair evaluators. In ACL 2024.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., and Stoica, I. 2023. Judging LLM-as-a-judge with MT-Bench and chatbot arena. In NeurIPS 2023 (Datasets and Benchmarks).

Zou, A., Wang, Z., Kolter, J. Z., and Fredrikson, M. 2023. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043.

Dubois, Y., Galambosi, P., Liang, P., and Hashimoto, T. 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Gemma Team, Google. 2024. Gemma: Open models based on Gemini research and techniques. arXiv preprint arXiv:2403.08295.

Jiang, A. Q., Sablayrolles, A., Mensch, A., et al. 2023. Mistral 7B. arXiv preprint arXiv:2310.06825.

Cai, Z., and others. 2024. InternLM2: Strong, scalable, and efficient multilingual language models. arXiv preprint arXiv:2403.17297.

He, P., Liu, X., Gao, J., and Chen, W. 2021. DeBERTa: Decoding-enhanced BERT with discrete auto-encoding. In ACL-IJCNLP 2021.

Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. 2023. QLoRA: Efficient finetuning of quantized LLMs. In NeurIPS 2023.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. 2023. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS 2023.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. 2023. Efficient memory management for large language model serving with PagedAttention. In SOSP 2023.

Perez, F., Huang, I., Hu, H., Song, D., Zou, A., Chen, X., and others. 2022. Red teaming language models to reduce harms: Methods, challenges, and lessons learned. arXiv preprint arXiv:2211.00593.

HumanCompatibleAI. 2025. TensorTrust: A dataset of prompt defenses and access codes. https://tensortrust.ai.
