# Part 1 Results Analysis

This document summarizes the main result story from the five Part 1 experiments on the same 100-task random ARC training sample.

## Main Result Story

The central finding is that the quality of the natural-language hypothesis strongly controls whether the downstream program-search pipeline succeeds. The bottleneck is not only program generation; it is also the quality of the abstract transformation description that guides program generation.

| Experiment | Hypothesis Model | Program Model | Test Example Accuracy | Task Accuracy |
| --- | --- | --- | --- | --- |
| Direct Output Grid | None | None | 14/105 = 13.33% | 11/100 = 11% |
| Program Only | None | GPT-5.4-mini | 34/105 = 32.38% | 30/100 = 30% |
| Hypothesis + Program | GPT-5.4-mini | GPT-5.4-mini | 27/105 = 25.71% | 25/100 = 25% |
| Hint + Hypothesis + Program | GPT-5.4-mini + human hint | GPT-5.4-mini | 43/105 = 40.95% | 38/100 = 38% |
| High Quality Hypothesis + Program | GPT-5.5 | GPT-5.4-mini | 87/105 = 82.86% | 82/100 = 82% |

The overall progression gives a clean story:

1. Directly generating the output grid is weak.
2. Generating executable programs is substantially better.
3. Adding a hypothesis step is not automatically helpful when the hypothesis model is weak.
4. Human hints partially recover the value of hypothesis-guided generation.
5. High-quality hypotheses produce the strongest performance by a large margin.

In short: direct prediction < executable programs < guided reasoning < high-quality abstract reasoning.

## Key Comparisons

### Program Generation Beats Direct Output Prediction

The direct output baseline achieves only 11% task accuracy, while the program-only baseline reaches 30%. This suggests that executable code is a better intermediate representation for ARC-style transformations than raw grid prediction.

This is an important baseline result: even without explicit natural-language hypotheses, asking the model to produce transformation programs gives it a more structured way to express the rule.

### Hypothesis Decomposition Is Not Automatically Helpful

The GPT-5.4-mini hypothesis + program experiment performs worse than the program-only baseline:

- Program Only: 30/100 tasks solved.
- GPT-5.4-mini Hypothesis + Program: 25/100 tasks solved.

Both settings effectively generate six candidate programs per task. The difference is that the hypothesis pipeline first constrains program generation through a natural-language hypothesis. When the hypothesis is incomplete or wrong, it can push the program generator into the wrong solution space.

This is an important negative result. It shows that adding an intermediate reasoning step does not guarantee improvement; the intermediate representation must be high quality.

### Human Hints Improve Weak Hypothesis Generation

Adding human-written hints improves the GPT-5.4-mini hypothesis pipeline from 25% to 38% task accuracy:

- GPT-5.4-mini Hypothesis + Program: 25/100 tasks solved.
- Hint + GPT-5.4-mini Hypothesis + Program: 38/100 tasks solved.

Pairwise, the hint condition gains 20 tasks that the no-hint condition failed, while losing 7 tasks that the no-hint condition solved. This means hints do not uniformly improve every task, but they substantially shift the distribution in a positive direction.

This supports the idea that targeted semantic guidance helps the model form better abstractions before program generation.

### High-Quality Hypotheses Are the Dominant Factor

The high-quality hypothesis experiment uses GPT-5.5 for hypothesis generation while keeping GPT-5.4-mini as the downstream program generator. It achieves:

- 87/105 = 82.86% test example accuracy.
- 82/100 = 82% task accuracy.

Compared with GPT-5.4-mini hypothesis generation, this is a +57 point task-accuracy improvement. Compared with the human-hint experiment, it is still a +44 point improvement.

This is the strongest evidence that abstract hypothesis quality is the main bottleneck. The downstream program generator is unchanged, but better hypotheses make the overall system much more successful.

## Difficulty Breakdown

Task accuracy by difficulty:

| Experiment | Easy | Medium | Hard | Expert |
| --- | --- | --- | --- | --- |
| Direct Output Grid | 10/62 = 16.1% | 1/25 = 4.0% | 0/9 = 0.0% | 0/4 = 0.0% |
| Program Only | 24/62 = 38.7% | 5/25 = 20.0% | 1/9 = 11.1% | 0/4 = 0.0% |
| Hypothesis + Program | 18/62 = 29.0% | 6/25 = 24.0% | 1/9 = 11.1% | 0/4 = 0.0% |
| Hint + Hypothesis + Program | 25/62 = 40.3% | 8/25 = 32.0% | 4/9 = 44.4% | 1/4 = 25.0% |
| High Quality Hypothesis + Program | 52/62 = 83.9% | 20/25 = 80.0% | 7/9 = 77.8% | 3/4 = 75.0% |

The high-quality hypothesis condition improves performance across all difficulty levels. It is not merely solving more easy tasks; it also performs strongly on harder categories, reaching 7/9 on hard tasks and 3/4 on expert tasks.

However, the hard and expert subsets are small, especially expert with only four tasks, so these category-level numbers should be interpreted as suggestive rather than definitive.

## Pairwise Task-Level Changes

Pairwise task-level comparisons show how each intervention changes the set of solved tasks:

| Comparison | Gained Tasks | Lost Tasks | Retained Tasks | Net Change |
| --- | ---: | ---: | ---: | ---: |
| Direct Output Grid -> Program Only | 21 | 2 | 9 | +19 |
| Program Only -> Hypothesis + Program | 9 | 14 | 16 | -5 |
| Hypothesis + Program -> Hint + Hypothesis + Program | 20 | 7 | 18 | +13 |
| Hint + Hypothesis + Program -> High Quality Hypothesis + Program | 45 | 1 | 37 | +44 |
| Hypothesis + Program -> High Quality Hypothesis + Program | 58 | 1 | 24 | +57 |
| Program Only -> High Quality Hypothesis + Program | 53 | 1 | 29 | +52 |

This analysis is useful because it shows that the improvements are not just small changes in example-level scoring. The high-quality hypothesis condition solves many tasks that other settings fail, while losing very few previously solved tasks.

## Suggested Claims for the Results Section

1. Executable program generation is substantially stronger than direct output-grid generation, indicating that code is a better intermediate representation for ARC transformations.

2. Hypothesis decomposition only helps when the hypothesis is high quality. Weak hypotheses can reduce performance by constraining program generation toward incorrect rules.

3. Human hints improve weak-model hypothesis generation, showing that even simple semantic guidance can help the model form better abstractions.

4. The largest gain comes from improving hypothesis quality, not from changing the downstream program generator. GPT-5.5 hypotheses paired with the same GPT-5.4-mini program generator achieve the best result.

5. Remaining failures in the high-quality hypothesis condition likely come from either imperfect hypotheses, program synthesis failures, or overfitting to the training examples. These failures are a natural target for qualitative error analysis.

## Recommended Framing

The result section can be organized around the following storyline:

> We compare direct output prediction, direct program synthesis, hypothesis-guided program synthesis, human-hint-guided hypothesis generation, and high-quality hypothesis generation. The results show that program synthesis is more reliable than direct grid prediction, but that hypothesis-guided search only succeeds when the hypotheses are sufficiently accurate. Human hints improve weak-model hypotheses, while GPT-5.5 hypotheses dramatically improve downstream program synthesis despite using the same program generator. This suggests that abstract rule induction is the key bottleneck in our current pipeline.

