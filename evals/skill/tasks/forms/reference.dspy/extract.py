"""Read the methods of each paper and fill in the codebook's form (DSPy).

Run it:   python extract.py
Results:  out/designs.csv (one row per paper), out/analyses.csv (one row per analysis)
Running it again costs nothing: DSPy keeps the model's replies on disk.
"""
import os
from typing import Literal

import dspy
import pandas as pd
from pydantic import BaseModel, Field

dspy.configure(lm=dspy.LM("openai/gpt-4.1-mini", temperature=0))
os.makedirs("out", exist_ok=True)


# =============================================================================
# The form (pydantic: DSPy shows the model the JSON schema, descriptions included)
# =============================================================================
class Analysis(BaseModel):
    outcome: str = Field(description="the measured quantity in lowercase, as a short noun phrase: no timing, no units, "
                                     "no group names ('Systolic blood pressure after eight weeks (mmHg)' is "
                                     "'systolic blood pressure'; 'annual salmon counts' is 'salmon count')")
    method: Literal["t_test", "anova", "rank_test", "correlation", "regression", "glm", "mixed_model", "other"] = Field(
        description="t_test: comparing two groups (any t-test); anova: three or more groups, or several factors, "
                    "without random effects; rank_test: any test on ranks: Mann-Whitney, Wilcoxon, Kruskal-Wallis, and "
                    "also Spearman and Kendall correlations; correlation: Pearson correlation only; regression: linear "
                    "regression; glm: logistic, Poisson, negative binomial or other generalized linear models; "
                    "mixed_model: any model with a random effect")
    pseudoreplicated: bool = Field(
        description="true when the test treats several measurements from the same independent unit as independent "
                    "replicates (all the fish of one aquarium, all the samples of one lake, all the pots of one bench); "
                    "false otherwise, including when each unit's values are averaged first")


class Study(BaseModel):
    design: Literal["randomized_experiment", "nonrandomized_experiment", "observational", "not_stated"] = Field(
        description="randomized_experiment: the researchers assigned the treatments and say it was at random (a coin "
                    "toss or random numbers count); nonrandomized_experiment: they assigned the treatments but do not "
                    "say it was at random; observational: they measured variation that already existed (surveys, "
                    "monitoring, sampling sites); not_stated: the text does not say")
    independent_units: int | None = Field(
        description="in an experiment, PER TREATMENT GROUP (the smallest group when they differ): the units that "
                    "received a treatment independently (the aquarium, cage, bench or plot, not the fish, mice or pots "
                    "inside them; 24 aquaria over 3 temperatures is 8); in an observational study, IN TOTAL: the "
                    "independently sampled sites, subjects or years (the lake, not its water samples); null when the "
                    "text does not give the number")
    analyses: list[Analysis] = Field(description="every statistical analysis the paper reports")


class ReadStudy(dspy.Signature):
    """Fill in the form from this methods section. Report what the authors did,
    not what they should have done. Never guess: use not_stated or null."""

    methods: str = dspy.InputField()
    study: Study = dspy.OutputField()


read_study = dspy.Predict(ReadStudy)


# =============================================================================
# The run
# =============================================================================
if __name__ == "__main__":
    papers = pd.read_csv("papers.csv")
    examples = [dspy.Example(methods=m).with_inputs("methods") for m in papers["methods"]]
    results = read_study.batch(examples, num_threads=8)
    designs, analyses = [], []
    for paper_id, result in zip(papers["paper_id"], results):
        s = result.study
        designs.append({"paper_id": paper_id, "design": s.design, "independent_units": s.independent_units})
        analyses += [{"paper_id": paper_id, **a.model_dump()} for a in s.analyses]
    pd.DataFrame(designs).astype({"independent_units": "Int64"}).to_csv("out/designs.csv", index=False)
    pd.DataFrame(analyses).to_csv("out/analyses.csv", index=False)
