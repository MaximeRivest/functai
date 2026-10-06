# Codebook: one form per paper

For each paper, record the study design and every statistical analysis it reports.

## The paper

**design**: one of
- `randomized_experiment`: the researchers assigned the treatments and say it was at random (a coin toss or random numbers count as random).
- `nonrandomized_experiment`: the researchers assigned the treatments but do not say it was at random.
- `observational`: the researchers measured variation that already existed (surveys, monitoring, sampling sites).
- `not_stated`: the text does not say.

**independent_units**: how many independent units the study has, counted our lab's way:
- in an experiment, **per treatment group** (the smallest group when they differ): the units that received a treatment independently, meaning the aquarium, the cage, the bench or the plot, not the fish, mice or pots inside them. 24 aquaria over 3 temperatures is 8.
- in an observational study, **in total**: the independently sampled sites, subjects or years (the lake, not the water samples taken in it).
- empty when the text does not give the number.

## Each analysis

**outcome**: the measured quantity in lowercase, as a short noun phrase: no timing, no units, no group names. "Systolic blood pressure after eight weeks (mmHg)" is `systolic blood pressure`; "annual salmon counts" is `salmon count`.

**method**: one of
- `t_test`: comparing two groups (any t-test).
- `anova`: three or more groups, or several factors, without random effects.
- `rank_test`: any test on ranks: Mann-Whitney, Wilcoxon, Kruskal-Wallis, and also Spearman and Kendall correlations.
- `correlation`: Pearson correlation only.
- `regression`: linear regression.
- `glm`: logistic, Poisson, negative binomial or other generalized linear models.
- `mixed_model`: any model with a random effect.
- `other`.

**pseudoreplicated**: true when the test treats several measurements from the same independent unit as independent replicates (all the fish of one aquarium, all the samples of one lake, all the pots of one bench). False otherwise, including when each unit's values are averaged first.
