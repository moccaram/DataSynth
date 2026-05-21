# What I learned building this

A first-person reflection on the v1 → v2 rebuild. The technical writeup is in
[reports/REPORT.md](../reports/REPORT.md); this doc is the meta-story.

---

## What I started with

I built the v1 notebook ([archive/notebooks/DataSynthis_ML_JobTask_original.ipynb](../archive/notebooks/DataSynthis_ML_JobTask_original.ipynb))
as an intern-task submission: ARIMA, Prophet, and an LSTM trained to predict
next-day log return on AAPL daily bars. The LSTM had directional accuracy
51%. The report I wrote at the time framed that as "marginally above 50%
and therefore meaningful." It is not meaningful at n=499 days — a binomial
test gives p > 0.2.

What broke my confidence in v1 wasn't a reviewer flagging the result. It was
realizing the deployed `app.py` was a Gradio "hello, name" template that
came from the official Gradio quickstart, not anything I had built — and
that my report claimed "interactive selection of model and forecast horizon"
which was literally untrue. That mistake on something as small as the
deployment section made me doubt every other claim.

So I went looking for what else I had probably gotten wrong.

## The references that changed how I thought about it

I read three things in order:

**1. Karpathy's 2015 RNN blog post.** Free, ~30 minutes. The mental model
worth taking away: the hidden state is a running summary; everything
downstream of step `t` is mediated through it. The blog also has a warning I
underweighted on first read: *"RNNs do not always show convincing signs of
generalizing in the correct way."* The 128/64-unit model I trained on ~3,000
events fits Karpathy's "memorize but not generalize" failure mode.

**2. Goodfellow Ch.10.** Specifically §10.7 (vanishing/exploding gradients),
§10.10 (LSTM gates and forget-gate-bias-of-1), and §10.11 (gradient clipping
+ regularization). The single biggest unlock: §10.11.1 eq 10.48-49 says to
clip the gradient norm. My v1 model had none of that, on a 60-step BPTT
chain. Figure 10.17 shows the "cliff" loss landscape that explains why my
training loss flatlined at epoch 2 — the optimizer was overshooting into a
flat region and never escaping.

**3. López de Prado *Advances in Financial Machine Learning*, Ch.3, 5, 7.**
This was the harder read but the bigger payoff. AFML Table 1.2 lists 10
common pitfalls in financial ML and the chapter that addresses each. My v1
notebook committed seven of them. The most important three:

- **Pitfall #5 (Fixed-time horizon labeling)** — Ch.3. Replace
  `target_return = log_return.shift(-1)` with the triple-barrier method.
- **Pitfall #4 (Integer differentiation)** — Ch.5. Replace `log_return`
  features with fractionally differenced log-price (preserves long memory).
- **Pitfall #8 (Cross-validation leakage)** — Ch.7. Replace single
  chronological split with PurgedKFold.

These aren't optimizations. They're "the loss you're computing isn't measuring
what you think it's measuring."

## What I expected the rebuild to do

I expected the LSTM to start working. The v1 LSTM was clearly broken (loss
flatlined, predicted mean). The v2 fixes — clipnorm, Huber→softmax, smaller
units, triple-barrier labels, purged CV — looked sufficient to get past the
collapse.

## What actually happened

The v2 LSTM did stop collapsing. It produced reasonable per-fold loss curves
and made non-degenerate predictions. But it didn't beat XGBoost.

Mean accuracy across 5 purged folds:

```
Majority: 35.0%   (random baseline is 33.3% for 3-class)
SES:      36.8%   (always abstains in practice)
ARIMA:    36.8%   (always abstains)
LSTM:     35.8%   ← my refined model
XGBoost:  37.8%   ← Jansen Ch.12 had predicted this
```

The directional accuracy *when each model actually picks a side* is 32-39%
across all models. Worse than 50%. So even when XGBoost commits to a
direction, it's wrong more than right.

## The two readings of that result

There are two ways to read this:

**(A) The pessimistic version.** A 2.8-point edge over majority isn't real
signal — it's noise. The whole exercise didn't produce a useful model.

**(B) The optimistic version.** *That's the result.* Three statistical
baselines collapsed to abstention; an LSTM and an XGBoost both reached ~36%
accuracy; directional accuracy stayed below 50%. This is consistent with a
broad literature finding equity return direction is hard at daily-bar
horizons. The rebuild's value wasn't producing a winning model — it was
*measuring honestly* enough to know there wasn't one to be found.

I keep flipping between (A) and (B). What pushes me toward (B) is that the
final XGBoost number (37.8%, p<0.05 in 3 of 5 folds) is statistically real,
even if economically tiny. A v3 with combinatorial purged CV, sample-
uniqueness weighting, meta-labeling, and richer features might push it
further. The pieces in place make that v3 possible.

## What I'd do differently if I started over

1. **Read AFML Ch.3 (Labeling) before writing any code.** The fixed-horizon
   label is *the* root mistake; everything else follows from it.

2. **Treat the evaluation protocol as more important than the model.** Time
   spent on PurgedKFold paid off more than time spent on LSTM architecture.

3. **Match the loss to the metric, immediately.** MSE on returns + reporting
   directional accuracy was the contradiction that produced the misleading
   v1 result. Cross-entropy on direction labels would have surfaced the
   problem faster.

4. **Start with the smallest viable model**, not the most ambitious one. The
   first LSTM should have been `LSTM(8) → Dense(3)` (Jansen 01-style) — a
   much faster way to discover that the model wasn't the bottleneck.

5. **Verify the deployment claim before writing it in the report.** The
   Gradio app embarrassment in v1 was a one-line check I never did.

## What this taught me beyond the technical content

The v1 notebook felt finished. It had a model that produced predictions, a
report that read confidently, a Gradio app section. *None of those things
were what they appeared to be.* What separated v1 from v2 wasn't capability —
it was knowing which references to read and which questions to ask of my own
work.

That generalizes beyond this project: the value of canonical references is
that they let you *audit your own assumptions*. The Goodfellow chapter
didn't teach me what an LSTM is (I knew); it taught me what conditions an
LSTM training run has to satisfy to be trustworthy. The AFML chapters
didn't teach me what supervised learning is; they taught me what supervised
learning *looks like under non-IID labels* and why the standard tools
silently fail.

This is the meta-lesson I want to remember:

> *With solid references and valid contextual knowledge, the same compute
> and the same data can produce a much more honest result. The honest
> result is often less impressive than the dishonest one and that's
> exactly why it's worth more.*

---

For the technical detail on each fix and the per-fold numbers, see
[reports/REPORT.md](../reports/REPORT.md). For the bibliography see
[docs/REFERENCES.md](REFERENCES.md).
