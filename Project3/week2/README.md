# Week 2: Does the model react the way it should?

In Week 1 you turned each Findings text from the ReXGradient dataset into a
representation (a vector from frozen BioClinicalBERT) and trained linear probes
to read information out of it, for example whether the report describes a
pleural effusion.

This week you will test whether these representations are sensitive to
controlled edits of the text.  The idea is to test different kinds of edits,
i.e., some edits are expected to change the probe's prediction, because they
change the input in a way that the model should be sensitive to, while other
edits should not change the prediction, because they change the text in a way
the model should be invariant to.
Note: the model we are testing here is BioClinicalBERT - we are asking whether
the model generates representations that are semantically invariant or
sensitive in the appropriate way.

An **intervention** is a small, controlled edit of a report, for example replacing
"small right pleural effusion" with "no pleural effusion", or exchanging "left" and
"right". For the question "Is there a pleural effusion, according to the
text?", some edits create a text that will change the answer
and some do not.
A good model will be sensitive to the first kind of edit (predicted to be
counterfactual), and change the representation sufficiently to change the
linear probe's prediction, but for the second kind of edit it should stay
stable (predicted to _not_ be counterfactual).

You will measure both the
probe's prediction and how far the representation itself moves, and you will
see that these two are not the same thing.

Available in addition to the Week 1 code: `embed_texts` in `src/embeddings.py` embeds
any list of texts with the same frozen BioClinicalBERT; `sentences` and
`find_sentence` in `src/edits.py` split a report into sentences and locate the
sentence that mentions a word.

## Pleural effusion, briefly

<img src="pleural_effusion.jpg" width="420" alt="Chest X-ray with a large left pleural effusion">

*Chest X-ray with a massive left pleural effusion: the lower left lung (on the right side of the
picture) is filled with fluid and appears white, with a curved upper edge. Image: Wikimedia Commons user
Clinical_Cases, CC BY-SA 2.5.*

A pleural effusion is fluid in the pleural space, the thin gap between the lung and
the chest wall. It is not a disease by itself but a sign of one: most often heart
failure, pneumonia, cancer, or kidney and liver disease.

Radiologists describe it in a predictable way: which side (left, right, bilateral), how much (trace, small, moderate, large), and how it changed since the last X-ray
(new, increased, decreased, resolved). They also mention what sits next to it: the
compressed lung at the base (atelectasis), a possible infection (consolidation), or
signs of heart failure (an enlarged heart, pulmonary edema), because these point to
the cause.

Keep this background knowledge in mind when you design edits: side and size change the
description but not the presence of an effusion, "resolved" or "no" changes the
presence, and neighbouring findings such as atelectasis are correlated with effusion,
so a model may lean on them even when the effusion sentence itself is gone.

## 1. Classifier

Week 1 used cross-validation, so there is no single trained classifier yet. Take the
clear effusion cases (`binary_label(studies, "pleural_effusion")`), keep the studies with a
usable Findings text as in Week 1 (`usable_findings`), split them once by patient into 70% training and 30% held-out
(`GroupShuffleSplit` with `patient_id` as groups), train the probe on the training
part (standardise the data, use logistic regression for the probe) and freeze
it. All interventions are evaluated with this classifier on the held-out
reports. If you also use TF-IDF, fit the vectoriser once on the original
training reports and only transform the edited reports.

## 2. Interventions

Design at least three edits, with at least two different hypothesized effects.
Negation must be one of them.

For every edit, before running it, write down your hypothesis: should the edit
change the text representation such that the probe's probability of effusion
drops, increases, or stays the same, and why. Choose the edits so that your
hypotheses are not all the same.

Note also that not all edits will be applicable to all texts: for example, editing "left" to "right" and vice-versa only applies to texts with "left" or "right" in the text.

Three examples:

| Edit | What it does | Applied to |
|------|--------------|------------|
| negate | replace the effusion sentence by "No pleural effusion." | held-out reports with an effusion |
| swap left / right | exchange every "left" and "right" | held-out reports with "left"/"right" and with an effusion |
| insert | append "There is a small left pleural effusion." | held-out reports without an effusion |

Other ideas: change the size word, use a synonym, add a typo, shuffle sentences, etc. Try to be creative!

Read a few edited reports to make sure the edit changed the text as you intended.

## 3. Measurements

For each edit, on all held-out reports where it applies: embed the edited text and
score it with the linear probe classifier that was trained in section (1). Report per edit:

- the mean probability of effusion before and after the edit,
- the flip rate: the fraction of reports where the decision (above or below 0.5) changed,
- the representation shift: the mean cosine distance (1 minus the cosine similarity)
  between the original and the edited embedding.

Sanity-check: Negation should clearly lower the probability for most reports;
if it does not, there is a mistake in the probe or in the edit.

Show your results in a table: edit type,
hypothesis, number of reports, mean probability before and after, flip rate, mean
cosine shift.

## 4. Visualization

We want to see two things; how you visualize them is up to you, as long as the result is easy to see.

1. **For each edit, how the prediction changed, and whether that matched your
   hypothesis.** (Hint: perhaps a dot plot or strip plot with two points for each edit, coded by hypothesis.)

2. **For a few reports, how far each edit moved the representation, and whether moving
   far goes together with flipping the decision.** (Hint: maybe a UMAP --- but here make sure that the UMAP is fit to the original embeddings, not the edits!)

## 5. Interpretation

- Were your hypotheses confirmed: did edits change the classifier predictions (or not) as expected?
- If your hypotheses were not confirmed, why do you think that's the case?
- Were there cases where flip rate and representation shift (cosine distance)
  did not match, e.g. where the representation moved a lot while the decision
  stayed the same, or the representation didn't move much while the decision
  flipped, and what does that say about cosine distance as a measure of change?

Your results for this week are the table, the two figures and the interpretation.

## Extra

Last week you compared the representations from BioClinicalBERT with representations generated with TF-IDF.
Try TF-IDF representations for the edits as well. Do these representations also lead to the same changes in predictions as when using BioClinicalBERT representations? What differences do you see, and what do you think causes them?

