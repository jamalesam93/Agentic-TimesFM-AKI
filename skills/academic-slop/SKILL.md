---
name: academic-slop
description: Remove AI-generated writing patterns from academic and scientific prose. Targets proposal padding, hedging abuse, methodology theater, citation stuffing, and overstatement of novelty in PhD proposals, journal articles, and grant applications.
metadata:
  trigger: Writing or editing academic proposals, theses, journal articles, grant applications, or scientific manuscripts
  author: Custom skill derived from stop-slop (hvpandya.com)
---

# Academic Slop

Eliminate predictable AI writing patterns from academic and scientific prose.

## Core Rules

1. **Name the specific thing.** No vague significance statements ("The implications are significant," "This has far-reaching consequences"). State the actual implication or consequence. See [references/phrases.md](references/phrases.md).

2. **Hedge with evidence, not words.** "May potentially" and "could possibly" add nothing. Either commit to the claim and cite evidence, or present it as a hypothesis with explicit uncertainty bounds.

3. **Strip methodology theater.** Standard procedures (k-fold cross-validation, train-test splits, SHAP analysis) do not need elaborate justification. Describe what you do, not why a well-known technique exists.

4. **Every citation earns its place.** Strings of [1,2,3,4,5] where one precise reference suffices are padding. One strong, specific citation beats five tangential ones.

5. **State what the research does, not what it promises.** "This study develops..." beats "This study will develop..." Objectives use present tense. Expected outcomes use "aims to" or "is expected to."

6. **Name the actor in Methods.** "Data were collected from MIMIC-IV" hides the action. "We extracted data from MIMIC-IV using PhysioNet's PostgreSQL interface" tells the reader what happened.

7. **Open with findings, not throat-clearing.** "It is well established that AKI causes morbidity" delays the point. "AKI affects 15-20% of hospitalized patients and increases mortality by 3-5x" gives the reader the fact immediately.

8. **Use technical terms, not metaphors.** "Bridges the gap," "paves the way," "landscape," "paradigm shift," "shed light on" are vague. Replace with the specific technical relationship.

9. **Cut filler transitions.** "Furthermore," "Moreover," "Additionally" between unrelated points are noise. Restructure so the logical connection is in the content, not the glue word.

10. **Verify novelty claims.** "Novel," "innovative," "first-of-its-kind" require a literature-backed statement that no prior work does this. If you cannot cite the gap explicitly, remove the claim.

## Quick Checks

Before delivering academic prose:

- Any "significant implications" without naming the implication? Name it.
- Any "may potentially" / "could possibly"? Commit or cut.
- Standard procedure described as if novel? Strip to essentials.
- Citation string [1,2,3,4,5]? Reduce to the one that supports the claim.
- "Will contribute/establish/demonstrate" in a proposal? Reframe in present tense.
- "Data were collected" without naming who collects? Add the actor.
- "It is well established that..." throat-clearing? State the finding directly.
- "Paradigm shift" / "landscape" / "bridges the gap"? Use technical language.
- "Furthermore/Moreover/Additionally" between unrelated points? Cut or restructure.
- "Novel/innovative" without evidence? Verify or remove.

## Scoring

Rate 1-10 on each dimension:

| Dimension | Question |
|-----------|----------|
| Precision | Are claims specific, or vague? |
| Economy | Does every sentence earn its space? |
| Authority | Does the evidence speak, or does hedging speak? |
| Originality | Are novelty claims verifiable? |
| Density | Can any sentence be cut without loss? |

Below 35/50: revise.

## Academic Mode Adjustments

When editing academic writing, these standard stop-slop rules are relaxed:

- **Passive voice in Methods**: Allowed where the agent is irrelevant ("The solution was heated to 90C" — the researcher is not the point).
- **Wh- sentence starters**: Allowed as section motivators ("Why does this matter?" as a rhetorical bridge).
- **Em-dashes**: Allowed for parenthetical technical clarifications (e.g., "Attention-based LSTMs—a variant that weights temporal inputs—outperform standard LSTMs").
- **Meta-joiners**: Allowed sparingly for proposal navigation ("Section 4 describes the methodology") when the document structure is complex.
- **"This study" / "This research"**: Allowed as subjects when they are the actual actor performing the action.

## References

- [references/phrases.md](references/phrases.md) — Banned phrases and replacements
- [references/structures.md](references/structures.md) — Structural patterns to avoid
- [references/examples.md](references/examples.md) — Before/after pairs from academic writing

## License

MIT (derived from stop-slop by Hardik Pandya)
