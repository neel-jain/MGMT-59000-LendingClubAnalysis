# Presentation Script

Companion to `docs/MGMT590_Capstone_Presentation.pptx` (which has its
own speaker notes embedded per slide — this document consolidates them
into a single readable script with timing, for rehearsal).

**Total suggested time: 8-10 minutes of slides + 3-5 minutes of live
demo (Slide 11) + open Q&A.**

| # | Slide | Timing | Script |
|---|---|---|---|
| 1 | Title | 0:30 | Introduce yourself and the one-line framing: "This project builds a complete lending risk platform for LendingClub's Indiana borrowers — not just a model, but a production-style application a lending executive could actually use." |
| 2 | Business Problem | 0:45 | Frame the problem in plain business terms. The four cards preview the whole project: score, explain, segment, act. |
| 3 | Research Questions | 0:40 | Don't read all seven aloud — note the shape of the set and flag RQ7 (segmentation) as the one supervised learning alone can't answer. |
| 4 | Dataset Overview | 0:40 | Scope precisely: Indiana only, a specific window, a carefully-defined target that excludes still-open loans. |
| 5 | Methodology / PDID | 0:45 | PDID is the thread connecting all seven phases. List phases quickly — you'll go deeper on 3, 4A, 4B, 5 next. |
| 6 | EDA Highlights | 0:45 | This chart validates RQ2 (grade predicts default) visually. Connect each bullet back to a research question. |
| 7 | Three ML Models | 0:45 | Answer "why three models" before anyone asks — each earns its place for a different reason. Logistic Regression stays in production for interpretability, not as a discarded baseline. |
| 8 | Model Comparison | 0:50 | State plainly that the numbers shown are illustrative (synthetic fixture data in this dev environment) — the methodology (why ROC-AUC, the full metric suite) is real and unchanged regardless of dataset. |
| 9 | Explainable AI | 0:40 | SHAP's rigorous foundation + consistent explainer choice across model types. Every explanation becomes plain business language before reaching the dashboard. |
| 10 | Borrower Segmentation | 0:45 | Segment names are illustrative of the NAMING LOGIC, not guaranteed labels. Emphasize the cross-check against supervised predictions as real validation. |
| 11 | Dashboard Demo | 0:20 + 3-5 min live | **[Switch to the live application here — see `DEMO_SCRIPT.md` for the click-by-click walkthrough.]** |
| 12 | Business Recommendations | 0:40 | Trace each recommendation back to a specific engine/module. |
| 13 | Limitations | 0:40 | State limitations plainly and unprompted — a credibility signal, not a weakness to hide. |
| 14 | Future Work | 0:35 | Specific, not vague — this is your answer to "what would you do with two more weeks." |
| 15 | Questions | open | Invite questions; see `FACULTY_QA.md` for prepared answers to the most likely ones. |

## Delivery Tips

- **Practice the demo separately from the slides** — Slide 11 is a
  deliberate hand-off point; know exactly which browser tab/window you
  are switching to before you start talking.
- **Have a fallback:** if live internet/localhost fails during a
  remote demo, `docs/DEMO_SCRIPT.md`'s walkthrough doubles as a
  narration you can give over static screenshots.
- **If asked a question mid-slide,** it's fine to say "great question —
  I'll cover that on the next slide" if it's coming up, or answer
  briefly and continue.
- **Total runtime target:** 12-15 minutes of prepared content, leaving
  room for a 5-10 minute Q&A in a typical 20-25 minute presentation slot.
