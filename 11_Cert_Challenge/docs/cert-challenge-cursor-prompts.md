Here’s how to refine the cursor prompt in th chat window depending on what you want Cursor to generate:

---

### ✅ **Recommended All-Purpose Prompt (Full Workflow)**

Use this when you want both checklist and slides in one pass:

```
Generate the Certification Challenge Evidence Map using the “Certification Challenge — Evidence Map & Slide Generator (v4)” rule.
Review files in @cert-challenge-grading/ for the official task list, rubric, and stack guidance.
Create /CERT_CHECKLIST.md first, assessing whether each of the 7 tasks is fully met, partially met, or missing.
Then generate /CERT_SLIDES.md with one slide per task plus intro and conclusion.
If gaps exist, provide clear next steps for the student to achieve “Yes” status.
If any parts are unclear or beyond scope, capture them under “Scope or Clarity Notes.”
```

---

### ⚙️ **For Checklist-Only Runs**

```
Generate the Certification Checklist using the “Certification Challenge — Evidence Map & Slide Generator (v4)” rule.
Review files in @cert-challenge-grading/ for the official task list, rubric, and stack guidance.
Scan the repository for evidence of all 7 tasks and fill out /CERT_CHECKLIST.md accordingly.
List missing artifacts in “One Improvement” and record any unclear or extra work under “Scope or Clarity Notes.”
```

---

### 🎥 **For Slide-Only Runs**

```
Generate CERT_SLIDES.md using the “Certification Challenge — Evidence Map & Slide Generator (v4)” rule.
Review files in @cert-challenge-grading/ for the official task list, rubric, and stack guidance.
Create 9 slides total — 1 intro, 1 per task (Tasks 1–7), and 1 conclusion.
Each task slide must include the goal, evidence, result, and next step.
Keep the content concise and presentation-ready for a 5-minute Loom video.
```

---

### 💡 Key Points

* Always include **the full rule name with (v2)** so Cursor loads the updated version.
* Mention **@cert-challenge-grading/** when it exists, so Cursor prioritizes that directory for rubric references.
* Keep verbs explicit (“Create /CERT_CHECKLIST.md”, “Then generate /CERT_SLIDES.md”) to guide Cursor’s multi-file output.
