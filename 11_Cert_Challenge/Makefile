.PHONY: run eval checklist slides

run:
	uv run python app/main.py || python app/main.py

eval:
	uv run python scripts/run_eval_harness.py || python scripts/run_eval_harness.py

checklist:
	@echo "Use Cursor: Generate Certification Checklist (v4 rule) -> /deliverables/CERT_CHECKLIST.md"

slides:
	@echo "Use Cursor: Generate CERT_SLIDES.md (intro, 1/task, conclusion)"
