# Contributing

Thanks for taking an interest! This is a portfolio project and is maintained in
"approvals welcome, code mostly solo" mode — but real bug reports, reproducible
issues, and small fixes are genuinely appreciated.

## Reporting issues

- Use the issue templates. For bugs, include: what you did, what you expected,
  the actual output (tracebacks verbatim), and your OS/Python version.
- Security-relevant findings: please don't open a public issue with exploit
  detail — mail **Omechechima@gmail.com** first.

## Local development

```bash
git clone https://github.com/Culzz/StudentRisk-AI.git
cd StudentRisk-AI
python -m venv .venv && .venv\Scripts\Activate.ps1   # or source .venv/bin/activate
pip install -r requirements-dev.txt
pytest tests/
```

## Pull requests

1. Reference an existing issue (or open one first for anything non-trivial).
2. Keep PRs focused — one fix or one feature.
3. Run the test suite before pushing; CI runs `pytest` on Python 3.12.
4. Match the existing style: no enforced linter, but keep lines ≤ 100 chars and
   docstrings on anything public.
