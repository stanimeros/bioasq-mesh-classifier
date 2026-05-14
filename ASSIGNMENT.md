# NLP Course Assignment

## Task
Multi-label classification of biomedical articles into MeSH (Medical Subject Headings) terms,
using **only the title and abstract** as input.

## Requirements
- Use **word embeddings** (e.g. Word2Vec) AND **pre-trained language models** (e.g. SciBERT, BioBERT)
- Multi-label classification problem
- Choose one year's dataset from BioASQ Task A
- Submit only the **paper** (ACM-style)
- Keep models, data, and results available until grades are posted on SiS

## Data
- BioASQ Task A datasets: https://participants-area.bioasq.org/datasets/
- Registration required on the BioASQ platform
- We use: **allMeSH_2022.json**

## Our choices
- Baseline: Word2Vec + MLP
- Main model: PubMedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`)
- Server: AIDA Server #01

## Submission
- Paper only (no code submission required)
- Team of 2: Panteleimon Stanimeros, Nikolaos Strafiotis

---

## Paper Format (SETN 2026)
- **Template**: ACM SIG Proceedings — https://www.acm.org/publications/proceedings-template
- **Language**: English
- **Page limits**:
  - Full paper: 6–10 pages
  - Short paper: 2–4 pages
  - Nectar (abstract of published work): 1–2 pages
- **Review**: Double-blind peer review
- **Submission**: EasyChair — https://www.easychair.org/conferences/?conf=setn2026
- **Deadlines**:
  - Submission: May 4, 2026 (AOE, extended)
  - Author notification: June 5, 2026 (provisional)
  - Camera-ready: June 20, 2026 (provisional)
- At least one author must register and attend to present
