# BioASQ MeSH Classifier

Multi-label classification of biomedical articles into MeSH terms using BioBERT.  
Dataset: BioASQ Task A 2015b | Course: NLP, MSc AIDA, University of Macedonia.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Train
python train.py --data data/allMeSH_limitjournals.json --output_dir output/

# Predict
python predict.py --data data/test.json --output_dir output/
```

## Team

- Panteleimon Stanimeros (aid26006)
- Nikolaos Strafiotis (aid26012)
