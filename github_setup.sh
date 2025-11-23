#!/usr/bin/env bash
# GitHub Issues Bootstrap Script for Clickbait Detection Project
# Usage:
#   1. Edit REPO and the assignee usernames below.
#   2. Make executable: chmod +x github_setup.sh
#   3. Run: ./github_setup.sh
#
# Requirements:
#   - GitHub CLI installed: https://cli.github.com/
#   - Logged in: gh auth login

# >>> EDIT THIS <<<
REPO="el-gato-ai/Clickbait_Machine_Learning_Project"

# Optional: edit GitHub usernames for assignees
ASSIGNEE_KOSTAS="konstantinosmpouros"
ASSIGNEE_THODORIS="thodoris-github"
ASSIGNEE_NIKOS="el-gato-ai"

echo "Creating issues in repository: $REPO"
read -p "Continue? (y/N) " yn
case $yn in
    [Yy]* ) ;;
    * ) echo "Aborted."; exit 1;;
esac

# 1. Scraping & Data Fetching
gh issue create -R "$REPO"   -t "Scraping: Fetch Greek news data"   -b "Owner: Κώστας
Description:
Υλοποίηση scraping scripts από 4–6 ελληνικά sites (π.χ. news247, in.gr, protothema, skai).
Εξαγωγή τίτλων, URL, ημερομηνίας, source. Καθαρισμός HTML/duplicates.

Deliverables:
- /data/raw/news_scraped.json
- /src/data_fetch/fetch_news.py

Deadline (Κυριακή): 2025-10-27"   -l "task,phase:prepare"   -a "$ASSIGNEE_KOSTAS"

# 2. GreekSUM Parsing
gh issue create -R "$REPO"   -t "GreekSUM: Extract and unify titles"   -b "Owner: Κώστας
Description:
Φόρτωση GreekSUM, εξαγωγή τίτλων, ενοποίηση format με scraped dataset.

Deliverables:
- /data/processed/greeksum_titles.csv
- /src/data_fetch/load_greeksum.py

Deadline (Κυριακή): 2025-11-03"   -l "task,phase:prepare"   -a "$ASSIGNEE_KOSTAS"

# 3. Annotation Guidelines
gh issue create -R "$REPO"   -t "Annotation: Define clickbait guidelines"   -b "Owners: Νίκος + Θοδωρής
Description:
Ορισμός κανόνων annotation (τι είναι clickbait / όχι), παραδείγματα, edge cases.

Deliverables:
- docs/annotation_guidelines.pdf

Deadline (Κυριακή): 2025-11-10"   -l "task,phase:annotation"   -a "$ASSIGNEE_NIKOS" -a "$ASSIGNEE_THODORIS"

# 4. Manual Annotation (500 titles)
gh issue create -R "$REPO"   -t "Annotation: Manual labeling of 500 titles"   -b "Owner: Θοδωρής
Description:
Χειροκίνητη επισημείωση περίπου 500 τίτλων, καταγραφή edge cases, βασική μέτρηση συμφωνίας.

Deliverables:
- data/labels/manual_labels.csv

Deadline (Κυριακή): 2025-11-24"   -l "task,phase:annotation"   -a "$ASSIGNEE_THODORIS"

# 5. LLM-assisted Annotation
gh issue create -R "$REPO"   -t "Annotation: LLM-assisted labeling (3000–5000 titles)"   -b "Owner: Θοδωρής
Description:
Χρήση LLM (π.χ. LLaMA/Gemma) για assisted labeling 3000–5000 τίτλων.
Human validation σε ~10% δείγμα.

Deliverables:
- data/labels/llm_assisted_labels.csv
- prompts & notes

Deadline (Κυριακή): 2025-12-15"   -l "task,phase:annotation"   -a "$ASSIGNEE_THODORIS"

# 6. Embedding Pipeline
gh issue create -R "$REPO"   -t "Features: Implement embedding pipeline for titles"   -b "Owner: Κώστας
Description:
Υλοποίηση script για μετατροπή τίτλων σε embeddings (π.χ. Gemma/LLaMA).
Batch processing, αποθήκευση .npy, sanity checks.

Deliverables:
- /src/features/embed_titles.py
- /data/embeddings/*.npy

Deadline (Κυριακή): 2026-01-05"   -l "task,phase:features"   -a "$ASSIGNEE_KOSTAS"

# 7. Dataset Preparation & Splits
gh issue create -R "$REPO"   -t "Data: Prepare final train/test splits"   -b "Owner: Νίκος
Description:
Συνένωση manual + LLM-assisted labels.
Stratified splits (train GreekSUM, test multi-source), αντιμετώπιση imbalance.

Deliverables:
- /data/final/train.csv
- /data/final/test.csv
- /src/data_preprocessing/prepare_dataset.py

Deadline (Κυριακή): 2026-01-12"   -l "task,phase:model"   -a "$ASSIGNEE_NIKOS"

# 8. Baseline Model Training
gh issue create -R "$REPO"   -t "Models: Train TF-IDF baselines"   -b "Owner: Νίκος
Description:
Εκπαίδευση baseline μοντέλων (TF-IDF + Logistic Regression / SVM).
Αρχική αξιολόγηση σε validation data.

Deliverables:
- /src/models/train_baselines.py
- αποτελέσματα σε .csv

Deadline (Κυριακή): 2026-01-19"   -l "task,phase:model,baseline"   -a "$ASSIGNEE_NIKOS"

# 9. Embedding-based Model Training
gh issue create -R "$REPO"   -t "Models: Train embedding-based classifiers"   -b "Owner: Νίκος
Description:
Εκπαίδευση μοντέλων πάνω σε sentence embeddings.
Σύγκριση με baselines, επιλογή τελικού μοντέλου.

Deliverables:
- /src/models/train_embedding_models.py
- evaluation reports

Deadline (Κυριακή): 2026-01-19"   -l "task,phase:model"   -a "$ASSIGNEE_NIKOS"

# 10. Generalization & Error Analysis
gh issue create -R "$REPO"   -t "Evaluation: Cross-site generalization & error analysis"   -b "Owner: Νίκος (με υποστήριξη Θοδωρή)
Description:
Αξιολόγηση performance ανά site, ανάλυση distribution shift.
Error analysis σε false positives/negatives, κατηγοριοποίηση λαθών.

Deliverables:
- notebooks/model_evaluation.ipynb

Deadline (Κυριακή): 2026-01-26"   -l "task,phase:evaluation"   -a "$ASSIGNEE_NIKOS" -a "$ASSIGNEE_THODORIS"

# 11. End-to-End Pipeline Notebook
gh issue create -R "$REPO"   -t "Docs: Build end-to-end pipeline notebook"   -b "Owner: Νίκος
Description:
Ενιαίο notebook που δείχνει: data loading → exploration → features → models → evaluation.

Deliverables:
- notebooks/full_pipeline.ipynb

Deadline (Κυριακή): 2026-01-26"   -l "task,phase:docs"   -a "$ASSIGNEE_NIKOS"

# 12. Repo Organization & Cleanup
gh issue create -R "$REPO"   -t "Infra: Organize and clean repository"   -b "Owners: Όλη η ομάδα
Description:
Τελική δομή φακέλων, αφαίρεση περιττών αρχείων, ενημέρωση README.

Deliverables:
- τακτοποιημένο repo
- ενημερωμένο README.md

Deadline (Κυριακή): 2026-01-26"   -l "task,phase:infra" 

# 13. Final Presentation
gh issue create -R "$REPO"   -t "Deliverable: Prepare final presentation"   -b "Owners: Όλη η ομάδα
Description:
Προετοιμασία 8λεπτης παρουσίασης: πρόβλημα, προσέγγιση, πειράματα, αποτελέσματα, συμπεράσματα.

Deliverables:
- presentation/final_presentation.pdf

Deadline (Κυριακή): 2026-01-26"   -l "task,phase:deliverable,presentation" 

# 14. Final Report
gh issue create -R "$REPO"   -t "Deliverable: Write final report"   -b "Owner: Νίκος
Description:
Συγγραφή τελικής αναφοράς σύμφωνα με τις οδηγίες του μαθήματος.
Περιλαμβάνει όλα τα στάδια του ML pipeline και τα αποτελέσματα.

Deliverables:
- docs/final_report.pdf

Deadline (εξωτερικό): 2026-01-31"   -l "task,phase:deliverable,report"   -a "$ASSIGNEE_NIKOS"

echo "Done. Issues created in $REPO."
