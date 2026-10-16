# Fake Job Posting Detector

A supervised machine learning project that classifies online job postings as **real** or **fake**, using text and metadata. It combines Logistic Regression, Random Forest and XGBoost with soft voting, and is evaluated with metrics suited to imbalanced data.

## Problem

Fake job postings are a form of online fraud. Scammers use them to collect fees, bank details and identity documents from job seekers. Only about 5% of postings in this dataset are fraudulent, so a model that always says "real" would score about 95% accuracy and catch no scams. The goal is a model that **catches as many fake postings as possible** while keeping false alarms reasonable.

## Dataset

- **Source:** [Real / Fake Job Posting Prediction (Kaggle)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction), originally from the University of the Aegean (EMSCAD)
- **Size:** 17,880 postings, 18 columns, 866 fraudulent
- **Target:** `fraudulent` (0 = real, 1 = fake)
- **Features used:** title, company profile, description, requirements and benefits (combined into one text field), plus employment type, experience, education, industry, function, country, remote flag, company-logo flag, screening-questions flag, and whether a salary range is given

The CSV is not included in this repository. Download it from the link above and place `fake_job_postings.csv` next to the notebook.

## Approach

1. **Cleaning:** filled missing values, removed duplicate postings (so identical ads cannot appear in both train and test), and engineered `country` and `has_salary` features.
2. **Split:** 80/20 train/test split, stratified so both sets keep the same fake/real ratio.
3. **Preprocessing pipeline:** TF-IDF (unigrams and bigrams, 20,000 features) for text, one-hot encoding for categorical columns, and pass-through for binary flags. Everything is inside a scikit-learn `Pipeline`, so encoders are fitted on training data only, which prevents data leakage.
4. **Class imbalance:** `class_weight="balanced"` for Logistic Regression and Random Forest, and `scale_pos_weight` for XGBoost.
5. **Models:** Logistic Regression, Random Forest (400 trees, max depth 7) and XGBoost (100 estimators).
6. **Soft voting:** the three models' fake-probabilities are averaged, and a posting is flagged as fake if the average is at least 0.5.
7. **Evaluation:** precision, recall, F1 and PR-AUC on the fake class, plus a check on hand-written sample postings.

## Results

Evaluated on a held-out test set of 3,505 postings (171 fake, 3,334 real):

| Model | Fake Precision | Fake Recall | Fake F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.602 | 0.918 | 0.727 | 0.908 |
| Random Forest | 0.460 | 0.836 | 0.593 | 0.808 |
| XGBoost | 0.616 | 0.883 | 0.726 | 0.885 |
| **Voting (ensemble)** | **0.660** | **0.918** | **0.768** | **0.916** |

The voting ensemble gave the best result. It detects about 92% of fake postings, and about two thirds of the postings it flags are truly fake. It correctly passes 98% of real postings.

**Sanity check:** the ensemble classified all 8 hand-written sample postings (4 scam-style, 4 genuine-style) correctly. This is a small illustrative check, not a measure of accuracy.

## How to run

```bash
pip install pandas scikit-learn xgboost jupyter
jupyter notebook
```

Open the notebook, make sure `fake_job_postings.csv` is in the same folder, and run all cells.

## Limitations

- The data was collected in 2012 to 2014 and is mostly English-language, US-focused postings. Newer scam styles and postings from other regions may be missed.
- The fake class is small (866 postings), so scores can vary between splits.
- Precision is about 0.66, so some genuine postings will be flagged for review. The model is a screening aid, not a final judge.
- The sample postings used in the final check were written by hand and are too few to measure real-world performance.

## Future work

- Tune the decision threshold on a validation split to balance precision and recall.
- Add SHAP or LIME explanations to show which words and features drive each prediction.
- Cross-validation for more stable estimates.
- Wrap the saved model in a simple web app (for example Streamlit) so users can paste a posting and get a result.

## Tech stack

Python, pandas, scikit-learn, XGBoost, Jupyter Notebook

## Author

**Rounak** 

Dataset credit: University of the Aegean, Laboratory of Information and Communication Systems Security (EMSCAD), shared on Kaggle by shivamb.
