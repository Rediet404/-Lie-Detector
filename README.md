Project: Lie Detection using the LIAR Dataset

Date: December 17, 2025

Group members:

Rediet Mesfin ATE/5020/14 Esrom Basazinew ATE/5227/14 Yohannes Siyum ATE/5195/14 Ayansa Adugna ATE/6100/14

    Introduction

The objective of this project was to develop a binary classification system capable of distinguishing between truthful and deceptive political statements. Using the LIAR dataset—a benchmark collection of 12.8K human-labeled short statements from PolitiFact—this study transforms a multi-class truthfulness scale into a binary "Lie vs. Truth" detector using classical machine learning algorithms and Natural Language Processing (NLP).

    Methodology

2.1 Data Processing & Label Mapping

The raw LIAR dataset provides six fine-grained labels. To create a binary detector, labels were mapped as follows:

    Target 1 (Lie): Pants-fire, False, Barely-true
    Target 0 (Truth): Half-true, Mostly-true, True

Data cleaning included the removal of missing values and duplicate statement-label pairs to ensure the model learned distinct linguistic patterns rather than memorizing repeated entries.

2.2 Feature Extraction

Textual data was converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency). The vectorizer was configured with:

    N-gram Range (1, 2): To capture both individual words and significant word pairs (bigrams).
    Stop Words: English "noise" words were removed to focus on substantive vocabulary.
    Max Features: Restricted to 5,000 to maintain computational efficiency.

2.3 Model Training & Tuning

Seven classical ML models were evaluated using Scikit-Learn Pipelines to prevent data leakage. Each pipeline was optimized via GridSearchCV with 3-fold stratified cross-validation, targeting the F1-score as the primary performance metric.

    Results and Model Selection

3.1 Validation Performance

The Random Forest classifier emerged as the best performer during the validation phase, striking the most effective balance between precision and recall.
Model 	Accuracy 	Precision 	Recall 	F1-Score 	ROC-AUC
Random Forest 	0.625 	0.608 	0.619 	0.613 	0.670
Logistic Regression 	0.603 	0.582 	0.614 	0.597 	0.656
Naive Bayes 	0.601 	0.584 	0.589 	0.586 	0.645
Linear SVM 	0.577 	0.556 	0.588 	0.571 	0.621
Gradient Boosting 	0.595 	0.686 	0.287 	0.405 	0.652

Note: Gradient Boosting showed high precision but failed significantly in recall, making it too "cautious" for a general-purpose detector.

3.2 Final Test Performance

The tuned Random Forest model was evaluated on the held-out test set (N=1,266) to assess its real-world generalizability.

    Final Accuracy: 60.8%
    Test ROC-AUC: 0.64
    Test F1 (binary): 0.61

The model demonstrated robust performance, correctly identifying truthful statements with a precision of 0.66 and lies with a recall of 0.58. This confirms the model is a reliable baseline for automated fact-checking, significantly outperforming random chance.

    Conclusion & Future Work

This project demonstrates that classical ML architectures, specifically Random Forest with TF-IDF, can capture the underlying signals of deception in short-form political text.

Future Work: To improve the system's predictive power beyond the 61% mark, future iterations should:

    Incorporate Metadata: The LIAR dataset includes features like "speaker affiliation", "job title", and "context" (e.g., tweet vs. debate). Integrating these could provide crucial social context that text alone misses.
    Deep Learning: Exploring Transformer-based models (like BERT) may better capture the semantic nuances and sarcasm often found in political rhetoric.

Submission Artifacts:

best_model.pkl: Serialized Random Forest pipeline. model_comparison.csv: Full metric log for all 7 tested algorithms.
