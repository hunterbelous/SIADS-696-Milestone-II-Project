##Predicting Death or Clinical Deterioration from the Emergency Department with Supervised and Unsupervised Learning 
Authors: Hunter Belous, Allen Chezick, Zach Sletten
SIADS 696 — Milestone II

#Summary

Machine learning project predicting short-term death or clinical deterioration in emergency department patients using the MIMIC-IV / MDS-ED dataset. Combines supervised (Random Forest, Logistic Regression, SVM) and unsupervised (PCA, KMeans) approaches to identify high-risk patient patterns.

#File Types

00_mds_ed.csv: Raw CSV of the MIMIC IV dataset.

.ipynb : Jupyter notebooks containing preprocessing, EDA, supervised, and unsupervised modeling scripts.

.pkl : Serialized dataset files (e.g., df_best.pkl) containing cleaned and engineered features.

.pdf : Final written report detailing methods, results, and conclusions.

.txt / .md : Documentation and dependencies (README.md, requirements.txt).

/outputs/ — Stores model artifacts and PCA results from unsupervised and supervised phases.

#Overview

This project develops an early warning system using the MIMIC-IV / MDS-ED dataset to predict short-term death or clinical deterioration for emergency department patients. Both supervised and unsupervised learning methods were used to analyze over 120,000 ED visits, focusing on vitals, labs, and demographics collected within the first 1.5 hours of presentation.

#Notebook Breakdown

01_preprocessing.ipynb
Loads and cleans the MDS-ED dataset. Removes identifiers, replaces sentinel values, drops sparse features, rounds continuous variables, and creates derived outcomes such as mortality_28d and death_or_deterioration_any. Saves the engineered dataset as df_best.pkl.

02_eda_cleaned.ipynb
Performs exploratory data analysis on the cleaned dataset. Displays missing value summaries, distributions, correlations, and outcome frequencies to confirm dataset readiness for modeling. 

03_unsupervised.ipynb
Runs PCA with n_components = 0.95, retaining 96% of the variance. Applies KMeans clustering, achieving a best silhouette score of about 0.93 with k = 3. Produces 2D and 3D cluster visualizations and outcome overlays to show subgroup enrichment.

04_supervised_phase1.ipynb
Uses PyCaret for initial supervised modeling and comparison across multiple algorithms. Provides insight into early performance patterns and helps select the top three models for deeper tuning.

05_three_models_milestone2.ipynb
Implements final supervised modeling. Builds Logistic Regression, Random Forest, and SVM pipelines with SMOTE class balancing and cross-validation. Evaluates models using AUPRC, precision-recall curves, and feature importance. The Random Forest model performed best overall.

06_three_models_milestone2_PCA.ipynb
Extends the final supervised modeling workflow by incorporating PCA-transformed data. Loads df_best.pkl and the corresponding PCA scores (pca_scores_2025_10_21.csv) to evaluate how dimensionality reduction affects model performance. Compares Logistic Regression, Random Forest, and SVM results with and without PCA features, confirming reduced AUPRC performance (~0.34 vs. 0.61 baseline) but faster training time.

Team_6_Milestone_II_report.pdf
Full detailed report with design summary and outputs.

#Outputs Folder

final_pkl/df_best.pkl: Final dataset

unsupervised_outputs/: PCA and clustering results

To run, install the requirements.txt file and load the preprocessed pkl under outputs/final_pkl/df_best.pkl. The notebooks run independently. You do not need to run in order as presented.