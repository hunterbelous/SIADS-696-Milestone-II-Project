## Predicting Death or Clinical Deterioration from the Emergency Department with Supervised and Unsupervised Learning
**Authors:** Hunter Belous, Allen Chezick, Zach Sletten  
**Course:** SIADS 696 — Milestone II

# Summary

Machine learning project predicting short-term death or clinical deterioration in emergency department patients using the MIMIC-IV / MDS-ED dataset. Combines supervised (Random Forest, Logistic Regression, SVM) and unsupervised (PCA, KMeans) approaches to identify high-risk patient patterns.

Data source from: https://physionet.org/content/multimodal-emergency-benchmark/1.0.0/

# File Types

**00_mds_ed.csv:** Raw CSV of the MIMIC IV dataset. Too big for push. Added a _sample.csv file per rubric requirements. See repo.

**mds_best.csv:** Cleaned and feature engineering dataset that was converted into df_best.pkl. Added a _sample.csv file per rubric requirements. See repo.

**.ipynb:** Jupyter notebooks containing preprocessing, EDA, supervised, and unsupervised modeling scripts.  

**.pkl:** Serialized dataset files (e.g., df_best.pkl) containing cleaned and engineered features. Too big for push. 

**.pdf:** Final written report detailing methods, results, and conclusions.  

**.txt / .md:** Documentation and dependencies (README.md, requirements.txt).  

**/outputs/:** Stores model artifacts and PCA results from unsupervised and supervised phases.

# Overview

This project develops an early warning system using the MIMIC-IV / MDS-ED dataset to predict short-term death or clinical deterioration for emergency department patients. Both supervised and unsupervised learning methods were used to analyze over 120,000 ED visits, focusing on vitals, labs, and demographics collected within the first 1.5 hours of presentation.

# Notebook Breakdown

**01_preprocessing.ipynb**  
Loads and cleans the MDS-ED dataset. Removes identifiers, replaces sentinel values, drops sparse features, rounds continuous variables, and creates derived outcomes such as mortality_28d and death_or_deterioration_any. Saves the engineered dataset as df_best.pkl.

**02_eda_cleaned.ipynb**  
Loads the MDS-ED dataset, performs exploratory data analysis before cleaning and preparing for machine learning with feature selection and engineering.

**03_unsupervised.ipynb**  
Runs PCA with n_components = 0.95, retaining 96% of the variance. Applies KMeans clustering, achieving a best silhouette score of about 0.93 with k = 3. Produces 2D and 3D cluster visualizations and outcome overlays to show subgroup enrichment.

**04_supervised_phase1.ipynb**  
Uses PyCaret to compare multiple supervised modeling techniques across a variety of performance metrics. Also includes a more dedicated comparison of SVM, logistic regression and random forest classifer.  Demonstrates that random forest classification performs best based on AUPRC.  

**05_three_models_milestone2.ipynb**  
Implements final supervised modeling. More in depth exploration of random forest classifier model with the best performing parameters from supervised_phase1.  Probes the model for insights into key features, model stability and failures.  

**06_three_models_milestone2_PCA.ipynb**  
Extends the final supervised modeling workflow by incorporating PCA-transformed data. Loads df_best.pkl and the corresponding PCA scores (pca_scores_2025_10_21.csv) to evaluate how dimensionality reduction affects model performance. Compares Logistic Regression, Random Forest, and SVM results with and without PCA features, confirming reduced AUPRC performance (~0.34 vs. 0.61 baseline) but faster training time.

**Team_6_Milestone_II_report.pdf**  
Full detailed report with design summary and outputs.

# Outputs Folder

**final_pkl/df_best.pkl:** Final dataset  (reference mds_best_sample.csv for final engineered dataset. Cannot import pkl due to size constraints.)
**unsupervised_outputs/:** PCA and clustering results  

To run, install the items as stated in the requirements.txt file. The notebooks run independently of order using the stored pkl file. 