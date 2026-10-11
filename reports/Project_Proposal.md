# Integrated Course Project Proposal

**Course Name**: Probability and Statistics  
**Course Code**: BAMAT207  
**Faculty Evaluator**: Dr. Pallavi Mishra  
**Academic Term**: Semester 3 (2026)  

---

### 1. Project Title
**Weather Prediction using Conditional Probability and Bayesian Intelligence: A Comparative Analysis of Empirical, Markovian, and Bayesian Formulations**

### 2. Team Information
- **Student Name / Group**: Individual / Group Submission (Max 3 Members)
  - Student 1 (Lead Investigator): *[Student Name / Enrollment Number]*
  - Student 2: *[Team Member 2 Name]*
  - Student 3: *[Team Member 3 Name]*
- **Department**: Computer Science & Engineering / Data Science

### 3. Selected Track
- **Track**: **TRACK 1: Bayesian Intelligence**
- **Core Syllabus Concepts Addressed**:
  - Conditional Probability: $P(A \mid B) = \frac{P(A \cap B)}{P(B)}$
  - Joint Probability Distributions: $P(A \cap B \cap C)$
  - Law of Total Probability: $P(B) = \sum_{i} P(B \mid A_i) P(A_i)$
  - Prior and Posterior Probability Dynamics: $P(Y \mid \mathbf{X}) \propto P(\mathbf{X} \mid Y) P(Y)$
  - Bayes' Theorem & Evidence Normalization: $P(Y=c \mid \mathbf{X}) = \frac{P(\mathbf{X} \mid Y=c)P(Y=c)}{\sum_k P(\mathbf{X} \mid Y=k)P(Y=k)}$
  - Discrete-Time Markov Property: $P(W_{t+1} \mid W_t, W_{t-1}, \dots, W_0) = P(W_{t+1} \mid W_t)$ and Stationary Distribution $\boldsymbol{\pi} = \boldsymbol{\pi} \mathbf{P}$

### 4. Problem Statement & Motivation
Atmospheric precipitation forecasting is a canonical problem in physical and environmental sciences. Traditional Numerical Weather Prediction (NWP) models require intensive supercomputing resources to solve Navier-Stokes fluid mechanics equations. However, short-term precipitation events exhibit strong probabilistic persistence and observable dependencies on atmospheric variables such as relative humidity, barometric pressure gradients, temperature dew-point depression, and wind speed. 

This project aims to bridge rigorous mathematical probability theory with real-world meteorological decision-making by constructing pure mathematical probabilistic models from first principles and assessing their predictive reliability against machine learning baselines.

### 5. Datasets & Methodology
- **Dataset**: 10-year observational meteorological time-series (3,652 daily observations from Open-Meteo Historical Atmospheric Archive).
- **Core Modeling Pipelines**:
  1. Empirical Conditional Probability Tables (with Laplace smoothing).
  2. First-Order and Second-Order Markov Chain state transitions.
  3. Continuous Gaussian Bayesian Classifier (Bayes' Theorem from scratch).
  4. Probabilistic calibration (Brier Score, Reliability Diagrams, Log-Loss).
- **Deliverables**: Complete source code, automated visualization pipeline, IEEE 8-10 page conference paper, and Viva defense documentation.

---
**Date of Submission**: October 2026  
**Status**: Approved & Initiated
