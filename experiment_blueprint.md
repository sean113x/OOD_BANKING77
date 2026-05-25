## Dataset Analysis

Before running the main experiments, we analyze the BANKING77 ID/OOD split and the embedding space.

### Dataset Analysis Plots

| Plot                                             | Purpose                                                                     |
| ------------------------------------------------ | --------------------------------------------------------------------------- |
| Intent-wise sample count plot                    | Check whether the dataset is balanced across intents                        |
| Known/OOD split count plot                       | Show how many samples are used for known training, known test, and OOD test |
| UMAP or t-SNE embedding plot                     | Visualize whether held-out OOD intents are close to known intents           |
| OOD-to-known centroid similarity histogram       | Quantify the difficulty of near-OOD intents                                 |
| Intra-class vs inter-class distance distribution | Analyze how well intent classes are separated in embedding space            |

### Key Point

BANKING77 held-out intents are not random outliers from a different domain.  
They are semantically close banking queries, so the task should be treated as a near-OOD detection problem.

## Research Questions

### RQ1. Overall OOD Performance

Which OOD method family performs best for held-out banking intent detection?

### RQ2. Hyperparameter and Threshold Sensitivity

How sensitive are OOD methods to hyperparameter choices and threshold calibration?

### RQ3. Near-OOD Difficulty

Are held-out intents that are semantically closer to known intents harder to detect?

### RQ4. Classification Accuracy vs OOD Reliability

Does higher closed-set intent classification accuracy imply better OOD detection performance?

### RQ5. Advanced OOD Techniques

Do advanced or SOTA-inspired techniques improve over simple OOD baselines?

## Experiment 1: Overall Method Comparison

### Goal

Compare all implemented OOD methods under the same embedding model, same known/OOD split, and same evaluation metrics.

### Compared Method Families

| Family                                  | Methods                                                                                           |
| --------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Class-wise boundary                     | Centroid distance, class-wise radius threshold, class-wise Mahalanobis                            |
| Distance / support / density estimation | kNN distance, global Mahalanobis, GMM, One-Class SVM                                              |
| Anomaly detection                       | Isolation Forest, LOF                                                                             |
| Classifier-output OOD scoring           | Logistic Regression, LDA, Gaussian Naive Bayes, MLP + MSP, entropy, margin, maximum logit, energy |
| Advanced / SOTA-inspired                | Temperature-scaled scores, ensemble disagreement                                                  |

### Metrics

| Metric        | Meaning                                                      |
| ------------- | ------------------------------------------------------------ |
| AUROC         | Threshold-independent OOD detection performance              |
| AUPR-OOD      | Precision-recall performance when OOD is treated as positive |
| FPR95         | False positive rate when OOD recall is 95%                   |
| OOD Precision | Among rejected samples, how many are truly OOD               |
| OOD Recall    | Among OOD samples, how many are correctly rejected           |
| OOD F1        | Harmonic mean of OOD precision and recall                    |
| ID Accuracy   | Closed-set classification accuracy on known intents          |
| Macro F1      | Balanced classification performance across known intents     |

### Main Analysis

This experiment answers which method family is strongest overall and whether embedding-distance methods, anomaly detection methods, or classifier-confidence methods are more reliable for near-OOD banking queries.

## Experiment 2: Hyperparameter and Threshold Sensitivity

### Goal

Analyze how much OOD performance changes depending on hyperparameters and threshold selection.

### Hyperparameters to Analyze

| Method                    | Hyperparameters                          |
| ------------------------- | ---------------------------------------- |
| Class-wise radius         | radius percentile: 90, 95, 97.5, 99      |
| kNN distance              | k: 1, 5, 10, 20, 50                      |
| One-Class SVM             | kernel, gamma, nu                        |
| GMM                       | n_components, covariance_type            |
| PCA reconstruction        | n_components or explained variance       |
| Logistic Regression       | regularization C                         |
| MLP                       | hidden dimension, dropout, learning rate |
| Temperature-scaled scores | temperature T                            |

### Threshold Strategies

| Strategy         | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| Fixed threshold  | Use a predefined threshold such as radius score > 1             |
| ID95 threshold   | Choose a threshold that accepts 95% of known validation samples |
| OOD-F1 threshold | Choose a threshold that maximizes OOD F1 on validation data     |

### Main Analysis

AUROC is threshold-independent, but deployment requires a concrete accept/reject threshold.  
This experiment shows whether each method remains stable under different hyperparameters and threshold choices.

## Experiment 3: Near-OOD Difficulty Analysis

### Goal

Analyze whether semantically similar held-out intents are harder to detect as OOD.

### Method

1. Compute the centroid of each known intent using training embeddings.
2. Compute the centroid of each held-out OOD intent.
3. For each OOD intent, compute the maximum cosine similarity to known intent centroids.
4. Group OOD intents into easy, medium, and hard OOD groups.
5. Evaluate OOD detection performance separately for each group.

### Difficulty Definition

| Difficulty | Meaning                                                                 |
| ---------- | ----------------------------------------------------------------------- |
| Easy OOD   | OOD intent centroid is far from all known intent centroids              |
| Medium OOD | OOD intent centroid has moderate similarity to known intents            |
| Hard OOD   | OOD intent centroid is very close to at least one known intent centroid |

### Main Analysis

Unsupported banking queries are difficult because they are often semantically close to supported banking intents.  
This experiment directly tests whether semantic closeness causes OOD detection failure.

## Experiment 4: Classification Accuracy vs OOD Reliability

### Goal

Test whether a stronger closed-set classifier also gives better OOD detection performance.

### Motivation

In our preliminary results, the MLP classifier achieved the best known-intent classification performance, but its confidence-based OOD scores were not strong.  
This suggests that better classification accuracy does not necessarily imply better OOD reliability.

### Compared Classifiers

| Classifier           | Output Used           |
| -------------------- | --------------------- |
| Logistic Regression  | probabilities, logits |
| LDA                  | probabilities         |
| Gaussian Naive Bayes | probabilities         |
| MLP Classifier       | probabilities, logits |

### OOD Scores

| Score         | Description                                       |
| ------------- | ------------------------------------------------- |
| MSP           | Low maximum softmax probability indicates OOD     |
| Entropy       | High predictive entropy indicates OOD             |
| Margin        | Small top-1 minus top-2 probability indicates OOD |
| Maximum logit | Low maximum logit indicates OOD                   |
| Energy        | High energy indicates OOD                         |

### Analyses

| Analysis                       | Purpose                                                             |
| ------------------------------ | ------------------------------------------------------------------- |
| Known accuracy and macro F1    | Measure closed-set classification performance                       |
| OOD AUROC and FPR95            | Measure OOD detection performance                                   |
| Accuracy vs AUROC scatter plot | Check whether classification accuracy correlates with OOD detection |
| ID/OOD MSP histogram           | Visualize confidence separation                                     |
| OOD samples with MSP > 0.9     | Measure overconfidence on held-out intents                          |

### Main Analysis

A better closed-set classifier is not necessarily a better OOD detector.  
The MLP may learn stronger known-class decision boundaries, but because it is not trained to reject unknown intents, it can still assign high confidence to held-out banking intents.

## Experiment 5: Advanced / SOTA-Inspired Techniques

### Goal

Evaluate whether advanced techniques improve over simple OOD baselines.

### Methods

| Method                     | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| Temperature-scaled MSP     | Apply temperature scaling before computing maximum softmax probability |
| Temperature-scaled entropy | Apply temperature scaling before computing entropy                     |
| Temperature-scaled energy  | Use temperature-scaled logits in the energy score                      |
| Ensemble disagreement      | Use disagreement across multiple classifiers as an OOD signal          |

### Main Analysis

This experiment tests whether calibration and ensemble-based uncertainty can improve classifier-output OOD scoring, especially when simple confidence scores are overconfident on near-OOD examples.

## Final Experiment Flow

1. Dataset Analysis
   - Explain the constructed known/OOD split.
   - Visualize sample counts and embedding structure.
   - Show that the task is a near-OOD banking intent detection problem.

2. Experiment 1: Overall Method Comparison
   - Compare all OOD methods using the same split and metrics.

3. Experiment 2: Hyperparameter and Threshold Sensitivity
   - Analyze how method performance changes with hyperparameters and threshold strategies.

4. Experiment 3: Near-OOD Difficulty Analysis
   - Group OOD intents by similarity to known intents and evaluate performance by difficulty.

5. Experiment 4: Classification Accuracy vs OOD Reliability
   - Show that high closed-set accuracy does not necessarily imply strong OOD detection.

6. Experiment 5: Advanced / SOTA-Inspired Techniques
   - Test whether temperature scaling and ensemble disagreement improve OOD detection.

## Key Expected Findings

1. BANKING77 held-out intent OOD detection is difficult because the OOD examples are semantically close to known banking intents.

2. Classifier confidence alone may be unreliable for near-OOD detection.

3. Higher closed-set classification accuracy does not necessarily lead to better OOD detection.

4. Threshold selection strongly affects practical OOD precision, recall, and F1.

5. Embedding-distance and class-wise boundary methods may provide more interpretable OOD signals than pure classifier-output confidence scores.

6. Temperature scaling and ensemble disagreement may reduce overconfidence, but they should be compared against simple baselines such as kNN, Mahalanobis, MSP, and energy.
