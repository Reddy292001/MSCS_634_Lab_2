Wine Classification with KNN and RNN Classifiers
This repository contains a Jupyter Notebook for a lab assignment that explores and compares the performance of K-Nearest Neighbors (KNN) and Radius Neighbors (RNN) classifiers on the sklearn Wine dataset.

Purpose
The primary goal of this lab was to:

Implement and evaluate K-Nearest Neighbors (KNN) and Radius Neighbors (RNN) classifiers.
Analyze how the choice of key parameters—k for KNN and radius for RNN—impacts model accuracy.
Visualize the performance trends to identify optimal parameter values.
Compare the two models and provide insights into when one might be preferable over the other for a given dataset.
Key Insights
K-Nearest Neighbors (KNN) Performance
The accuracy of the KNN model was highly sensitive to the number of neighbors, k.

Low k (e.g., k=1): The model was likely overfitting to the training data, resulting in lower accuracy on the test set due to high variance and sensitivity to noise.
Optimal k (e.g., k=5, 11): A "sweet spot" was observed where the model achieved its highest accuracy. At these values, the classifier effectively captured the local structure of the data without being overly influenced by noise or outliers.
High k (e.g., k=21): As k became too large, the model started to underfit. The decision boundaries became overly smooth, and the model's accuracy decreased as it began to misclassify points near class boundaries by including too many neighbors from other classes.
Radius Neighbors (RNN) Performance
The RNN model's performance was critically dependent on the radius value.

Small Radius: A radius that is too small can lead to instances where a test point has no neighbors within the specified radius, causing a ValueError. This makes RNN less robust to sparse regions of the feature space.
Increasing Radius: As the radius increased, more neighbors were included for classification, which initially improved accuracy. However, there is a point of diminishing returns.
Large Radius: An excessively large radius can encompass points from multiple classes, leading to poor classification accuracy and effectively blurring the class boundaries.
Comparison and Observations
For the Wine dataset, KNN generally outperformed RNN in both accuracy and stability.

KNN's Advantage: The Wine dataset has relatively uniform density and well-defined clusters. KNN's approach of using a fixed number of neighbors is more effective in this scenario, as it adapts to local density variations naturally.
RNN's Challenge: Selecting an appropriate radius is difficult. A single global radius may be too small for sparse regions and too large for dense regions, leading to inconsistent performance across the dataset.
Conclusion: For datasets with consistent class density like the Wine dataset, KNN is the more reliable and straightforward choice. RNN might be preferable in datasets with significant variations in point density, where a fixed distance threshold is more meaningful than a fixed neighbor count.
Challenges and Decisions
Challenges Faced
Selecting RNN Radius: Choosing an appropriate range for the radius parameter was the primary challenge. Unlike k, the radius is a distance value in a multi-dimensional space, making its initial selection non-intuitive. We had to handle potential ValueError exceptions for radii that were too small.
Data Scaling: Both KNN and RNN are distance-based algorithms, meaning they are highly sensitive to the scale of the features. Failing to standardize the data would have resulted in features with larger ranges dominating the distance calculation, leading to biased and inaccurate models.
Decisions Made
Standardization: A deliberate decision was made to use StandardScaler to normalize all features to have a mean of 0 and a standard deviation of 1 before training the models. This ensures that all features contribute equally to the distance computation.
Systematic Parameter Tuning: Instead of guessing, we implemented a systematic approach, testing a predefined list of k and radius values to plot their effect on accuracy and identify the optimal parameter empirically.
Error Handling for RNN: To prevent the RNN evaluation loop from crashing, a try-except block was implemented to catch ValueError for cases where no neighbors were found, assigning an accuracy of 0 for that specific radius value.
