# Smart Knowledge Engine based on WebKB

Virtual Env: anaconda python3.10.9

The Smart Knowledge Engine leverages the WebKB dataset to enable advanced text mining, entity recognition, and visualization. This system is designed to integrate traditional machine learning methods and modern natural language processing techniques to provide actionable insights. The following sections outline its conceptual framework, example applications, feature visualization, and evaluation results.
## Concept frame:

![Alt Text](img/frame.png)

The conceptual framework outlines the key components and interactions of the knowledge engine, including data preprocessing, feature extraction, model training, and evaluation. It serves as the foundation for building intelligent and interpretable models.

## An example for Entity Recognition:
This example demonstrates the entity recognition capability of the system. Named entities such as locations, organizations, and people are highlighted, showcasing the accuracy and relevance of the entity recognition pipeline. It uses pre-trained models fine-tuned on WebKB data to identify domain-specific entities.
![Alt Text](img/entity_recognition_example.png)

## Word Cloud
The word cloud provides a high-level visualization of the most frequently occurring terms in the dataset. Larger words indicate higher frequency, allowing users to quickly identify key topics and trends within the dataset.
![Alt Text](img/word_cloud_example.png)

## Local Dataset Overview 
This overview highlights the structure of the dataset, enabling a better understanding of its composition and complexity.
![Alt Text](img/data_statis.jpg)

## 2 dimensional features
Using dimensionality reduction techniques like PCA (Principal Component Analysis) or t-SNE (t-Distributed Stochastic Neighbor Embedding), the dataset is visualized in two dimensions. This visualization helps identify clusters and outliers, offering insights into the separability of different classes.
![Alt Text](img/2dim_features.jpg)

## Traditional Method

### SVM
SVM is effective for this dataset due to its ability to handle high-dimensional feature spaces. The confusion matrix highlights the areas where the model excels and where it could be improved.
![Alt Text](img/SVM_confusion_matrix.jpg)

### Gaussian Naive Bayes
Gaussian Naive Bayes is a fast and interpretable method that works well for datasets with normally distributed features. Despite being slightly less accurate than SVM, it provides valuable probabilistic insights.
![Alt Text](img/NB_confusion_matrix.jpg)


