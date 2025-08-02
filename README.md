# 🧠 ModelToAIG: Action Influence Graphs for ML Interpretability

`ModelToAIG` is a Python tool that converts trained machine learning models into **Action Influence Graphs (AIGs)** for better explainability and visualization. It supports decision trees, random forests, and black-box models (like neural networks) using SHAP.

## 🚀 Features

- Convert ML models to graph-based explanations
- Supports:
  - `DecisionTreeClassifier`
  - `RandomForestClassifier`
  - `MLPClassifier` (and other black-box models via SHAP)
- Visualize decision logic interactively with Plotly
- Calculate:
  - Centrality metrics
  - Influence scores
  - Shortest paths
- Perform node clustering using Spectral Clustering

## 📦 Installation

Install required packages using pip:

```bash
pip install numpy networkx matplotlib shap scikit-learn plotly
```

Note: SHAP can be slow on large datasets. Consider sampling your input.

## 📁 File Overview

- `project.py` — Contains the full `ModelToAIG` class and example usage.
- Example usage with:
  - Decision Trees
  - Random Forests
  - Neural Networks (via SHAP)

## 🧩 Usage

```python
from project import ModelToAIG
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load data
data = load_iris()
X, y = data.data, data.target

# Train model
model = DecisionTreeClassifier(max_depth=3).fit(X, y)

# Create AIG
aig = ModelToAIG(model, data.feature_names, data.target_names)

# Visualize
aig.visualize()

# Print explanations
print(aig.explain())
```

## 📊 Visualizations

- Graphs are rendered using Plotly.
- Nodes are colored by type (event vs outcome) or by clusters.
- Arrows show direction of influence.

## 🧠 SHAP Integration

For black-box models (e.g., neural networks):

```python
from sklearn.neural_network import MLPClassifier

nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000).fit(X, y)
aig_nn = ModelToAIG(nn_model, data.feature_names, data.target_names, X_sample=X[:50])
aig_nn.visualize()
```

## 📈 Graph Analysis

```python
# Centrality
centrality = aig.calculate_centrality_measures()

# Influence scores
influence_scores = aig.calculate_influence_scores()

# Clustering
clusters = aig.cluster_nodes(n_clusters=3)

# Shortest path
path = aig.calculate_shortest_path(source_node="petal length (cm) <= 2.45", target_node="setosa")
```

## 🧪 Supported Models

| Model Type         | Supported | Notes                                     |
|--------------------|-----------|-------------------------------------------|
| Decision Trees      | ✅        | Full path tracing                         |
| Random Forests      | ✅        | Feature importance based                  |
| Neural Networks     | ✅        | SHAP KernelExplainer (needs X_sample)     |
| SVMs or others      | ⚠️        | SHAP-based if `X_sample` is provided      |

## 📄 License

MIT License © 2025
