import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.tree import _tree

class ModelToAIG:
    def __init__(self, model, feature_names, target_names=None, X_sample=None):
        """
        Convert a trained ML model into an Action Influence Graph (AIG).

        :param model: Trained ML model (DecisionTree, RandomForest, XGBoost, NeuralNet, etc.)
        :param feature_names: List of feature names
        :param target_names: List of target class names (for classification)
        :param X_sample: Sample input data (needed for SHAP with black-box models)
        """
        self.model = model
        self.feature_names = feature_names
        self.target_names = target_names
        self.X_sample = X_sample  # Needed for SHAP

        self.graph = nx.DiGraph()
        self.node_types = {}
        self.explanations = {}

        self._parse_model()

    def _parse_model(self):
        """Automatically determine how to extract rules based on model type."""
        if hasattr(self.model, "tree_"):  # DecisionTree, RandomForest, XGBoost
            self._parse_tree(self.model.tree_)
        elif hasattr(self.model, "feature_importances_"):  # Feature importance models
            self._parse_feature_importance()
        elif self.X_sample is not None:  # Black-box models (SVM, Neural Networks)
            self._parse_shap_explanations()
        else:
            raise ValueError("Unsupported model type or missing sample data for SHAP.")

    def _parse_tree(self, tree_):
        """Extracts decision paths from tree-based models."""
        def traverse(node_index, parent=None, edge_label=None):
            if tree_.feature[node_index] != _tree.TREE_UNDEFINED:  # Internal node
                feature = self.feature_names[tree_.feature[node_index]]
                threshold = tree_.threshold[node_index]
                node_name = f"{feature} <= {threshold:.2f}"

                self.graph.add_node(node_name)
                self.node_types[node_name] = "event"
                self.explanations[node_name] = f"Decision Split: {node_name}"

                if parent:
                    self.graph.add_edge(parent, node_name)
                    self.explanations[(parent, node_name)] = edge_label

                traverse(tree_.children_left[node_index], node_name, "Yes")
                traverse(tree_.children_right[node_index], node_name, "No")

            else:  # Leaf node
                value = tree_.value[node_index]
                outcome_name = self._get_outcome_label(value)

                self.graph.add_node(outcome_name)
                self.node_types[outcome_name] = "outcome"
                self.explanations[outcome_name] = f"Outcome: {outcome_name}"

                if parent:
                    self.graph.add_edge(parent, outcome_name)
                    self.explanations[(parent, outcome_name)] = edge_label

        traverse(0)

    def _parse_feature_importance(self):
        """Extracts feature influences from models with feature_importances_ attribute."""
        importance = self.model.feature_importances_
        significant_features = []
        for i, imp in enumerate(importance):
            if imp > 0:
                node_name = self.feature_names[i]
                self.graph.add_node(node_name)
                self.node_types[node_name] = "event"
                self.explanations[node_name] = f"Feature Importance: {imp:.4f}"
                significant_features.append(node_name)

        # Create edges to form a chain-like structure (if there are at least 2 features)
        if len(significant_features) >= 2:
            for i in range(len(significant_features) - 1):
                self.graph.add_edge(significant_features[i], significant_features[i + 1])
                self.explanations[(significant_features[i], significant_features[i + 1])] = "Sequential"  # Label the edges

    def _parse_shap_explanations(self):
        """Uses SHAP KernelExplainer for Neural Networks."""
        if self.X_sample is None:
            raise ValueError("SHAP requires X_sample (sample input data) for explanation.")

        # Check if model supports predict_proba (for classification)
        predict_fn = self.model.predict_proba if hasattr(self.model, "predict_proba") else self.model.predict

        explainer = shap.KernelExplainer(predict_fn, shap.sample(self.X_sample, 5))
        shap_values = explainer.shap_values(self.X_sample[:5])  # Compute SHAP for a small subset

        print(f"SHAP values computed: {np.shape(shap_values)}")  # Debugging check

        # Calculate the mean absolute SHAP value for each feature across all samples/classes
        mean_shap = np.abs(np.array(shap_values)).mean(axis=0)

        # If shap_values is a 3D array, take the mean across the classes as well
        if mean_shap.ndim == 2:
            mean_shap = mean_shap.mean(axis=0)

        # Create nodes for features with non-zero SHAP values
        significant_features = []
        for i, val in enumerate(mean_shap):
            if val > 0:
                node_name = self.feature_names[i]
                self.graph.add_node(node_name)
                self.node_types[node_name] = "event"
                self.explanations[node_name] = f"SHAP Impact: {val:.4f}"
                significant_features.append(node_name)

        # Create edges to form a chain-like structure (if there are at least 2 features)
        if len(significant_features) >= 2:
            for i in range(len(significant_features) - 1):
                self.graph.add_edge(significant_features[i], significant_features[i + 1])
                self.explanations[(significant_features[i], significant_features[i + 1])] = "Sequential"  # Label the edges



    def _get_outcome_label(self, value):
        """
        Returns a readable outcome label for both classification and regression models.
        """
        value = np.squeeze(value)  # Ensure it's a 1D array

        if hasattr(self.model, "classes_"):  # Classification model
            outcome_class = np.argmax(value)  # Get index of max probability/class
            return self.target_names[outcome_class] if self.target_names is not None else f"Class {outcome_class}"

        # Regression model: Just return the numeric value
        return f"Value: {float(value):.2f}"  # Convert to float for readability


    def visualize(self, color_by_cluster=False):
        """
        Generates interactive AIG visualization using Plotly.
        Can optionally color nodes by cluster ID if available.
        """
        import plotly.graph_objects as go
        import plotly.express as px # Needed for color scales

        # Use Kamada-Kawai layout for potentially better structure
        try:
            pos = nx.kamada_kawai_layout(self.graph)
        except nx.NetworkXException: # Fallback if Kamada-Kawai fails (e.g., disconnected graph)
            print("Warning: Kamada-Kawai layout failed, falling back to spring layout.")
            pos = nx.spring_layout(self.graph)

        node_x = []
        node_y = []
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[node for node in self.graph.nodes()],
            hoverinfo='text',
            marker=dict(
                showscale=False,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Portland' | 'Jet' | 'Hot' | 'Blackbody' | 'Earth' |
                #'Electric' | 'Viridis' |
                #colorscale='YlGnBu',
                #color=[{"event": "#4ECDC4", "outcome": "#45B7D1"}[self.node_types[node]] for node in self.graph.nodes()], # Original color
                size=30,
                line_width=2))

        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_hover_text = []
        for node in self.graph.nodes():
            node_info = f"{node} - {self.explanations[node]}"
            if 'cluster_id' in self.graph.nodes[node]:
                node_info += f"<br>Cluster ID: {self.graph.nodes[node]['cluster_id']}"
            if 'influence_score' in self.graph.nodes[node]:
                node_info += f"<br>Influence Score: {self.graph.nodes[node]['influence_score']:.4f}"
            node_hover_text.append(node_info)
        node_trace.hovertext = node_hover_text

        # Determine node colors
        if color_by_cluster and 'cluster_id' in next(iter(self.graph.nodes(data=True)))[1]:
            cluster_ids = [data['cluster_id'] for node, data in self.graph.nodes(data=True)]
            num_clusters = len(set(cluster_ids))
            colors = px.colors.qualitative.Plotly[:num_clusters] # Use Plotly's qualitative colors
            node_colors = [colors[cid % len(colors)] for cid in cluster_ids]
            node_trace.marker.color = node_colors
            node_trace.marker.showscale = False # No color scale needed for categorical colors
        else:
            # Default color by node type
            node_trace.marker.color = [{"event": "#4ECDC4", "outcome": "#45B7D1"}[self.node_types[node]] for node in self.graph.nodes()]

        # Add arrows for directed edges
        annotations = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            annotations.append(
                dict(
                    ax=x0, ay=y0, axref='x', ayref='y',
                    x=x1, y=y1, xref='x', yref='y',
                    showarrow=True,
                    arrowhead=2,  # Style of arrowhead
                    arrowsize=2,  # Increased size
                    arrowwidth=2, # Increased width
                    arrowcolor='#888'
                )
            )

        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title=f'{self.model} Action Influence Graph (AIG)',
                        title_font_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=annotations + [ dict( # Combine edge arrows with existing annotation
                            text="AIG Visualization",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.show()

    def explain(self):
        """Prints decision-making rules or feature influences."""
        explanations = []
        for node in self.graph.nodes:
            if self.node_types[node] == "event":
                # If it's an event node, simply print its explanation
                explanations.append(f"Feature: {node} - {self.explanations[node]}")
            elif self.node_types[node] == "outcome":
                # If it's an outcome node, print the path to it
                paths = list(nx.all_simple_paths(self.graph, source=list(self.graph.nodes)[0], target=node))
                for path in paths:
                    rule = " → ".join(path)
                    explanations.append(f"Rule: {rule}")
        return "\n".join(explanations)

    def calculate_shortest_path(self, source_node, target_node):
        """
        Calculates the shortest path between two nodes in the AIG.
        """
        try:
            shortest_path = nx.shortest_path(self.graph, source=source_node, target=target_node)
            return shortest_path
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

    def calculate_centrality_measures(self):
        """
        Calculates various centrality measures for the AIG nodes.
        """
        centrality_measures = {}
        centrality_measures['betweenness_centrality'] = nx.betweenness_centrality(self.graph)
        centrality_measures['closeness_centrality'] = nx.closeness_centrality(self.graph)
        try:
            centrality_measures['eigenvector_centrality'] = nx.eigenvector_centrality(self.graph, max_iter=1000) # Increased iterations for convergence
        except nx.PowerIterationFailedConvergence:
            centrality_measures['eigenvector_centrality'] = {}  # Return empty dict if convergence fails
        centrality_measures['graph_density'] = nx.density(self.graph)
        # Clustering coefficient is not well-defined for directed graphs, using average clustering
        centrality_measures['average_clustering'] = nx.average_clustering(self.graph.to_undirected())
        return centrality_measures

    def calculate_influence_scores(self):
        """
        Calculates influence scores for nodes based on eigenvector centrality.
        Stores scores as 'influence_score' node attribute.
        """
        try:
            influence_scores = nx.eigenvector_centrality(self.graph, max_iter=1000)
            nx.set_node_attributes(self.graph, influence_scores, "influence_score")
            return influence_scores
        except nx.PowerIterationFailedConvergence:
            print("Warning: Eigenvector centrality failed to converge. Influence scores not calculated.")
            return {}

    def cluster_nodes(self, n_clusters=3): # Default to 3 clusters
        """
        Performs node clustering using Spectral Clustering on the undirected graph.
        Stores cluster assignments as 'cluster_id' node attribute.
        :param n_clusters: The number of clusters to form.
        """
        from sklearn.cluster import SpectralClustering

        # Spectral Clustering works on the adjacency matrix of the undirected graph
        undirected_graph = self.graph.to_undirected()
        node_list = list(undirected_graph.nodes()) # Keep track of node order

        if undirected_graph.number_of_nodes() < n_clusters or undirected_graph.number_of_edges() == 0:
            print(f"Warning: Not enough nodes/edges for {n_clusters} clusters. Assigning each node to its own cluster.")
            partition = {node: i for i, node in enumerate(node_list)}
        else:
            try:
                adjacency_matrix = nx.adjacency_matrix(undirected_graph, nodelist=node_list)
                # Convert sparse matrix to dense numpy array to avoid int64 index issue
                adjacency_matrix_dense = adjacency_matrix.toarray()

                # Apply Spectral Clustering
                sc = SpectralClustering(n_clusters=n_clusters,
                                        assign_labels='discretize', # Common strategy for spectral clustering
                                        random_state=0,
                                        affinity='precomputed', # Use precomputed adjacency matrix
                                        n_init=10) # Number of times to run with different centroid seeds
                labels = sc.fit_predict(adjacency_matrix_dense) # Use dense array

                # Create partition dictionary
                partition = {node: label for node, label in zip(node_list, labels)}
            except Exception as e:
                print(f"Warning: Spectral Clustering failed ({e}). Assigning each node to its own cluster.")
                partition = {node: i for i, node in enumerate(node_list)}

        # Store cluster id as a node attribute
        nx.set_node_attributes(self.graph, partition, "cluster_id")
        return partition

    def calculate_shortest_path(self, source_node, target_node):
        """
        Calculates the shortest path between two nodes in the AIG.
        """
        try:
            shortest_path = nx.shortest_path(self.graph, source=source_node, target=target_node)
            return shortest_path
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

# Example Usage:
if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    import shap

    # Load dataset and train models
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    dt_model = DecisionTreeClassifier(max_depth=3).fit(X, y)
    rf_model = RandomForestClassifier(n_estimators=10, max_depth=5).fit(X, y)

    # Convert DT to AIG
    dt_to_aig = ModelToAIG(dt_model, feature_names, target_names)
    dt_to_aig.visualize()
    print("\nDecision Tree-Based Rules:\n")
    print(dt_to_aig.explain())
    print("\nDecision Tree AIG Edges:\n")
    print(dt_to_aig.graph.edges)
    
    # Shortest Path Example
    start_node_dt = list(dt_to_aig.graph.nodes)[0]
    end_node_dt = list(dt_to_aig.graph.nodes)[-1]
    shortest_path_dt = dt_to_aig.calculate_shortest_path(start_node_dt, end_node_dt)
    if shortest_path_dt:
        print(f"\nShortest Path in Decision Tree AIG from '{start_node_dt}' to '{end_node_dt}': {shortest_path_dt}")
    else:
        print(f"\nNo path found in Decision Tree AIG from '{start_node_dt}' to '{end_node_dt}'")

    # Centrality Measures Example
    centrality_dt = dt_to_aig.calculate_centrality_measures()
    print("\nCentrality Measures for Decision Tree AIG:")
    for measure, values in centrality_dt.items():
        print(f"\n{measure}:")
        if isinstance(values, dict):  # Check if values is a dictionary
            for node, value in values.items():
                print(f"  {node}: {value:.4f}")
        else:  # Handle float values for graph density and average clustering
            print(f"  {measure}: {values:.4f}")

    # Influence Scores Example (DT) - Moved after initialization
    influence_scores_dt = dt_to_aig.calculate_influence_scores()
    print("\nInfluence Scores for Decision Tree AIG:")
    if influence_scores_dt:
        for node, score in influence_scores_dt.items():
            print(f"  {node}: {score:.4f}")
    else:
        print("  Influence scores could not be calculated.")

    # Cluster Nodes Example (DT)
    partition_dt = dt_to_aig.cluster_nodes()
    print("\nNode Clustering (Partition) for Decision Tree AIG:")
    print(partition_dt)
    # Visualize with cluster colors
    print("\nVisualizing Decision Tree AIG with Cluster Colors...")
    dt_to_aig.visualize(color_by_cluster=True)
    # Note: Removed duplicate/misplaced influence_scores_rf call from here
    # Convert Random Forest to AIG
    rf_to_aig = ModelToAIG(rf_model, feature_names, target_names)
    print("\nRandom Forest-Based Rules:\n")
    print(rf_to_aig.explain())
    # Calculate influence scores before visualizing
    influence_scores_rf = rf_to_aig.calculate_influence_scores()
    rf_to_aig.visualize() # Visualize after calculation

    # Shortest Path Example
    start_node_rf = list(rf_to_aig.graph.nodes)[0]
    end_node_rf = list(rf_to_aig.graph.nodes)[-1]
    shortest_path_rf = rf_to_aig.calculate_shortest_path(start_node_rf, end_node_rf)
    if shortest_path_rf:
        print(f"\nShortest Path in Random Forest AIG from '{start_node_rf}' to '{end_node_rf}': {shortest_path_rf}")
    else:
        print(f"\nNo path found in Random Forest AIG from '{start_node_rf}' to '{end_node_rf}'")

    # Centrality Measures Example
    centrality_rf = rf_to_aig.calculate_centrality_measures()
    print("\nCentrality Measures for Random Forest AIG:")
    for measure, values in centrality_rf.items():
        print(f"\n{measure}:")
        if isinstance(values, dict):  # Check if values is a dictionary
            for node, value in values.items():
                print(f"  {node}: {value:.4f}")
        else:  # Handle float values for graph density and average clustering
            print(f"  {measure}: {values:.4f}")

    
    # Cluster Nodes Example (RF)
    partition_rf = rf_to_aig.cluster_nodes()
    print("\nNode Clustering (Partition) for Random Forest AIG:")
    print(partition_rf)
    # Visualize with cluster colors
    print("\nVisualizing Random Forest AIG with Cluster Colors...")
    rf_to_aig.visualize(color_by_cluster=True)
    # Convert Neural Network (Black Box Model) to AIG using SHAP
    from sklearn.neural_network import MLPClassifier
    nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, learning_rate_init=0.01).fit(X, y)

    # Initialize NN AIG
    nn_to_aig = ModelToAIG(nn_model, feature_names, target_names, X_sample=X[:50])
    print("\nNeural Network-Based SHAP Explanations:\n")
    print(nn_to_aig.explain())

    # Calculate Influence Scores Example (NN) - Moved after initialization
    influence_scores_nn = nn_to_aig.calculate_influence_scores()
    print("\nInfluence Scores for Neural Network AIG:")
    if influence_scores_nn:
        for node, score in influence_scores_nn.items():
            print(f"  {node}: {score:.4f}")
    else:
        print("  Influence scores could not be calculated.")

    # Visualize NN AIG - Moved after score calculation
    nn_to_aig.visualize()

    # Cluster Nodes Example (NN)
    partition_nn = nn_to_aig.cluster_nodes()
    print("\nNode Clustering (Partition) for Neural Network AIG:")
    print(partition_nn)
    # Visualize with cluster colors
    print("\nVisualizing Neural Network AIG with Cluster Colors...")
    nn_to_aig.visualize(color_by_cluster=True)

    # Shortest Path Example (NN)
    start_node_nn = list(nn_to_aig.graph.nodes)[0]
    end_node_nn = list(nn_to_aig.graph.nodes)[-1]
    shortest_path_nn = nn_to_aig.calculate_shortest_path(start_node_nn, end_node_nn)
    if shortest_path_nn:
        print(f"\nShortest Path in Neural Network AIG from '{start_node_nn}' to '{end_node_nn}': {shortest_path_nn}")
    else:
        print(f"\nNo path found in Neural Network AIG from '{start_node_nn}' to '{end_node_nn}'")

    # Centrality Measures Example (NN)
    centrality_nn = nn_to_aig.calculate_centrality_measures()
    print("\nCentrality Measures for Neural Network AIG:")
    for measure, values in centrality_nn.items():
        print(f"\n{measure}:")
        if isinstance(values, dict):  # Check if values is a dictionary
            for node, value in values.items():
                print(f"  {node}: {value:.4f}")
        else:  # Handle float values for graph density and average clustering
            print(f"  {measure}: {values:.4f}")
