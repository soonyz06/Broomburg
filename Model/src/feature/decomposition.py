import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.metrics.pairwise import pairwise_distances
import copy
import networkx as nx
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from pandas.api.types import is_object_dtype, is_string_dtype

from src.utils.logg import log_plot

def pca_scree_plot(X, model): #how many components to keep
    if not hasattr(model, "explained_variance_ratio_"): return
    explained_variance = model.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(range(1, len(explained_variance)+1), explained_variance,
                 marker='o', linestyle='--')
    axes[0].set_title('Scree Plot')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_xticks(range(1, len(explained_variance)+1))
    axes[0].tick_params(axis="x", rotation=90, labelsize=5)
    axes[0].grid(True)

    axes[1].plot(range(1, len(cumulative_variance)+1), cumulative_variance,
                 marker='o', linestyle='--', color='orange')
    axes[1].set_title(f'Cumulative Explained Variance')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Cumulative Variance Ratio')
    axes[1].set_xticks(range(1, len(cumulative_variance)+1))
    axes[1].grid(True)
    axes[1].tick_params(axis="x", labelsize=5)
    plt.tight_layout()
    log_plot(fig)

def fa_noise_plot(X, model):
    if not hasattr(model, "noise_variance_"): return
    noise_var = model.noise_variance_
    if np.allclose(noise_var, 0): return
    
    fig = plt.figure(figsize=(8,5))
    plt.bar(X.columns, noise_var, color="skyblue", edgecolor="k")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Noise variance (idiosyncratic)")
    plt.tight_layout()
    log_plot(fig)
    
def get_importances(model):
    if hasattr(model, "explained_variance_"):
        return model.explained_variance_
    elif hasattr(model, "eigenvalues_"):
        return model.eigenvalues_
    else:
        return None

def parallel_analysis_plot(X, model, n_iter=500):    
    rng = np.random.default_rng(42)
    n_samples, n_features = X.shape

    real_eigenvals = get_importances(model)
    if real_eigenvals is None: return

    n_components_model = len(real_eigenvals)
    rand_eigenvals = np.zeros((n_iter, n_components_model))
    model_rand = copy.deepcopy(model)

    for i in range(n_iter):
        X_rand = np.zeros_like(X.values)   
        for j in range(n_features):
            X_rand[:, j] = rng.permutation(X.values[:, j])
        model_rand.fit(X_rand)
        rand_eigenvals[i, :] = get_importances(model_rand)
        if rand_eigenvals[i, :] is None: return
    mean_rand_eigenvals = rand_eigenvals.mean(axis=0)
    n_components = np.sum(real_eigenvals > mean_rand_eigenvals)
    
    fig = plt.figure(figsize=(8,5))
    plt.plot(range(1, n_components_model+1), real_eigenvals, marker="o", label="Observed data")
    plt.plot(range(1, n_components_model+1), mean_rand_eigenvals, marker="x", label="Simulated data")
    plt.axhline(1, color="red", linestyle="--", label="Kaiser criterion")
    plt.title("Parallel Analysis Scree Plot")
    plt.xlabel("Component")
    plt.ylabel("Eigenvalue")
    plt.xticks(range(1, n_components_model+1))
    plt.legend()
    log_plot(fig)
    return n_components
    
def loadings_plot(X, model):
    if not hasattr(model, "components_"): return
    n_comp_actual = model.components_.shape[0]    
    loadings_df = pd.DataFrame(
        model.components_.T,
        columns=[f'F{i+1}' for i in range(n_comp_actual)],
        index=X.columns
    )
    
    fig = plt.figure(figsize=(1.5*n_comp_actual, 0.5*len(X.columns)))
    sns.heatmap(loadings_df, annot=True, cmap='RdBu', center=0)
    plt.title('Loadings (Feature Contributions)')
    log_plot(fig)
    return loadings_df

def biplot(X, model, pc1=1, pc2=2, top_k=3, c=None, feature_names=None):
    scores = X[:, [pc1-1, pc2-1]]

    importance = get_importances(model)
    if importance is None:
        importance = np.ones(model.components_.shape[0])

    scores_scaled = scores / np.sqrt(importance[[pc1-1, pc2-1]])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=scores_scaled[:, 0], y=scores_scaled[:, 1], alpha=0.6, hue=c, ax=ax)

    if hasattr(model, "components_"):
        loadings = model.components_.T
        loadings_scaled = loadings[:, [pc1-1, pc2-1]] * np.sqrt(importance[[pc1-1, pc2-1]])
        if top_k is not None:
            top_pc1 = np.argsort(np.abs(loadings[:, pc1-1]))[::-1][:top_k]
            top_pc2 = np.argsort(np.abs(loadings[:, pc2-1]))[::-1][:top_k]
            selected = np.unique(np.concatenate([top_pc1, top_pc2]))
        else:
            selected = range(loadings.shape[0])

        for i in selected:
            feature = feature_names[i] if feature_names is not None else str(i)
            ax.arrow(0, 0,
                     loadings_scaled[i, 0],
                     loadings_scaled[i, 1],
                     color='r', alpha=0.5, head_width=0.02)
            ax.text(loadings_scaled[i, 0] * 1.15,
                    loadings_scaled[i, 1] * 1.15,
                    feature, color='r', ha='center', va='center', fontsize=7)
    ax.set_xlabel(f"PC{pc1} (λ={importance[pc1-1]:.2f})")
    ax.set_ylabel(f"PC{pc2} (λ={importance[pc2-1]:.2f})")
    ax.grid(True)
    log_plot(fig)
    
def plot_network_2d(df, threshold=0.3, metric="cosine"):
    labels = df.iloc[:, 0].astype(str).values
    X = df.iloc[:, 1:3].values
    dist = pairwise_distances(X, metric=metric)
    
    G = nx.Graph()
    for label in labels:
        G.add_node(label)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if dist[i, j] <= threshold:
                G.add_edge(labels[i], labels[j], weight=dist[i, j])

    pos = {labels[i]: X[i, :2] for i in range(len(labels))} 
    components = list(nx.connected_components(G))
    palette = sns.color_palette("Set2", len(components))
    comp_color_map = {node: palette[i] for i, comp in enumerate(components) for node in comp}
    node_colors = [comp_color_map[node] for node in G.nodes()]

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(
        G, pos, edgelist=G.edges(data=True),
        width=[(1 - d['weight']) * 3 for (_, _, d) in G.edges(data=True)],
        alpha=0.6, edge_color="gray"
    )
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors,
        node_size=800, edgecolors="black"
    )
    nx.draw_networkx_labels(
        G, pos, font_size=8, font_weight="bold",
        verticalalignment="center", horizontalalignment="center",
        zorder=3  
    )
    ax.margins(0.2)
    ax.set_title(f"{df.columns[0]} Network Graph ({metric} distance ≤ {threshold:.2f})")
    ax.axis("off")
    log_plot(fig)

def plot_network_3d(df, threshold=0.3, metric="cosine"):
    labels = df.iloc[:, 0].astype(str).values
    X = df.iloc[:, 1:4].values
    dist = pairwise_distances(X, metric=metric)

    G = nx.Graph()
    for label in labels:
        G.add_node(label)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if dist[i, j] <= threshold:
                G.add_edge(labels[i], labels[j], weight=dist[i, j])

    pos3d = {labels[i]: X[i] for i in range(len(labels))}
    components = list(nx.connected_components(G))
    palette = sns.color_palette("Set2", len(components))
    comp_color_map = {node: palette[i] for i, comp in enumerate(components) for node in comp}
    node_colors = [comp_color_map[node] for node in G.nodes()]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    for u, v, d in G.edges(data=True):
        x = [pos3d[u][0], pos3d[v][0]]
        y = [pos3d[u][1], pos3d[v][1]]
        z = [pos3d[u][2], pos3d[v][2]]
        ax.plot(x, y, z, color="gray", alpha=0.6, linewidth=(1 - d['weight']) * 2, zorder=1)
    for i, node in enumerate(G.nodes()):
        x, y, z = pos3d[node]
        ax.scatter(x, y, z, color=node_colors[i], s=100, edgecolors="black", zorder=2)
        ax.text(
            x, y, z, node,
            fontsize=7, fontweight="bold",
            zorder=3,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)  
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(f"{df.columns[0]} Network Graph ({metric} distance ≤ {threshold:.2f})")
    log_plot(fig)

def get_clusters(df, threshold=0.3, metric="cosine"):
    labels = df.iloc[:, 0].astype(str).values
    X = df.iloc[:, 1:].values

    dist = pairwise_distances(X, metric=metric)
    adjacency = (dist <= threshold).astype(int)

    graph = csr_matrix(adjacency)
    n_components, component_labels = connected_components(csgraph=graph, directed=False)

    clusters = [labels[component_labels == i].tolist() for i in range(n_components)]
    return clusters

class Custom_Decomposition: 
    def __init__(self, df, features, model, n_components=None):
        self.df = df #assumes already normalized
        self.features = features #assumes only numerical
        self.model = model
        self.model.random_state=42
        self.set_n_components(n_components)
        
    def get_model(self):
        return self.model
    def set_features(self, features):
        self.features = features
        return self
    def get_features(self):
        return self.features
    def set_n_components(self, n_components):
        self.model.n_components = n_components
        print(f"[MODEL]n_components set to {n_components}")
        return self

    def parallel_analysis(self, features=None):
        if features is None: features = self.features
        model = self.model
        X = self.df[features]
        model.fit(X)
        
        n_components = parallel_analysis_plot(X, model)
        pca_scree_plot(X, model)
        fa_noise_plot(X, model)
        self.set_n_components(n_components)
        return self
    
    def loadings(self, features=None):
        if features is None: features = self.features
        model = self.model
        X = self.df[features]
        model.fit(X)
        
        loadings_df = loadings_plot(X, model)
        return loadings_df

    def biplot(self, pc1=1, pc2=2, top_k=3, features=None, colour=None):
        if features is None: features = self.features
        model = self.model
        X = self.df[features]
        c = self.df[colour] if colour is not None else None
        X_transformed = model.fit_transform(X)
        
        biplot(X_transformed, model, pc1, pc2, top_k, c=c, feature_names=features)
        return self

    def fit_transform(self, features=None):
        if features is None: features = self.features
        model = self.model
        X = self.df[features]
        X_transformed = model.fit_transform(X)
        X_df = pd.DataFrame(X_transformed, columns=[f"X{i+1}" for i in range(X_transformed.shape[1])])            
        return X_df

    def transform(self, features=None):
        if features is None:
            features = self.features
        model = self.model
        X = self.df[features]
        X_transformed = model.transform(X)
        X_df = pd.DataFrame(X_transformed, columns=[f"X{i+1}" for i in range(X_transformed.shape[1])])
        return X_df

    def network_plot(self, n_dim, features=None, threshold=0.3, metric="cosine", label_df=None, labels=None):
        if features is None: features = self.features
        n_dim = min(n_dim, 3)

        X_df = self.set_n_components(n_dim).fit_transform(features)
        
        if label_df is None or label_df.height != X_df.height:
            label_df = pd.DataFrame({"label": range(len(X_df))})
        df = pd.concat([label_df.reset_index(drop=True), X_df], axis=1)
        if labels is not None:
            df = df[df.iloc[:,0].isin(names)].reset_index(drop=True)

        if n_dim >0 and <= 2:
            plot_network_2d(df, threshold)
        elif n_dim == 3:
            plot_network_3d(df, threshold)
        return get_clusters(df, threshold=threshold, metric=metric)

    def clusters(self, n_dim, features=None, threshold=0.3, metric="cosine", label_df=None):
        if features is None: features = self.features

        X_df = self.set_n_components(n_dim).fit_transform(features)
        if label_df is None or label_df.height != X_df.height:
            label_df = pd.DataFrame({"label": range(len(X_df))})
        df = pd.concat([label_df.reset_index(drop=True), X_df], axis=1)

        clusters = get_clusters(df, threshold=threshold, metric=metric)
        print(f"\n{df.columns[0].title()}\n")
        for i, c in enumerate(clusters):
            print(f"Cluster{i}: {c}")
        return clusters
        
    




