import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.model.results import plot_learning_curve
from src.feature.decomposition import Custom_Decomposition


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_mapping(train_df, cat_cols): 
    mappings = {}
    unk_indices = {}
    for col in cat_cols:
        cats = train_df[col].unique()
        mapping = {cat: i for i, cat in enumerate(cats)}
        unk_index = len(mapping)
        mappings[col] = mapping
        unk_indices[col] = unk_index
    return mappings, unk_indices

def apply_mapping(df, mappings, unk_indices):
    new_cols = pd.DataFrame({
        f"{col}_idx": df[col].map(lambda x: mappings[col].get(x, unk_indices[col]))
        for col in mappings
    })
    df = pd.concat([df, new_cols], axis=1)
    return df
        
class EmbeddingModel(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_dim):
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(cat_dim+1, emb_dim)
            for cat_dim, emb_dim in zip(cat_dims, emb_dims) 
        ])
        self.emb_dropout = nn.Dropout(p=0.3)
        
        total_input_dim = sum(emb_dims) + num_dim
        #self.fc = nn.Linear(total_input_dim, 1)        
        self.fc = nn.Sequential(
            nn.Linear(total_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1)
        )
    def forward(self, cats, nums):
        cat_embs = [emb(cat) for emb, cat in zip(self.emb_layers, cats)]
        cat_embs = [self.emb_dropout(emb) for emb in cat_embs]
        x = torch.cat(cat_embs+[nums], dim=1)
        return self.fc(x)
        
def train_embedding(df, feat, params=None):
    if params is None:
        params = {"lr": 0.01, "loss": nn.MSELoss(), "epochs": 5}

    nums = feat["num"]
    cat_dims = [df[cat].nunique() for cat in feat["cat"]]
    emb_dims = [min(max(1, int(np.sqrt(n_cat))), 50) for n_cat in cat_dims]
    num_dim = len(nums)
    
    model = EmbeddingModel(cat_dims, emb_dims, num_dim)
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = params["loss"]

    evals_result = {"train": {"loss": []}}
    for epoch in range(params["epochs"]):
        cat_tensors = [torch.tensor(df[f"{cat}_idx"].values, dtype=torch.long) for cat in feat["cat"]]
        num_tensor = torch.tensor(df[nums].values, dtype=torch.float32)
        y_tensor = torch.tensor(df[feat["target"]].values, dtype=torch.float32).view(-1, 1)

        optimizer.zero_grad()
        preds = model(cat_tensors, num_tensor) #runs a forward pass
        loss = loss_fn(preds, y_tensor) 
        loss.backward() 
        optimizer.step() #updates model parameters based on gradient of loss (𝜃←𝜃−𝜂⋅∇𝜃𝐿) where 𝜂 is lr
        evals_result["train"]["loss"].append(loss.item())
    return model, evals_result

def transform_embedding(df, model, feat):
    df_out = df.copy()
    for i, cat in enumerate(feat["cat"]):
        idxs = torch.tensor(df[f"{cat}_idx"].values, dtype=torch.long)
        emb_matrix = model.emb_layers[i].weight.detach().cpu().numpy()
        emb_vals = emb_matrix[idxs.numpy()]  #(n_samples, emb_dim)
        for j in range(emb_vals.shape[1]):
            df_out[f"{cat}_emb{j}"] = emb_vals[:, j]
        df_out = df_out.drop(columns=[f"{cat}_idx"])
    return df_out

def lookup_embeddings(df, model, feat):
    lookup_dict = {}
    for i, cat in enumerate(feat["cat"]):
        emb_matrix = model.emb_layers[i].weight.detach().cpu().numpy() #(cat_dim, emb_dim)
        n_cat = emb_matrix.shape[0]

        idx2val = df[[cat, f"{cat}_idx"]].drop_duplicates().set_index(f"{cat}_idx")[cat] 
        idx2val = idx2val.reindex(range(n_cat)) #get cat names + NaN

        emb_df = pd.DataFrame(
            emb_matrix,
            columns=[f"{cat}_emb{j}" for j in range(emb_matrix.shape[1])]
        )
        emb_df.insert(0, cat, idx2val.values) #insert cat names
        lookup_dict[cat] = emb_df
    return lookup_dict 

def learned_embedding(datasets, feat, params=None):
    set_seed(42)
    mappings, unk_indices = build_mapping(datasets[0], feat["cat"])
    datasets = [apply_mapping(d, mappings, unk_indices) for d in datasets]
    
    model, evals_result = train_embedding(datasets[0], feat, params)
    #plot_learning_curve("learned_embedding", evals_result, metrics=["loss"], SHOW=True)
    
    datasets_embedded = []
    for d in datasets:
        df_out = transform_embedding(d, model, feat)
        datasets_embedded.append(df_out)
    lookup_dict = lookup_embeddings(datasets[0], model, feat)
    return datasets_embedded, lookup_dict

def visualise_embeddings(lookup_dict, model, n_dim=3, threshold=0.3, metric="cosine", POOL=False):
    pooled_df = pd.DataFrame()
    clusters = None
    for key, emb_df in lookup_dict.items(): #emb_df.columns = [label_key, emb_0, ..., emb_n]
        if emb_df.shape[1]-1 < n_dim: #num_PC > emb_dim
            continue
        pca = Custom_Decomposition(emb_df, emb_df.columns[1:], model=model, n_components=None)
        if POOL:
            X_df = pca.set_n_components(n_dim).fit_transform()
            X_df.rename(columns={X_df.columns[0]: "label"}, inplace=True)
            pooled_df = pd.concat([pooled_df, X_df], axis=0, ignore_index=True)
        else:
            clusters = pca.network_plot(n_dim=n_dim, threshold=threshold, metric=metric)
    if POOL and not pooled_df.empty:
        clusters = pca.network_plot(n_dim=n_dim, label_df=pooled_df, threshold=threshold, metric=metric)
    return clusters






    
