# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import streamlit as st
import joblib
import plotly.graph_objects as go
from fpdf import FPDF
import time

# %%
big5_df = pd.read_excel('Job-profile.xlsx', sheet_name='Big Five Domains')

features = big5_df[['Neuroticism (M)', 'Extraversion (M)', 
                    'Openness (M)', 'Agreeableness (M)', 
                    'Conscientiousness (M)']]
job_names = big5_df['Job'].tolist()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# %%
big5_df

# %%

def augment_data(features, noise_ratio, n_samples_per_job):
    noise_scale = np.std(features, axis=0) * noise_ratio

    X_augmented = []
    y_augmented = []

    for job_idx, original_feature in enumerate(features):
        samples = original_feature + np.random.normal(scale=noise_scale, size=(n_samples_per_job, features.shape[1]))
        X_augmented.append(samples)
        y_augmented.extend([job_idx] * n_samples_per_job)

    X = np.vstack(X_augmented)
    y = np.array(y_augmented)
    return X, y

class JobRecommenderMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(JobRecommenderMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)


X_aug, y_aug = augment_data(scaled_features, noise_ratio=0.1, n_samples_per_job=1000)


X_train, X_val, y_train, y_val = train_test_split(X_aug, y_aug, test_size=0.25, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

input_dim = X_aug.shape[1]
hidden_dim = 128
output_dim = len(job_names)
model = JobRecommenderMLP(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
patience = 3
best_val_loss = float('inf')
wait = 0

train_loss_list, val_loss_list = [], []
train_acc_list, val_acc_list = [], []
train_f1_list, val_f1_list = [], []

for epoch in range(n_epochs):
    model.train()
    train_preds, train_targets = [], []
    total_train_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        pred_labels = torch.argmax(output, dim=1).detach().cpu().numpy()
        train_preds.extend(pred_labels)
        train_targets.extend(batch_y.detach().cpu().numpy())

    avg_train_loss = total_train_loss / len(train_loader)
    train_acc = accuracy_score(train_targets, train_preds)
    train_f1 = f1_score(train_targets, train_preds, average='macro')

    model.eval()
    val_preds, val_targets = [], []
    total_val_loss = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            output = model(batch_X)
            loss = criterion(output, batch_y)
            total_val_loss += loss.item()

            pred_labels = torch.argmax(output, dim=1).detach().cpu().numpy()
            val_preds.extend(pred_labels)
            val_targets.extend(batch_y.detach().cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds, average='macro')

    train_loss_list.append(avg_train_loss)
    val_loss_list.append(avg_val_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_f1_list.append(train_f1)
    val_f1_list.append(val_f1)

    print(f"Epoch {epoch+1}/{n_epochs} | "
          f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | "
          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
          f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

model.load_state_dict(best_model_state)










# %%
# Loss
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy
plt.subplot(1,3,2)
plt.plot(train_acc_list, label='Train Acc')
plt.plot(val_acc_list, label='Val Acc')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# F1
plt.subplot(1,3,3)
plt.plot(train_f1_list, label='Train F1')
plt.plot(val_f1_list, label='Val F1')
plt.title("F1 Score Curve")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()

plt.tight_layout()
plt.show()


# %%
# %%
job_codes = big5_df['Code'].tolist()

from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(scaled_features)
# similarity_matrix.shape = (263, 263)



def compute_pca_weights(features):
    pca = PCA(n_components=features.shape[1])
    pca.fit(features)
    weights = pca.explained_variance_ratio_
    return weights

def compute_weighted_euclidean_similarity(user_big5, job_features, weights):
    user_df = pd.DataFrame([user_big5], columns=features.columns)
    user_scaled = scaler.transform(user_df)[0]
    diffs = job_features - user_scaled
    weighted_dists = np.sqrt(np.sum(weights * (diffs ** 2), axis=1))

    # 归一化距离为相似度（距离越小相似度越高）
    dist_min, dist_max = weighted_dists.min(), weighted_dists.max()
    normalized = (weighted_dists - dist_min) / (dist_max - dist_min + 1e-10)
    similarities = 1 - normalized
    return similarities

def recommend_jobs_weighted_euclidean(user_big5_scores, model, job_features, weights, top_k=10):
    # 加权欧几里得距离相似度
    similarities = compute_weighted_euclidean_similarity(user_big5_scores, job_features, weights)

    # MLP 输出 logits
    user_df = pd.DataFrame([user_big5_scores], columns=features.columns)
    user_scaled = scaler.transform(user_df)
    user_tensor = torch.tensor(user_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = model(user_tensor).numpy().flatten()

    match_scores = similarities * logits
    top_indices = np.argsort(match_scores)[-top_k:][::-1]
    top_jobs = [(job_codes[i], job_names[i], match_scores[i]) for i in top_indices]
    return top_jobs


original_features = big5_df[['Neuroticism (M)', 'Extraversion (M)', 
                             'Openness (M)', 'Agreeableness (M)', 
                             'Conscientiousness (M)']].values

pca_weights = compute_pca_weights(original_features)

user_input = [51.46,	51.17,	46.09,	52.15,	49.41]


recommendations = recommend_jobs_weighted_euclidean(user_input, model, scaled_features, pca_weights, top_k=10)

for i, (code, job, score) in enumerate(recommendations):
    print(f"{i+1}. {code} - {job} (score: {score:.4f})")



def evaluate_precision_recall_f1(model, X_val, y_val, job_features, weights, top_k=10):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_users = len(X_val)

    for i in range(num_users):
        user_input = scaler.inverse_transform([X_val[i]])[0]  # 还原为原始人格
        true_label = y_val[i]

        recommendations = recommend_jobs_weighted_euclidean(
            user_input, model, job_features, weights, top_k=top_k
        )
        recommended_indices = [job_codes.index(code) for code, _, _ in recommendations]

        # 判断 TP, FP, FN
        if true_label in recommended_indices:
            TP = 1
            FP = top_k - 1
            FN = 0
        else:
            TP = 0
            FP = top_k
            FN = 1

        precision = TP / (TP + FP)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    avg_precision = total_precision / num_users
    avg_recall = total_recall / num_users
    avg_f1 = total_f1 / num_users

    print(f"Precision@{top_k}: {avg_precision:.4f}")
    print(f"Recall@{top_k}: {avg_recall:.4f}")
    print(f"F1-score@{top_k}: {avg_f1:.4f}")

    return avg_precision, avg_recall, avg_f1



evaluate_precision_recall_f1(
    model=model,
    X_val=X_val,
    y_val=y_val,
    job_features=scaled_features,
    weights=pca_weights,  
    top_k=10
)




# %%

torch.save(model.state_dict(), "your_model.pth")


scaler = StandardScaler()
scaler.fit(X_train)  # 这里 X_train 是你训练数据的特征
joblib.dump(scaler, "your_scaler.pkl")



job_names = big5_df['Job'].tolist() 
job_codes = big5_df['Code'].tolist()


np.save("job_names.npy", job_names)
np.save("job_codes.npy", job_codes)



np.save("scaled_features.npy", scaled_features)


np.save("similarity_matrix.npy", similarity_matrix)







