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


def recommend_jobs(user_big5_scores, model, similarity_matrix, top_k=10):
    model.eval()
    with torch.no_grad():
        user_scaled = scaler.transform([user_big5_scores])
        user_tensor = torch.tensor(user_scaled, dtype=torch.float32)

        logits = model(user_tensor).numpy().flatten()

        match_score = similarity_matrix @ logits

        top_indices = np.argsort(match_score)[-top_k:][::-1]
        top_jobs = [(job_codes[i], job_names[i], match_score[i]) for i in top_indices] 

        return top_jobs

user_input = [51.46,	51.17,	46.09,	52.15,	49.41]

recommendations = recommend_jobs(user_input, model, similarity_matrix, top_k=10)

for i, (code, job, score) in enumerate(recommendations):
    print(f"{i+1}. {code} - {job} (score: {score:.2f})")

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







