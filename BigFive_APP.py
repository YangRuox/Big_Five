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
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# %%

   # 读取Excel文件
big5_df = pd.read_excel('Job-profile.xlsx', sheet_name='Big Five Domains')

# 提取五大特质特征
features = big5_df[['Neuroticism (M)', 'Extraversion (M)', 
                    'Openness (M)', 'Agreeableness (M)', 
                    'Conscientiousness (M)']]
job_names = big5_df['Job'].tolist()

# 标准化处理
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


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


# %%
# 数据增强
X_aug, y_aug = augment_data(scaled_features, noise_ratio=0.1, n_samples_per_job=1000)



# 数据划分
X_train, X_val, y_train, y_val = train_test_split(X_aug, y_aug, test_size=0.25, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 模型定义（你已有）
input_dim = X_aug.shape[1]
hidden_dim = 128
output_dim = len(job_names)
model = JobRecommenderMLP(input_dim, hidden_dim, output_dim)

# 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练参数
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

    # 验证阶段
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

    # 保存记录
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

# 加载最佳模型
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
job_codes = big5_df['Code'].tolist()

from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(scaled_features)
# similarity_matrix.shape = (263, 263)


def recommend_jobs(user_big5_scores, model, similarity_matrix, top_k=10):
    model.eval()
    with torch.no_grad():
        # 标准化用户输入
        user_scaled = scaler.transform([user_big5_scores])
        user_tensor = torch.tensor(user_scaled, dtype=torch.float32)

        # MLP 输出 logits
        logits = model(user_tensor).numpy().flatten()

        # 计算 similarity-aware score（逻辑输出 × 相似度）
        match_score = similarity_matrix @ logits

        # 取 Top-k
        top_indices = np.argsort(match_score)[-top_k:][::-1]
        top_jobs = [(job_codes[i], job_names[i], match_score[i]) for i in top_indices]  # 加上代码

        return top_jobs

# %%
# 示例人格分数（原始 Big Five T 分数）
user_input = [51.46,	51.17,	46.09,	52.15,	49.41]

recommendations = recommend_jobs(user_input, model, similarity_matrix, top_k=10)

for i, (code, job, score) in enumerate(recommendations):
    print(f"{i+1}. {code} - {job} (score: {score:.2f})")

# %%
import torch

# 假设你的模型是 model
torch.save(model.state_dict(), "your_model.pth")


# %%
import joblib
from sklearn.preprocessing import StandardScaler

# 假设你已经对特征使用了 scaler
scaler = StandardScaler()
scaler.fit(X_train)  # 这里 X_train 是你训练数据的特征
joblib.dump(scaler, "your_scaler.pkl")


# %%
import numpy as np

# 假设 job_names 和 job_codes 是你拥有的职业名称和代码列表
job_names = big5_df['Job'].tolist() # 你的职业名称列表
job_codes = big5_df['Code'].tolist() # 你的职业代码列表

# 保存为 numpy 文件
np.save("job_names.npy", job_names)
np.save("job_codes.npy", job_codes)


# %%
# 假设 scaled_features 是你的职业特征经过标准化后的数据
 # 假设 job_features 是你未标准化的职业特征

# 保存标准化后的职业特征
np.save("scaled_features.npy", scaled_features)


# %%
from sklearn.metrics.pairwise import cosine_similarity


# 保存相似度矩阵
np.save("similarity_matrix.npy", similarity_matrix)


# %%
import torch
import torch.nn as nn
import numpy as np
from joblib import load
import streamlit as st


# 定义你的模型架构
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

# 加载训练好的模型和数据
model = JobRecommenderMLP(input_dim=5, hidden_dim=128, output_dim=263)  # 调整为你的模型结构
model.load_state_dict(torch.load("your_model.pth", map_location=torch.device("cpu")))
model.eval()  # 切换到评估模式

# 加载其他资源
scaler = load("your_scaler.pkl")  # 你训练模型时使用的 scaler
job_names = np.load("job_names.npy")  # 263个职业名称
job_codes = np.load("job_codes.npy")  # 对应职业代码
scaled_features = np.load("scaled_features.npy")  # 原始职业特征（用于相似度）
similarity_matrix = np.load("similarity_matrix.npy")

# 44道题 + 每题所属维度 + 正反向
items = [
    # (题目, 维度名, 是否反向)
    ("I am the life of the party.", "Extraversion", False),
    ("I don't talk a lot.", "Extraversion", True),
    ("I feel comfortable around people.", "Extraversion", False),
    ("I keep in the background.", "Extraversion", True),
    ("I start conversations.", "Extraversion", False),
    ("I have little to say.", "Extraversion", True),
    ("I talk to a lot of different people at parties.", "Extraversion", False),
    ("I don't like to draw attention to myself.", "Extraversion", True),
    ("I am quiet around strangers.", "Extraversion", True),
    
    ("I enjoy trying new things.", "Openness", False),
    ("I have a rich vocabulary.", "Openness", False),
    ("I have a vivid imagination.", "Openness", False),
    ("I am quick to understand things.", "Openness", False),
    ("I spend time reflecting on things.", "Openness", False),
    ("I am not interested in abstract ideas.", "Openness", True),
    ("I do not like art.", "Openness", True),
    ("I have difficulty understanding abstract ideas.", "Openness", True),
    ("I have a lot of artistic interests.", "Openness", False),
    
    ("I am easily disturbed.", "Neuroticism", False),
    ("I get upset easily.", "Neuroticism", False),
    ("I change my mood a lot.", "Neuroticism", False),
    ("I get nervous easily.", "Neuroticism", False),
    ("I worry about things.", "Neuroticism", False),
    ("I often feel blue.", "Neuroticism", False),
    ("I am relaxed most of the time.", "Neuroticism", True),
    ("I rarely feel depressed.", "Neuroticism", True),
    ("I am easily embarrassed.", "Neuroticism", False),
    
    ("I feel little concern for others.", "Agreeableness", True),
    ("I am not interested in other people's problems.", "Agreeableness", True),
    ("I insult people.", "Agreeableness", True),
    ("I sympathize with others' feelings.", "Agreeableness", False),
    ("I take time out for others.", "Agreeableness", False),
    ("I am not really interested in others.", "Agreeableness", True),
    ("I feel others' emotions.", "Agreeableness", False),
    ("I make people feel at ease.", "Agreeableness", False),
    
    ("I am often preoccupied with details.", "Conscientiousness", False),
    ("I follow a schedule.", "Conscientiousness", False),
    ("I am exacting in my work.", "Conscientiousness", False),
    ("I am always prepared.", "Conscientiousness", False),
    ("I leave my belongings around.", "Conscientiousness", True),
    ("I pay attention to details.", "Conscientiousness", False),
    ("I make plans and stick to them.", "Conscientiousness", False),
    ("I get chores done right away.", "Conscientiousness", False),
]

# 显示表单
st.title("🔍 Big Five Personality Test + Career Recommender")

st.markdown("Please rate the following statements based on your true feelings: **1 (Strongly Disagree) to 5 (Strongly Agree)**")

response_dict = {}
# 用 st.form 包裹所有问题
with st.form("bfi_form"):
    st.subheader("👇 Please fill in your questionnaire answers")
    
    for i, (q, trait, reverse) in enumerate(personality_questions):
        key = f"q{i}"  # session_state 中的 key

        # 如果该题没有值，默认设为 3 分（中性）
        if key not in st.session_state:
            st.session_state[key] = 3
        
        st.session_state[key] = st.slider(
            f"{i+1}. {q}",
            min_value=1, max_value=5, value=st.session_state[key],
            key=key
        )

    # 提交按钮放在 form 内部
    submitted = st.form_submit_button("🎯 Submit and Recommend Careers")


if submitted:
    # 收集所有 slider 的值
    trait_scores = {"Extraversion": [], "Openness": [], "Neuroticism": [], "Agreeableness": [], "Conscientiousness": []}
    
    for i, (_, trait, is_reverse) in enumerate(personality_questions):
        score = st.session_state[f"q{i}"]
        if is_reverse:
            score = 6 - score
        trait_scores[trait].append(score)
    
    # 每个维度取平均
    final_scores = []
    for trait in ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]:
        final_scores.append(np.mean(trait_scores[trait]))

    # 标准化 + 模型预测
    scaled_input = scaler.transform([final_scores])
    with torch.no_grad():
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        logits = model(input_tensor).numpy().flatten()
        scores = similarity_matrix @ logits

        top_indices = np.argsort(scores)[-10:][::-1]
        st.subheader("🧠 Recommended Careers Top-10")
        for i in top_indices:
            st.write(f"{job_codes[i]} - {job_names[i]}  (Similarity Score: {scores[i]:.3f})")


# %%


# %%


# %%


# %%



