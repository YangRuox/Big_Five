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

user_input = [51.46,	51.17,	46.09,	52.15,	49.41]

recommendations = recommend_jobs(user_input, model, similarity_matrix, top_k=10)

for i, (code, job, score) in enumerate(recommendations):
    print(f"{i+1}. {code} - {job} (score: {score:.2f})")


# %%
# 假设你的模型是 model
torch.save(model.state_dict(), "your_model.pth")


# %%
# 假设你已经对特征使用了 scaler
scaler = StandardScaler()
scaler.fit(X_train)  # 这里 X_train 是你训练数据的特征
joblib.dump(scaler, "your_scaler.pkl")


# %%
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


# 保存相似度矩阵
np.save("similarity_matrix.npy", similarity_matrix)


# %%



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


@st.cache_data
def load_data():
    mean_norms = pd.read_csv('meanNorms.tsv', sep='\t')
    sd_norms = pd.read_csv('sdNorms.tsv', sep='\t')
    questions = pd.read_csv('questions.tsv', sep='\t')
    weights = pd.read_csv('weightsB5.tsv', sep='\t')
    return mean_norms, sd_norms, questions, weights
   
mean_norms, sd_norms, questions, weights = load_data()

# 74道题 
items = list(questions['en'])

response_dict = {}

with st.form("bfi_form"):
   # 显示表单
   st.title("🔍 Big Five Personality Test + Career Recommender")
   st.markdown("Please rate the following statements based on your true feelings: **1 (Strongly Disagree) to 5 (Strongly Agree)**")
  
   # 性别年龄选择
   gender = st.selectbox("Select your gender:", ["Female", "Male"])
   age = st.number_input("Enter your age:", min_value=18, max_value=70, value=25)
   
   if age < 18 or age > 70:
       st.warning("Sorry, your age does not meet the requirements.")
       st.stop()  # 提交表单之前停止执行

   if "age" not in st.session_state:
        st.session_state.age = 25  # 默认年龄

   if "gender" not in st.session_state:
        st.session_state.gender = "Female"  # 默认性别
    
   st.subheader("👇 Please fill in your questionnaire answers")

    # 问题的滑动条
    for i, q in enumerate(questions["en"]):
        key = f"q{i}"
        response_dict[key] = st.slider(
            q,
            min_value=1, max_value=6,
            value=st.session_state.get(key, 3),
            key=key
        )

    # 检查是否完成所有问题
    if all(v is not None for v in response_dict.values()):
        submitted = st.form_submit_button("🎯 Submit and Recommend Careers")
    else:
        submitted = False
        st.warning("Please answer all questions before submitting.")  # 提示用户回答所有问题

if submitted:
    st.session_state.age = age
    st.session_state.gender = gender

   # 分组
    if gender == "Female":
        normgroup = 1 if age < 35 else 2
    else:
        normgroup = 3 if age < 35 else 4

    progress = st.progress(0, text="⏳ Processing...")
    for i in range(60):
        time.sleep(0.005)
        progress.progress(i + 1)
 
    # Step 1: 获取 norm μ 和 σ
    mu = mean_norms[mean_norms['group'] == normgroup].iloc[0, 1:].values  # 跳过 group 列
    sigma = sd_norms[sd_norms['group'] == normgroup].iloc[0, 1:].values

    # Step 2: 用户回答转 numpy
    responses = np.array([response_dict[f"q{i}"] for i in range(len(questions))])

    # Step 3: 计算 Z 
    Z = (responses - mu) / sigma

    # Step 4: 加权求 Big Five 得分（weightsB5 为 74x5，T 为 74x1，输出为 5x1）
    big5_scores = np.dot(Z, weights.values)  # shape: (5,)
    T_scores = 10 * big5_scores + 50 

    # Step 5: 标准化（用你的 scaler）
    scaled_input = scaler.transform([T_scores])

    trait_names = ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]
    # 闭合雷达图数据（起点和终点一致）
    radar_values = list(T_scores) + [T_scores[0]]
    radar_labels = trait_names + [trait_names[0]]
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=radar_labels,
        fill='toself',
        name='Your Big Five T Scores',
        line=dict(color='royalblue')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
        ),
        showlegend=False,
        title="🧬 Your Big Five Personality Profile (T scores)"
    )

    st.plotly_chart(fig)
    

    # Step 6: 模型预测
    with torch.no_grad():
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        logits = model(input_tensor).numpy().flatten()
        scores = similarity_matrix @ logits
        top_indices = np.argsort(scores)[-10:][::-1]  # 前10个推荐
        bottom_indices = np.argsort(scores)[:10]    # 最不推荐的10个

        st.subheader("🧠 Recommended Careers Top-10")
        for rank, idx in enumerate(top_indices, 1):
            st.write(f"NO.{rank} - {job_names[idx]}")

        st.subheader("😬 Least Recommended Careers Bottom-10")
        for rank, idx in enumerate(bottom_indices, 1):
            st.write(f"NO.{rank} - {job_names[idx]}")

        # 🌟 生成 PDF
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)

    # 添加标题
    pdf.cell(200, 10, txt="Big Five Personality Test Results", ln=True, align='C')

    # 个人信息
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)

    # Big Five Scores
    pdf.ln(10)
    pdf.cell(200, 10, txt="Big Five Personality Scores (T scores):", ln=True)
    for trait, score in zip(trait_names, T_scores):
        pdf.cell(200, 10, txt=f"{trait}: {score:.2f}", ln=True)

    # 推荐职业
    pdf.ln(10)
    pdf.cell(200, 10, txt="Recommended Careers Top-10:", ln=True)
    for rank, idx in enumerate(top_indices, 1):
        pdf.cell(200, 10, txt=f"{rank}. {job_names[idx]}", ln=True)

    # 最不推荐职业
    pdf.ln(10)
    pdf.cell(200, 10, txt="Least Recommended Careers Bottom-10:", ln=True)
    for rank, idx in enumerate(bottom_indices, 1):
        pdf.cell(200, 10, txt=f"{rank}. {job_names[idx]}", ln=True)

    # 保存 PDF 文件
    pdf_output = "BigFive_Test_Result.pdf"
    pdf.output(pdf_output)

    # 提供下载链接
    with open(pdf_output, "rb") as f:
        st.download_button("Download Your PDF Report", f, file_name=pdf_output)










   

# %%


# %%


# %%


# %%



