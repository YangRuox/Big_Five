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

# %%
job_names = np.load("job_names.npy", allow_pickle=True)
job_codes = np.load("job_codes.npy", allow_pickle=True)

scaler = joblib.load("your_scaler.pkl")  
similarity_matrix = np.load("similarity_matrix.npy")

# %%
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


# %%
model = JobRecommenderMLP(input_dim=5, hidden_dim=128, output_dim=len(job_names))
model.load_state_dict(torch.load("your_model.pth"))
model.eval()



# %%
@st.cache_data
def load_data():
    mean_norms = pd.read_csv('meanNorms.tsv', sep='\t')
    sd_norms = pd.read_csv('sdNorms.tsv', sep='\t')
    questions = pd.read_csv('questions.tsv', sep='\t')
    weights = pd.read_csv('weightsB5.tsv', sep='\t')
    return mean_norms, sd_norms, questions, weights

mean_norms, sd_norms, questions, weights = load_data()

language_display = {
    'ar': 'العربية',
    'en': 'English',
    'es': 'Español',
    'fr': 'Français',   
    'ru': 'Русский',
    'zh': '中文'
}
text_dict = {
    "en": [
        "🔍 Big Five Personality Test + Career Recommender",
        "Please rate the following statements based on your true feelings: **1 (Strongly Disagree) to 6 (Strongly Agree)**",
        "Select your gender:",
        "Enter your age:",
        "Sorry, your age does not meet the requirements.",
        "25",  # Default age
        "Female",  # Default gender
        "👇 Please fill in your questionnaire answers",
        "Please answer all questions before submitting."
    ],
    "fr": [
        "🔍 Test de personnalité Big Five + Recommandation de carrière",
        "Veuillez évaluer les déclarations suivantes en fonction de vos véritables sentiments : **1 (Pas du tout d'accord) à 6 (Tout à fait d'accord)**",
        "Sélectionnez votre sexe :",
        "Entrez votre âge :",
        "Désolé, votre âge ne répond pas aux exigences.",
        "25",  # Âge par défaut
        "Femme",  # Sexe par défaut
        "👇 Veuillez remplir vos réponses au questionnaire",
        "Veuillez répondre à toutes les questions avant de soumettre."
    ],
    "es": [
        "🔍 Test de personalidad Big Five + Recomendación de carrera",
        "Por favor, califique las siguientes afirmaciones según sus verdaderos sentimientos: **1 (Totalmente en desacuerdo) a 6 (Totalmente de acuerdo)**",
        "Seleccione su género:",
        "Ingrese su edad:",
        "Lo siento, su edad no cumple con los requisitos.",
        "25",  # Edad por defecto
        "Femenino",  # Género por defecto
        "👇 Por favor, complete sus respuestas al cuestionario",
        "Por favor, responda todas las preguntas antes de enviar."
    ],
    "ar": [
        "🔍 اختبار الشخصية Big Five + توصية المهن",
        "يرجى تقييم العبارات التالية بناءً على مشاعرك الحقيقية: **1 (لا أوافق بشدة) إلى 6 (أوافق بشدة)**",
        "حدد جنسك:",
        "أدخل عمرك:",
        "عذرًا، عمرك لا يتوافق مع المتطلبات.",
        "25",  # العمر الافتراضي
        "أنثى",  # الجنس الافتراضي
        "👇 يرجى ملء إجاباتك على الاستبيان",
        "يرجى الإجابة على جميع الأسئلة قبل الإرسال."
    ],
    "ru": [
        "🔍 Тест личности Big Five + Рекомендатор профессий",
        "Пожалуйста, оцените следующие утверждения, основываясь на ваших истинных чувствах: **1 (Совсем не согласен) до 6 (Полностью согласен)**",
        "Выберите ваш пол:",
        "Введите ваш возраст:",
        "Извините, ваш возраст не соответствует требованиям.",
        "25",  # По умолчанию
        "Женский",  # По умолчанию
        "👇 Пожалуйста, заполните свои ответы на вопросы",
        "Пожалуйста, ответьте на все вопросы перед отправкой."
    ],
    "zh": [
        "🔍 五大人格测试 + 职业推荐器",
        "请根据您的真实感受对以下陈述进行评分：**1（非常不同意）到6（非常同意）**",
        "选择您的性别：",
        "请输入您的年龄：",
        "抱歉，您的年龄不符合要求。",
        "25",  # 默认年龄
        "女性",  # 默认性别
        "👇 请填写您的问卷答案",
        "请在提交前回答所有问题。"
    ]
}


language_options = list(language_display.values())

col1, col2 = st.columns(2)
with col1:
    selected_language_name = st.selectbox("Select your language:", language_options)

selected_language_code = [key for key, value in language_display.items() if value == selected_language_name][0]


selected_questions = questions[selected_language_code]
selected_text = text_dict[selected_language]

# 显示表单
with st.form("bfi_form"):
    st.title(selected_text[0])  # 标题
    st.markdown(selected_text[1])  # 介绍
    
    # 性别选择
    gender = st.selectbox(selected_text[2], ["Female", "Male"])
    age = st.number_input(selected_text[3], min_value=18, max_value=70, value=25)
    
    if age < 18 or age > 70:
        st.warning(selected_text[4])  # 年龄警告
        st.stop()  # 停止执行

    if "age" not in st.session_state:
        st.session_state.age = 25  # 默认年龄

    if "gender" not in st.session_state:
        st.session_state.gender = "Female"  # 默认性别
    
    st.subheader(selected_text[7])  # 问卷填写提示

    response_dict = {}
    for i, q in enumerate(selected_text[5:]):
        key = f"q{i}"
        response_dict[key] = st.slider(
            q,
            min_value=1, max_value=6,
            value=st.session_state.get(key, 3),
            key=key
        )

    if all(v is not None for v in response_dict.values()):
        submitted = st.form_submit_button("🎯 Submit and Recommend Careers")
    else:
        submitted = False
        st.warning(selected_text[8])  # 提示用户回答所有问题

# %%
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
    mu = mean_norms[mean_norms['group'] == normgroup].iloc[0, 1:].values
    sigma = sd_norms[sd_norms['group'] == normgroup].iloc[0, 1:].values

    # Step 2: 用户回答转 numpy
    responses = np.array([response_dict[f"q{i}"] for i in range(len(questions))])

    # Step 3: Z 分数
    Z = (responses - mu) / sigma

    # Step 4: Big Five 得分
    big5_scores = np.dot(Z, weights.values)
    T_scores = 10 * big5_scores + 50

    # Step 5: 标准化
    scaled_input = scaler.transform([T_scores])

    trait_names = ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]
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
        top_indices = np.argsort(scores)[-10:][::-1]
        bottom_indices = np.argsort(scores)[:10]

        st.subheader("🧠 Recommended Careers Top-10")
        for rank, idx in enumerate(top_indices, 1):
            st.write(f"NO.{rank} - {job_names[idx]}")

        st.subheader("😬 Least Recommended Careers Bottom-10")
        for rank, idx in enumerate(bottom_indices, 1):
            st.write(f"NO.{rank} - {job_names[idx]}")

    # 安全字符处理函数
    def safe_text(text):
        return str(text).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")

    # PDF 报告生成
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=safe_text("Big Five Personality Test Results"), ln=True, align='C')

    pdf.ln(10)
    pdf.cell(200, 10, txt=safe_text(f"Gender: {gender}"), ln=True)
    pdf.cell(200, 10, txt=safe_text(f"Age: {age}"), ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt=safe_text("Big Five Personality Scores (T scores):"), ln=True)
    for trait, score in zip(trait_names, T_scores):
        pdf.cell(200, 10, txt=safe_text(f"{trait}: {score:.2f}"), ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt=safe_text("Recommended Careers Top-10:"), ln=True)
    for rank, idx in enumerate(top_indices, 1):
        pdf.cell(200, 10, txt=safe_text(f"{rank}. {job_names[idx]}"), ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt=safe_text("Least Recommended Careers Bottom-10:"), ln=True)
    for rank, idx in enumerate(bottom_indices, 1):
        pdf.cell(200, 10, txt=safe_text(f"{rank}. {job_names[idx]}"), ln=True)

    pdf_output = "BigFive_Test_Result.pdf"
    pdf.output(pdf_output)

    with open(pdf_output, "rb") as f:
        st.download_button("Download Your PDF Report", f, file_name=pdf_output)






   





