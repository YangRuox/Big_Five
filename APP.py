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
from collections import defaultdict
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
jobs = np.load("jobs.npy", allow_pickle=True)

scaler = joblib.load("your_scaler.pkl")  
similarity_matrix = np.load("similarity_matrix.npy")

text_dict = np.load("text_dict.npy", allow_pickle=True).item()
language_display = np.load("language_display.npy", allow_pickle=True).item()

title_translations = {
    "en": "🧬 Your Big Five Personality Profile (T scores)",
    "zh": "🧬 你的大五人格雷达图（T分）",
    "es": "🧬 Tu Perfil de Personalidad Big Five (Puntajes T)",
    "fr": "🧬 Votre profil de personnalité Big Five (scores T)",
    "ru": "🧬 Ваш профиль личности по Big Five (T-баллы)",
    "ar": "🧬 ملف الشخصية الخاص بك (Big Five) بدرجات T",
}
trait_names = {
    "en": ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"],
    "zh": ["神经质", "外向性", "开放性", "宜人性", "尽责性"],
    "es": ["Neuroticismo", "Extraversión", "Apertura", "Amabilidad", "Conciencia"],
    "fr": ["Névrosisme", "Extraversion", "Ouverture", "Amabilité", "Conscienciosité"],
    "ru": ["Невротизм", "Экстраверсия", "Открытость", "Доброжелательность", "Сознательность"],
    "ar": ["الضيق العصبي", "الانفتاح", "الانبساطية", "التعاطف", "الضمير المهني"]
}

job_en = jobs
job_ar = np.load("job_ar.npy", allow_pickle=True).item()
job_fr = np.load("job_fr.npy", allow_pickle=True).item()
job_es = np.load("job_es.npy", allow_pickle=True).item()
job_ru = np.load("job_ru.npy", allow_pickle=True).item()
job_zh = np.load("job_zh.npy", allow_pickle=True).item()


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
        user_scaled = scaler.transform([user_big5_scores])
        user_tensor = torch.tensor(user_scaled, dtype=torch.float32)

        logits = model(user_tensor).numpy().flatten()

        match_score = similarity_matrix @ logits

        top_indices = np.argsort(match_score)[-top_k:][::-1]
        top_jobs = [(job_codes[i], job_names[i], match_score[i]) for i in top_indices]  

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

language_options = list(language_display.values())

col1, col2 = st.columns(2)
with col1:
    selected_language_name = st.selectbox("language:", language_options)

selected_language_code = [key for key, value in language_display.items() if value == selected_language_name][0]


selected_questions = questions[selected_language_code]
selected_text = text_dict[selected_language_code]

with st.form("bfi_form"):
    st.title(selected_text[0])  
    st.markdown(selected_text[1]) 

    gender = st.selectbox(selected_text[2], ["Female", "Male"])
    age = st.number_input(selected_text[3], min_value=18, max_value=70, value=25)
    
    if age < 18 or age > 70:
        st.warning(selected_text[4]) 
        st.stop()  

    if "age" not in st.session_state:
        st.session_state.age = 25 

    if "gender" not in st.session_state:
        st.session_state.gender = "Female" 
    
    st.subheader(selected_text[5])  

    response_dict = {}
    for i, q in enumerate(selected_questions):
        key = f"q{i}"
        response_dict[key] = st.slider(
            q,
            min_value=1, max_value=6,
            value=st.session_state.get(key, 3),
            key=key
        )

    if all(v is not None for v in response_dict.values()):
        submitted = st.form_submit_button(selected_text[9])
    else:
        submitted = False
        st.warning(selected_text[6])  

# %%
if submitted:
    st.session_state.age = age
    st.session_state.gender = gender

    if gender == "Female":
        normgroup = 1 if age < 35 else 2
    else:
        normgroup = 3 if age < 35 else 4

    progress = st.progress(0, text="⏳ Processing...")
    for i in range(60):
        time.sleep(0.005)
        progress.progress(i + 1)

    mu = mean_norms[mean_norms['group'] == normgroup].iloc[0, 1:].values
    sigma = sd_norms[sd_norms['group'] == normgroup].iloc[0, 1:].values

    responses = np.array([response_dict[f"q{i}"] for i in range(len(questions))])

    Z = (responses - mu) / sigma

    big5_scores = np.dot(Z, weights.values)
    T_scores = 10 * big5_scores + 50


    scaled_input = scaler.transform([T_scores])
    
    if selected_language_code == 'en':
        trait_names_local = trait_names["en"]
        job_names_local = job_en
        title = "🧬 Your Big Five Personality Profile (T scores)"
        top_subheader = selected_text[7]
        bottom_subheader = selected_text[8]

    elif selected_language_code == 'fr':
        trait_names_local = trait_names["fr"]
        job_names_local = job_fr
        title = "🧬 Votre profil de personnalité Big Five (scores T)"
        top_subheader = selected_text[7]
        bottom_subheader = selected_text[8]

    elif selected_language_code == 'es':
        trait_names_local = trait_names["es"]
        job_names_local = job_es
        title = "🧬 Tu Perfil de Personalidad Big Five (Puntajes T)"
        top_subheader = selected_text[7]
        bottom_subheader = selected_text[8]

    elif selected_language_code == 'zh':
        trait_names_local = trait_names["zh"]
        job_names_local = job_zh
        title = "🧬 你的大五人格雷达图（T分）"
        top_subheader = selected_text[7]
        bottom_subheader = selected_text[8]

    elif selected_language_code == 'ru':
        trait_names_local = trait_names["ru"]
        job_names_local = job_ru
        title = "🧬 Ваш профиль личности по Big Five (T-баллы)"
        top_subheader = selected_text[7]
        bottom_subheader = selected_text[8]

    elif selected_language_code == 'ar':
        trait_names_local = trait_names["ar"]
        job_names_local = job_ar
        title = "🧬 ملف الشخصية الخاص بك (Big Five) بدرجات T"
        top_subheader = selected_text[7]
        bottom_subheader = selected_text[8]





    
    radar_values = list(T_scores) + [T_scores[0]]
    radar_labels = trait_names_local + [trait_names_local[0]]

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
            radialaxis=dict(visible=True, range=[-100, 100], tickfont=dict(size=10)),
        ),
        showlegend=False,
        title=title_translations[selected_language_code]
    )

    st.plotly_chart(fig)

    with torch.no_grad():
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        logits = model(input_tensor).numpy().flatten()
        scores = similarity_matrix @ logits
        top_indices = np.argsort(scores)[-10:][::-1]
        bottom_indices = np.argsort(scores)[:10]

        st.subheader(selected_text[7])
        if selected_language_code == 'en':
            job_display = job_en
        elif selected_language_code == 'zh':
            job_display = job_zh
        elif selected_language_code == 'fr':
            job_display = job_fr
        elif selected_language_code == 'es':
            job_display = job_es
        elif selected_language_code == 'ru':
            job_display = job_ru
        elif selected_language_code == 'ar':
            job_display = job_ar
        else:
            job_display = job_en 

        for rank, idx in enumerate(top_indices, 1):
            st.write(f"NO.{rank} - {job_display[idx]}")
        st.subheader(selected_text[8])  
        
        for rank, idx in enumerate(bottom_indices, 1):
            st.write(f"NO.{rank} - {job_display[idx]}")
        st.subheader(selected_text[8])
        


    def safe_text(text):
        return str(text).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=safe_text("Big Five Personality Test Results"), ln=True, align='C')

    pdf.ln(10)
    pdf.cell(200, 10, txt=safe_text(f"Gender: {gender}"), ln=True)
    pdf.cell(200, 10, txt=safe_text(f"Age: {age}"), ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt=safe_text("Big Five Personality Scores (T scores):"), ln=True)
    for trait, score in zip(trait_names_local, T_scores):
        pdf.cell(200, 10, txt=safe_text(f"{trait}: {score:.2f}"), ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt=safe_text("Big Five Personality Scores (Z scores):"), ln=True)
    for trait, z in zip(trait_names_local, Z):
        pdf.cell(200, 10, txt=safe_text(f"{trait}: {z:.2f}"), ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt=safe_text("Recommended Careers Top-10:"), ln=True)
    for rank, idx in enumerate(top_indices, 1):
        pdf.cell(200, 10, txt=safe_text(f"{rank}. {job_display[idx]}"), ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt=safe_text("Least Recommended Careers Bottom-10:"), ln=True)
    for rank, idx in enumerate(bottom_indices, 1):
        pdf.cell(200, 10, txt=safe_text(f"{rank}. {job_display[idx]}"), ln=True)

    pdf_output = "BigFive_Test_Result.pdf"
    pdf.output(pdf_output)

    with open(pdf_output, "rb") as f:
        st.download_button("Download Your PDF Report", f, file_name=pdf_output)






   





