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
from collections import defaultdict
from joblib import load
import streamlit as st
import joblib
import plotly.graph_objects as go
from fpdf import FPDF
import time
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer



# %%
big5_df = pd.read_excel('Job-profile.xlsx', sheet_name='Big Five Domains')
features = big5_df[['Neuroticism (M)', 'Extraversion (M)', 
                    'Openness (M)', 'Agreeableness (M)', 
                    'Conscientiousness (M)']]

# %%
job_names = np.load("job_names.npy", allow_pickle=True)
job_codes = np.load("job_codes.npy", allow_pickle=True)
jobs = np.load("jobs.npy", allow_pickle=True)

pca_weights = np.load("pca_weights.npy")
scaled_features = np.load("scaled_job_features.npy")


scaler = joblib.load("your_scaler.pkl")  


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

disclaimer_text = {
    "en": "The career recommendations provided here are for your reference only. It's important to consider your personal circumstances, preferences, and goals when making a career decision. We encourage you to explore different options and take the time to evaluate each one carefully. May you find a fulfilling and rewarding career that aligns with your values and aspirations. Best of luck on your journey to success! 😊",
    "zh": "这里提供的职业推荐仅供参考。在做出职业选择时，请务必考虑您的个人情况、兴趣和目标。我们鼓励您探索不同的职业选项，并仔细评估每一个选择。希望您能够找到一个符合自己价值观和人生目标的理想职业，祝您在职业生涯中取得圆满成功！😊",
    "es": "Las recomendaciones de carrera proporcionadas aquí son solo para su referencia. Es importante tener en cuenta sus circunstancias personales, preferencias y objetivos al tomar una decisión sobre su carrera. Le animamos a explorar diferentes opciones y tomarse el tiempo necesario para evaluar cada una de ellas con cuidado. ¡Le deseamos mucho éxito en su camino hacia una carrera gratificante y satisfactoria! 😊",
    "fr": "Les recommandations de carrière fournies ici sont uniquement à titre de référence. Il est important de prendre en compte vos circonstances personnelles, vos préférences et vos objectifs lorsque vous prenez une décision concernant votre carrière. Nous vous encourageons à explorer différentes options et à prendre le temps d'évaluer chaque choix avec soin. Nous vous souhaitons de trouver une carrière épanouissante et gratifiante qui corresponde à vos valeurs et aspirations. Bonne chance dans votre parcours vers le succès ! 😊",
    "ru": "Предоставленные рекомендации по карьере предназначены только для вашего ознакомления. Важно учитывать ваши личные обстоятельства, предпочтения и цели при принятии решения о карьере. Мы призываем вас исследовать различные варианты и уделять достаточно времени на тщательную оценку каждого из них. Желаем вам найти карьеру, которая будет соответствовать вашим ценностям и устремлениям, и успешного пути к успеху! 😊",
    "ar": "التوصيات المهنية المقدمة هنا هي للإشارة فقط. من المهم أن تأخذ في اعتبارك ظروفك الشخصية واهتماماتك وأهدافك عند اتخاذ قرار بشأن مهنتك. نشجعك على استكشاف خيارات مختلفة وأخذ الوقت الكافي لتقييم كل خيار بعناية. نتمنى لك التوفيق في إيجاد مهنة مجزية ومرضية تتناسب مع قيمك وطموحاتك. نتمنى لك كل النجاح في مسيرتك المهنية! 😊"
}
job_display = {
    "en": np.load("job_en.npy", allow_pickle=True),
    "zh": np.load("job_zh.npy", allow_pickle=True),
    "es": np.load("job_es.npy", allow_pickle=True),
    "fr": np.load("job_fr.npy", allow_pickle=True), 
    "ru": np.load("job_ru.npy", allow_pickle=True),
    "ar": np.load("job_ar.npy", allow_pickle=True)
}
ideal_job_prompt = {
    "en": "Please enter your ideal career (e.g., Data Scientist):",
    "zh": "请输入您的理想职业（例如：数据科学家）：",
    "es": "Por favor, introduzca su carrera ideal (por ejemplo: Científico de datos):",
    "fr": "Veuillez saisir votre métier idéal (par exemple : Data Scientist) :",
    "ru": "Пожалуйста, введите вашу идеальную профессию (например: специалист по данным):",
    "ar": "يرجى إدخال مهنتك المثالية (مثال: عالم بيانات):"
}

ideal_job_warning = {
    "en": "⚠️ Please enter your ideal career.",
    "zh": "⚠️ 请输入您的理想职业。",
    "es": "⚠️ Por favor, introduzca su carrera ideal.",
    "fr": "⚠️ Veuillez saisir votre métier idéal.",
    "ru": "⚠️ Пожалуйста, введите вашу идеальную профессию.",
    "ar": "⚠️ يرجى إدخال مهنتك المثالية."
}

ideal_job_result_text = {
    "en": "The career closest to your ideal is: **{}**",
    "zh": "您的理想职业最相近的是：**{}**",
    "es": "La carrera más cercana a su ideal es: **{}**",
    "fr": "Le métier le plus proche de votre idéal est : **{}**",
    "ru": "Самая близкая к вашей идеальной профессия: **{}**",
    "ar": "أقرب مهنة إلى مهنتك المثالية هي: **{}**"
}




traits =  ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]
trait_list = {
    "en": ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"],
    "zh": ["神经质", "外向性", "开放性", "宜人性", "尽责性"],
    "es": ["Neuroticismo", "Extraversión", "Apertura", "Amabilidad", "Responsabilidad"],
    "fr": ["Névrosisme", "Extraversion", "Ouverture", "Agréabilité", "Conscience"],
    "ru": ["Невротизм", "Экстраверсия", "Открытость", "Доброжелательность", "Добросовестность"],
    "ar": ["العُصابية", "الانبساطية", "الانفتاح", "القبول", "الضمير الحي"]
}

job_en = np.load("job_en.npy", allow_pickle=True)
job_ar = np.load("job_ar.npy", allow_pickle=True)
job_fr = np.load("job_fr.npy", allow_pickle=True)
job_es = np.load("job_es.npy", allow_pickle=True)
job_ru = np.load("job_ru.npy", allow_pickle=True)
job_zh = np.load("job_zh.npy", allow_pickle=True)

job_dict = {
    "en": np.load("job_en.npy", allow_pickle=True),
    "zh": np.load("job_zh.npy", allow_pickle=True),
    "es": np.load("job_es.npy", allow_pickle=True),
    "fr": np.load("job_fr.npy", allow_pickle=True),
    "ru": np.load("job_ru.npy", allow_pickle=True),
    "ar": np.load("job_ar.npy", allow_pickle=True)
}


closest_text = {
    "en": "Your trait closest to the ideal career:",
    "zh": "与理想职业特征最接近的是：", 
    "es": "Tu rasgo más cercano al trabajo ideal:",
    "fr": "Votre trait le plus proche du métier idéal :",
    "ru": "Ваша черта, наиболее близкая к идеальной профессии:",
    "ar": "سمتك الأقرب إلى المهنة المثالية:"
}

furthest_text = {
    "en": "Your trait furthest from the ideal career:",
    "zh": "与理想职业特征差距最大的是：",
    "es": "Tu rasgo más alejado del trabajo ideal:",
    "fr": "Votre trait le plus éloigné du métier idéal :", 
    "ru": "Ваша черта, наиболее далёкая от идеальной профессии:",
    "ar": "سمتك الأبعد عن المهنة المثالية:"
}


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
        
def compute_weighted_euclidean_similarity(user_big5, job_features, weights):
    user_df = pd.DataFrame([user_big5], columns=features.columns)
    user_scaled = scaler.transform(user_df)[0]
    diffs = job_features - user_scaled
    weighted_dists = np.sqrt(np.sum(weights * (diffs ** 2), axis=1))

    dist_min, dist_max = weighted_dists.min(), weighted_dists.max()
    normalized = (weighted_dists - dist_min) / (dist_max - dist_min + 1e-10)
    similarities = 1 - normalized
    return similarities

def recommend_jobs_weighted_euclidean(user_big5_scores, model, job_features, weights, top_k=10):
    similarities = compute_weighted_euclidean_similarity(user_big5_scores, job_features, weights)

    user_df = pd.DataFrame([user_big5_scores], columns=features.columns)
    user_scaled = scaler.transform(user_df)
    user_tensor = torch.tensor(user_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = model(user_tensor).numpy().flatten()

    match_scores = similarities * logits
    top_indices = np.argsort(match_scores)[-top_k:][::-1]
    top_jobs = [(job_codes[i], job_names[i], match_scores[i]) for i in top_indices]
    return top_jobs



# %%


# %%
device = torch.device('cpu') 
model = JobRecommenderMLP(input_dim=5, hidden_dim=128, output_dim=len(job_names))
model.load_state_dict(torch.load("your_model.pth", map_location=device))
model.to(device)
model.eval()

model_embedding = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
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

    ideal_job_input = st.text_input(ideal_job_prompt[selected_language_code], key="ideal_job")
   
    response_dict = {}
    for i, q in enumerate(selected_questions):
        key = f"q{i}"
        response_dict[key] = st.slider(
            q,
            min_value=1, max_value=6,
            value=st.session_state.get(key, 3),
            key=key
        )

    submitted = st.form_submit_button(selected_text[9])
  
    if submitted:
       if not all(v is not None for v in response_dict.values()):
           st.warning(selected_text[6])
           st.stop()  
      
 



# %%
if submitted:
    if not ideal_job_input:
        st.warning(ideal_job_warning[selected_language_code])
        st.stop()

    user_input_job = ideal_job_input
    language_code = selected_language_code
    user_embedding = model_embedding.encode([user_input_job], convert_to_tensor=True)
  
    job_list = job_dict[language_code]
    job_embeddings = model_embedding.encode(job_list.tolist(), convert_to_tensor=True)
    def to_numpy(tensor):
        if hasattr(tensor, "cpu"):  
            return tensor.cpu().detach().to(torch.float32).numpy()
        elif hasattr(tensor, "numpy"): 
            return tensor.numpy()
        return np.array(tensor)  
    
    user_np = model_embedding.encode([user_input_job], convert_to_numpy=True)
    job_np = model_embedding.encode(job_list.tolist(), convert_to_numpy=True)
  
    similarities = cosine_similarity(user_np, job_np)[0]
    best_match_index = np.argmax(similarities)
    best_match_job = job_list[best_match_index]
    ideal_big5_score = big5_df.iloc[best_match_index][[
        'Neuroticism (M)', 'Extraversion (M)', 
        'Openness (M)', 'Agreeableness (M)', 
        'Conscientiousness (M)'
    ]].values

    st.markdown(ideal_job_result_text[language_code].format(best_match_job))
 
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
    
    user_df = pd.DataFrame([T_scores], columns=[
        'Neuroticism (M)', 'Extraversion (M)', 
        'Openness (M)', 'Agreeableness (M)', 
        'Conscientiousness (M)'
    ])

    diffs = np.abs(T_scores - ideal_big5_score)
    closest_idx = np.argmin(diffs)
    furthest_idx = np.argmax(diffs)

    st.markdown(ideal_job_result_text[language_code].format(best_match_job))
    st.write(f"{closest_text[language_code]} **{trait_list[language_code][closest_idx]}**")
    st.write(f"{furthest_text[language_code]} **{trait_list[language_code][furthest_idx]}**")

    user_scaled = scaler.transform(user_df)
    user_tensor = torch.tensor(user_scaled, dtype=torch.float32)
  
    with torch.no_grad():
        current_job_display = job_display[selected_language_code]
        logits = model(user_tensor).numpy().flatten()
        logits = (logits - logits.min()) / (logits.max() - logits.min() + 1e-9)

        similarities = compute_weighted_euclidean_similarity(T_scores, scaled_features, pca_weights)
        all_scores = similarities * logits

        top_indices = np.argsort(all_scores)[-10:][::-1]
        bottom_indices = np.argsort(all_scores)[:10]

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
        user_scaled = scaler.transform(user_df)
        user_tensor = torch.tensor(user_scaled, dtype=torch.float32)
        logits = model(user_tensor).numpy().flatten()
        similarities = compute_weighted_euclidean_similarity(T_scores, scaled_features, pca_weights)
        
        all_scores = similarities * logits
        top_indices = np.argsort(all_scores)[-10:][::-1]
        bottom_indices = np.argsort(all_scores)[:10]
    
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

        st.subheader(selected_text[7])
        for rank, idx in enumerate(top_indices, 1):
            st.write(f"NO.{rank} - {job_display[idx]}")
        st.subheader(selected_text[8])  
        
        for rank, idx in enumerate(bottom_indices, 1):
            st.write(f"NO.{rank} - {job_display[idx]}")

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
        st.markdown(f"<h3 style='color:red; font-weight:bold;'>NOTE</h3><p>{disclaimer_text[selected_language_code]}</p>",unsafe_allow_html=True)

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
    for trait, score in zip(traits, T_scores):
        pdf.cell(200, 10, txt=safe_text(f"{trait}: {score:.2f}"), ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt=safe_text("Big Five Personality Scores (Z scores):"), ln=True)
    for trait, z in zip(traits, Z):
        pdf.cell(200, 10, txt=safe_text(f"{trait}: {z:.2f}"), ln=True)

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
        st.download_button("Download Your PDF Report(English)", f, file_name=pdf_output)



