import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import clean_text_deep

st.set_page_config(page_title="Deep Learning AI Detector", page_icon="🧠", layout="centered")

st.title("🧠 Yapay Sinir Ağı & Aktif Öğrenme")
st.markdown("Model yanılırsa aşağıdan doğrusunu işaretleyerek kendi kendine öğrenmesini sağlayabilirsiniz.")

@st.cache_resource
def load_resources():
    model = load_model('results/deep_model.keras')
    with open('results/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

with st.spinner('Model Yükleniyor...'):
    model, tokenizer = load_resources()

user_input = st.text_area("Analiz edilecek İngilizce metni buraya yapıştırın:", height=200)

if st.button("Metni Analiz Et", type="primary"):
    if user_input.strip() != "":
        cleaned_text = clean_text_deep(user_input)
        seq = tokenizer.texts_to_sequences([cleaned_text])
        
        # TEŞHİS KORUMASI: Eğer metinde bilinen hiçbir kelime yoksa (Örn: Türkçe yazıldıysa)
        if len(seq[0]) == 0:
            st.warning("⚠️ Bu metin modelin sözlüğünde yok! (Farklı bir dil veya anlamsız harfler olabilir). Lütfen İngilizce ve anlamlı cümleler girin.")
        else:
            padded_seq = pad_sequences(seq, maxlen=250, padding='post', truncating='post')
            
            # --- RÖNTGEN KODLARI (BUNLARI EKLE) ---
            st.warning(f"🔍 **RÖNTGEN 1 (Modele Giren Sayılar):** {padded_seq[0][:15]}...")
            # -------------------------------------

            prediction_score = model.predict(padded_seq)[0][0]
            
            # --- RÖNTGEN KODLARI 2 (BUNU EKLE) ---
            st.info(f"🔍 **RÖNTGEN 2 (Arka Plandaki Ham Skor):** {prediction_score}")
            # -------------------------------------

            st.divider()
            if prediction_score > 0.5:
                eminlik = prediction_score * 100
                st.error("### 🤖 SONUÇ: YAPAY ZEKA (AI)")
                st.session_state['last_prediction'] = 1 # Tahmini hafızaya al
            else:
                eminlik = (1 - prediction_score) * 100
                st.success("### 👤 SONUÇ: İNSAN (HUMAN)")
                st.session_state['last_prediction'] = 0
                
            st.write(f"**Eminlik Oranı:** %{eminlik:.2f}")
            st.session_state['last_text'] = user_input # Metni hafızaya al

# --- KENDİ KENDİNE ÖĞRENME (FEEDBACK LOOP) ---
if 'last_text' in st.session_state:
    st.markdown("### 🔄 Modele Doğrusunu Öğret (Geri Bildirim)")
    st.write("Model hata mı yaptı? Doğru sınıfı seçerek veri setine eklenmesini sağla.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Bu metin aslında İNSAN (0)"):
            new_data = pd.DataFrame([{"text": st.session_state['last_text'], "generated": 0}])
            new_data.to_csv("data/feedback_data.csv", mode='a', header=not os.path.exists("data/feedback_data.csv"), index=False)
            st.success("Veri eklendi! Model bir sonraki eğitimde bu hatayı düzeltecek.")
    with col2:
        if st.button("Bu metin aslında YAPAY ZEKA (1)"):
            new_data = pd.DataFrame([{"text": st.session_state['last_text'], "generated": 1}])
            new_data.to_csv("data/feedback_data.csv", mode='a', header=not os.path.exists("data/feedback_data.csv"), index=False)
            st.success("Veri eklendi! Model bir sonraki eğitimde bu hatayı düzeltecek.")