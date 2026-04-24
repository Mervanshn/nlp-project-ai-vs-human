import pandas as pd
import numpy as np
import pickle
import os

# Grafiklerin arka planda güvenle çizilmesi için Sessiz Mod (En tepede olmalı)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Kendi yazdığın modüller
from src.preprocessing import clean_text_deep
from src.model_arch import build_lstm_model

def run_fine_tuning():
    print("1. Veri hazırlanıyor...")
    df = pd.read_csv('data/AI_Human.csv')
    
    # 15.000 İnsan, 15.000 Yapay Zeka (Dengeli Veri)
    df_human = df[df['generated'] == 0].sample(15000, random_state=42)
    df_ai = df[df['generated'] == 1].sample(15000, random_state=42)
    df_all = pd.concat([df_human, df_ai]).sample(frac=1, random_state=42).reset_index(drop=True)

    print("2. Metinler temizleniyor...")
    df_all['text_clean'] = df_all['text'].apply(clean_text_deep)

    print("3. Sayısallaştırma (Tokenization)...")
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df_all['text_clean'])
    X = pad_sequences(tokenizer.texts_to_sequences(df_all['text_clean']), maxlen=250, padding='post', truncating='post')
    y = df_all['generated'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # İnsan metnini kaçırmamak için uyguladığımız özel ağırlık (Projenin yıldızı)
    custom_weights = {0: 2.0, 1: 1.0} 

    print("4. Model Kuruluyor...")
    model = build_lstm_model(20000, 128, 250)
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("\n5. Eğitim Başlıyor (Bu eğitimden sonra grafikler kesin çıkacak!)...")
    history = model.fit(
        X_train, y_train, 
        epochs=10, 
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=custom_weights,
        callbacks=[early_stop]
    )

    print("\n6. Model Kaydediliyor...")
    model.save('results/deep_model.keras')
    with open('results/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
        
    print("7. Eğitim Grafikleri Çiziliyor (Güvenli Mod)...")
    plt.close('all') # Önceki tüm açık pencereleri zorla kapat
    fig = plt.figure(figsize=(12, 5))
    
    # Başarı (Accuracy) Grafiği
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history.history['accuracy'], 'b-', label='Eğitim Başarısı', linewidth=2)
    ax1.plot(history.history['val_accuracy'], 'r--', label='Test Başarısı', linewidth=2)
    ax1.set_title('Model Öğrenme Eğrisi (Accuracy)', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Doğruluk', fontsize=12)
    ax1.legend()
    ax1.grid(True)
    
    # Kayıp (Loss) Grafiği
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(history.history['loss'], 'g-', label='Eğitim Kaybı', linewidth=2)
    ax2.plot(history.history['val_loss'], 'y--', label='Test Kaybı', linewidth=2)
    ax2.set_title('Hata Düşüş Eğrisi (Loss)', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Kayıp (Hata)', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    # Dosya zaten varsa ve kilitliyse önce onu silmeyi dene
    output_path = os.path.join('results', 'learning_curve.png')
    if os.path.exists(output_path):
        try:
            os.remove(output_path) 
        except Exception as e:
            print(f"Uyarı: Eski dosya silinemedi. {e}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig) # Belleği tamamen boşalt
    print(f"✅ HARİKA! Çizgi grafik '{output_path}' olarak mükemmel bir şekilde kaydedildi.")

if __name__ == "__main__":
    run_fine_tuning()