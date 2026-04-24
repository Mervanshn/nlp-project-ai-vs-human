import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from src.preprocessing import clean_text_deep

def evaluate_and_plot():
    print("1. Model ve Sözlük (Tokenizer) Yükleniyor...")
    try:
        model = load_model('results/deep_model.keras')
        with open('results/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    except FileNotFoundError:
        print("Hata: Model veya Tokenizer bulunamadı. Önce main_deep.py'yi çalıştırın.")
        return

    print("2. Test Verisi Yeniden Hazırlanıyor...")
    # Eğitimi yaptığımız aynı rastgelelik (random_state=42) ile veriyi çekiyoruz ki 
    # modelin daha önce hiç görmediği gerçek "Test" verisini bulabilelim.
    df = pd.read_csv('data/AI_Human.csv')
    df_human = df[df['generated'] == 0].sample(15000, random_state=42)
    df_ai = df[df['generated'] == 1].sample(15000, random_state=42)
    df_all = pd.concat([df_human, df_ai]).sample(frac=1, random_state=42).reset_index(drop=True)

    df_all['text_clean'] = df_all['text'].apply(clean_text_deep)
    X = pad_sequences(tokenizer.texts_to_sequences(df_all['text_clean']), maxlen=250, padding='post', truncating='post')
    y = df_all['generated'].values

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("3. Model Test Verisi Üzerinde Tahmin Yapıyor (Lütfen bekleyin)...")
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int) # 0.5'ten büyükleri 1 (AI), küçükleri 0 (İnsan) yap

    print("\n" + "="*50)
    print("📊 SINIFLANDIRMA RAPORU (Hocanın İstediği Sayılar)")
    print("="*50)
    # Hocanın raporda görmek isteyeceği F1-Score, Precision ve Recall değerleri:
    print(classification_report(y_test, y_pred, target_names=['İnsan (0)', 'Yapay Zeka (1)']))

    print("\n4. Karmaşıklık Matrisi (Confusion Matrix) Çiziliyor...")
    cm = confusion_matrix(y_test, y_pred)
    
    # Grafik Ayarları
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['İnsan', 'Yapay Zeka'], 
                yticklabels=['İnsan', 'Yapay Zeka'],
                annot_kws={"size": 16}) # Sayıları büyüttük
    
    plt.title('Karmaşıklık Matrisi (Confusion Matrix)', fontsize=16)
    plt.ylabel('Gerçek Değer', fontsize=14)
    plt.xlabel('Modelin Tahmini', fontsize=14)
    
    # Grafiği kaydet
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Başarılı! Grafik 'results/confusion_matrix.png' olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()