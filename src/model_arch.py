from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D

def build_lstm_model(vocab_size, embedding_dim, max_length):
    model = Sequential([
        # 1. Gözlük Takılı (Sıfırları Atlar)
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, mask_zero=True),
        
        SpatialDropout1D(0.3),
        
        # 2. KRİTİK DEĞİŞİKLİK: return_sequences=True SİLİNDİ!
        # Artık kelime kelime değil, LSTM'den "tüm cümlenin tek bir özetini" istiyoruz.
        Bidirectional(LSTM(64)), 
        
        # 3. GlobalMaxPool1D SİLİNDİ! (Maskelemeyle çakışıp modeli %50'ye kilitliyordu)
        
        # 4. Karar Aşaması
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    # Öğrenme hızını biraz daha canlandıralım ki %50'de takılı kalmasın
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model