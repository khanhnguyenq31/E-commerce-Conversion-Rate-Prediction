import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout

np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 1. LOAD DỮ LIỆU
# ==========================================
print("--- Đang tải dữ liệu ---")
csv_data = "/kaggle/input/your-dataset.csv"
df = pd.read_csv(csv_data)

# ==========================================
# 2. TIỀN XỬ LÝ (PREPROCESSING)
# ==========================================
print("\n--- Tiền xử lý dữ liệu ---")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['visitorid', 'timestamp'])
df['itemid'] = df['itemid'].fillna(0).astype(int)
df['event'] = df['event'].fillna('view')

# Encode Item & Event
item_encoder = LabelEncoder()
df['item_idx'] = item_encoder.fit_transform(df['itemid'])
vocab_size_item = len(item_encoder.classes_) + 1 

event_encoder = LabelEncoder()
df['event_idx'] = event_encoder.fit_transform(df['event'])
vocab_size_event = len(event_encoder.classes_) + 1

# Gom nhóm thành chuỗi (Sequence)
grouped = df.groupby('visitorid').agg({
    'item_idx': list,
    'event_idx': list,
    'event': lambda x: 1 if 'transaction' in x.values else 0
}).rename(columns={'event': 'label'})

# Padding
MAX_LEN = 20 # Độ dài chuỗi quan sát
X_item = pad_sequences(grouped['item_idx'], maxlen=MAX_LEN, padding='post', truncating='post')
X_event = pad_sequences(grouped['event_idx'], maxlen=MAX_LEN, padding='post', truncating='post')
y = grouped['label'].values

# Chia tập Train/Test
indices = np.arange(len(y))
X_item_train, X_item_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_item, y, indices, test_size=0.2, random_state=42, stratify=y
)
X_event_train = X_event[idx_train]
X_event_test = X_event[idx_test]

# ==========================================
# 3. CÂN BẰNG DỮ LIỆU (OVERSAMPLING)
# ==========================================
print("\n--- Cân bằng dữ liệu (Oversampling) ---")
idx_buy = np.where(y_train == 1)[0]
idx_nobuy = np.where(y_train == 0)[0]

# Nhân bản lớp Mua cho bằng lớp Không mua
idx_buy_upsampled = np.random.choice(idx_buy, size=len(idx_nobuy), replace=True)
idx_balanced = np.concatenate([idx_nobuy, idx_buy_upsampled])
np.random.shuffle(idx_balanced)

X_item_resampled = X_item_train[idx_balanced]
X_event_resampled = X_event_train[idx_balanced]
y_resampled = y_train[idx_balanced]
print(f"Số lượng mẫu Train sau cân bằng: {len(y_resampled)} (50% Mua - 50% Không mua)")

# ==========================================
# 4. TRAINING LSTM
# ==========================================
# Input Item
in_item = Input(shape=(MAX_LEN,), name='input_item')
emb_item = Embedding(vocab_size_item, 32)(in_item)

# Input Event
in_event = Input(shape=(MAX_LEN,), name='input_event')
emb_event = Embedding(vocab_size_event, 8)(in_event)

# Merge & LSTM
merged = Concatenate()([emb_item, emb_event])
lstm = LSTM(256)(merged)
dropout = Dropout(0.3)(lstm)
out = Dense(1, activation='sigmoid')(dropout)

model = Model(inputs=[in_item, in_event], outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- Bắt đầu Train ---")
model.fit(
    [X_item_resampled, X_event_resampled],
    y_resampled,
    epochs=10, 
    batch_size=64, 
    verbose=1
)

# ==========================================
# 5. ĐÁNH GIÁ & XUẤT KẾT QUẢ
# ==========================================
print("\n" + "="*30)
print("KẾT QUẢ ĐÁNH GIÁ (TRÊN TẬP TEST GỐC)")
print("="*30)

# Dự đoán
y_pred_prob = model.predict([X_item_test, X_event_test])
y_pred = (y_pred_prob > 0.5).astype(int)

# 1. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Không mua', 'Mua']))

# 2. Confusion Matrix (Lưu ảnh)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ko Mua', 'Mua'], yticklabels=['Ko Mua', 'Mua'])
plt.title('Confusion Matrix')
plt.ylabel('Thực tế')
plt.xlabel('Dự báo')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

# ==========================================
# 6. LƯU MODEL & ENCODER (ĐỂ DÙNG LẠI)
# ==========================================
print("\n--- Lưu trữ Model ---")
model.save('lstm_model.h5')
with open('item_encoder.pkl', 'wb') as f: pickle.dump(item_encoder, f)
with open('event_encoder.pkl', 'wb') as f: pickle.dump(event_encoder, f)
print("-> Đã lưu model (.h5) và encoder (.pkl)")