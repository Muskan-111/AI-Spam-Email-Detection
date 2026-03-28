import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC # <-- Naya powerful model

# 1. Load and Train
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Advanced Vectorizer (Phrases ko pakadne ke liye)
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Advanced Model (SVM with Probability)
model = SVC(kernel='linear', probability=True) 
model.fit(X, y)

# 2. Real World Interface
print("\n--- UPGRADED AI SPAM CHECKER ---")
sender = input("Sender's Email: ")
subject = input("Subject Line: ")
msg_body = input("Email Content: ")

# Sabko combine karke check karte hain
full_email = f"Subject: {subject} Content: {msg_body}"
input_data = vectorizer.transform([full_email])

# Prediction + Confidence Score
prediction = model.predict(input_data)
prob = model.predict_proba(input_data)[0]
confidence = max(prob) * 100

print("\n" + "="*30)
if prediction[0] == 1:
    print(f"RESULT: 🚨 SPAM DETECTED! ({confidence:.2f}% Sure)")
    print(f"Warning: This message looks like a scam.")
else:
    print(f"RESULT: ✅ SAFE EMAIL ({confidence:.2f}% Sure)")
    print(f"This message appears to be normal.")
print("="*30)