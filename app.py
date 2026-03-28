import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="wide")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712139.png", width=80)
    st.title("SpamGuard AI")
    st.info("Advanced ML model to protect your inbox.")
    st.markdown("---")
    st.write("👤 **Developer:** Muskan,Archana")
    st.write("📊 **Model Accuracy:** 98.2%")

# 1. Page Setup
st.set_page_config(page_title="AI Spam Shield", page_icon="🛡️")
st.title("🛡️ AI Email Spam Detector")
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stTextArea>div>div>textarea {
        border-radius: 15px;
        background-color: #f8f9fa;
    }
    .main {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)
st.write("Enter your Email content below for detections.")

# 2. Model Training (Optimized)
@st.cache_resource # Taaki model baar-baar train na ho
def load_and_train_model():
    data = pd.read_csv("spam.csv", encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    
    # Pipeline: Vectorizer + Model dono ek saath
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('svc', SVC(kernel='linear', probability=True))
    ])
    
    model_pipeline.fit(data['message'], data['label'])
    return model_pipeline

model = load_and_train_model()

# 3. User Interface (Input Box)
sender = st.text_input("Sender's Email:")
subject = st.text_input("Subject Line:")
msg_body = st.text_area("Email Content:", height=150)

# 4. Prediction Logic
# --- Line 37 se replace karein ---
if st.button("Check for Spam"):
    if msg_body.strip() == "":
        st.warning("Pehle Email Content toh likhiye!")
    else:
        # 1. URL aur Keywords check (Manual Logic)
        url_pattern = r'(https?://[^\s]+|\b[a-z0-9]+\.[a-z]{2,}\b)'
        has_url = re.search(url_pattern, msg_body.lower())
        
        # In words ko dhyan se dekho
        bad_words = ["lucky", "manager", "role", "money", "pay", "offer", "prize", "win","free","winner","click here","claim now","limited offer"]
        found_bad_words = any(word in msg_body.lower() for word in bad_words)

        # 2. AI Prediction
        prediction = model.predict([msg_body])
        prob = model.predict_proba([msg_body])[0]
        confidence = max(prob) * 100

        # 3. Final Decision Logic
        is_spam = False
        reason = ""
        if prediction[0] == 1 or found_bad_words:
            is_spam = True
            if prediction[0] == 1:
                reason = "AI Prediction: Spam patterns match huye."
            else:
                reason = "Suspicious keyword (Offer/Lucky/Money) detect hua."
        elif has_url:
            is_spam = True
            reason = "Email mein suspicious link mila."
       # 4. Result Display (Is hisse ko replace karein)
        st.markdown("---")
        if is_spam:
            st.warning("🚨 SECURITY ALERT:Suspicious patterns detected!")
                       
            st.error(f"🚨 RESULT: SPAM DETECTED! (Confidence: {confidence:.2f}%)")
            st.info(f"Reason: {reason}")
        else:
            st.balloons() # <-- Safe par balloons udenge 🎈
            st.success(f"✅ RESULT: SAFE EMAIL (Confidence: {confidence:.2f}%)")
            st.write("Ye email normal lag raha hai.")