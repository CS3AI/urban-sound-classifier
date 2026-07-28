import streamlit as st
import numpy as np
import librosa
import joblib
import os

st.set_page_config(page_title="Urban Sound Classifier", page_icon="🔊")

@st.cache_resource
def load_models():
    router = joblib.load('router_final.pkl')
    expert_1 = joblib.load('expert_1_final.pkl')
    expert_2 = joblib.load('expert_2_final.pkl')
    scaler = joblib.load('scaler_final.pkl')
    return router, expert_1, expert_2, scaler

router_final, expert_1_final, expert_2_final, scaler_final = load_models()

CATEGORY_NAMES = {0: 'Human/Transient', 1: 'Mechanical/Vehicle'}
CLASS_NAMES = {0: 'air_conditioner', 1: 'car_horn', 2: 'children_playing', 3: 'dog_bark',
               4: 'drilling', 5: 'engine_idling', 6: 'gun_shot', 7: 'jackhammer',
               8: 'siren', 9: 'street_music'}

def extract_features_v2(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean, zcr_std = np.mean(zcr), np.std(zcr)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)

        spec_feats = [
            np.mean(spec_cent), np.std(spec_cent),
            np.mean(spec_bw), np.std(spec_bw),
            np.mean(spec_roll), np.std(spec_roll)
        ]
        return np.hstack([mfcc_mean, mfcc_std, zcr_mean, zcr_std, spec_feats])
    except Exception:
        return None

def classify(file_path):
    feat = extract_features_v2(file_path)
    if feat is None:
        return None
    feat_scaled = scaler_final.transform(feat.reshape(1, -1))
    category_pred = router_final.predict(feat_scaled)[0]
    category_proba = router_final.predict_proba(feat_scaled)[0]

    if category_pred == 0:
        final_pred = expert_1_final.predict(feat_scaled)[0]
    else:
        final_pred = expert_2_final.predict(feat_scaled)[0]

    return category_pred, category_proba[category_pred], final_pred

st.title("Urban Sound Classifier — Hybrid Mixture-of-Experts Demo")
st.write("Upload a city sound clip, or try one of the sample sounds below. The system first "
         "determines whether it falls into the 'Human/Transient' or 'Mechanical/Vehicle' category, "
         "then passes it to the corresponding expert model for final, fine-grained classification.")

st.subheader("Try a sample sound")
SAMPLE_DIR = "samples"
sample_files = sorted([f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith('.wav')]) if os.path.isdir(SAMPLE_DIR) else []

selected_sample = None
if sample_files:
    for fname in sample_files:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.audio(os.path.join(SAMPLE_DIR, fname))
        with col2:
            if st.button(f"Use {fname}", key=f"sample_{fname}"):
                selected_sample = os.path.join(SAMPLE_DIR, fname)
else:
    st.caption("(No sample sounds added yet)")

st.subheader("Or upload your own")
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.audio_path = "temp_audio.wav"
    st.audio(uploaded_file, format="audio/wav")
elif selected_sample is not None:
    st.session_state.audio_path = selected_sample

if st.session_state.audio_path is not None:
    if st.button("Analyze Sound", type="primary"):
        with st.spinner("Extracting features and classifying..."):
            result = classify(st.session_state.audio_path)

        if result is None:
            st.error("Unable to process this audio; please try a different file.")
        else:
            category_pred, confidence, final_pred = result
            st.subheader(f"Predicted Category: {CATEGORY_NAMES[category_pred]}")
            st.write(f"Router confidence: {confidence:.1%}")
            st.subheader(f"Final Predicted Class: **{CLASS_NAMES[final_pred]}**")
