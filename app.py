import streamlit as st
import librosa
import numpy as np
import pandas as pd
from collections import Counter
import datetime
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Amapiano Master | Universal", page_icon="⚫", layout="wide")

# Initialisation de l'historique de session
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- STYLE CSS SOFT MONOCHROME (Anti-Fatigue) ---
st.markdown("""
    <style>
    .stApp { background-color: #0F0F0F; color: #E0E0E0; }
    
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        letter-spacing: 4px;
        color: #E0E0E0;
        text-align: center;
        border-bottom: 1px solid #222;
        padding-bottom: 20px;
    }

    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 0.4; }
        100% { opacity: 0.8; }
    }
    .analyzing { animation: pulse 2s infinite; }

    .track-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem !important;
        font-weight: 700;
        text-transform: uppercase;
        color: #E0E0E0;
        text-align: center;
        margin: 30px 0;
        border: 1px solid #333;
        padding: 25px;
        background-color: #151515;
    }

    div[data-testid="stMetric"] {
        background-color: #121212;
        border: 1px solid #333;
        border-radius: 4px;
        padding: 20px;
    }
    
    div[data-testid="stMetricLabel"] { color: #888888 !important; font-size: 0.9rem !important; }
    div[data-testid="stMetricValue"] { color: #E0E0E0 !important; font-size: 2rem !important; }

    .history-card {
        background-color: #0F0F0F;
        border-bottom: 1px solid #222;
        padding: 12px;
        color: #AAAAAA;
        font-family: 'Courier New', monospace;
    }

    .stFileUploader { border: 1px dashed #444; border-radius: 4px; background-color: #111; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE MUSICALE ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def get_camelot(key):
    camelot_map = {
        'G# minor': '1A', 'D# minor': '2A', 'Bb minor': '3A', 'F minor': '4A', 
        'C minor': '5A', 'G minor': '6A', 'D minor': '7A', 'A minor': '8A', 
        'E minor': '9A', 'B minor': '10A', 'F# minor': '11A', 'C# minor': '12A'
    }
    return camelot_map.get(f"{key} minor", "12A")

def analyze_audio(file):
    try:
        # Chargement robuste (nécessite ffmpeg sur le serveur pour le M4A)
        y, sr = librosa.load(file, duration=45, offset=30, sr=22050)
        
        # Extraction harmonique (isole le Log Drum et les mélodies)
        y_harm = librosa.effects.hpss(y)[0]
        
        # Analyse de tonalité via CQT (Précision basses fréquences)
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
        chroma_avg = np.mean(chroma, axis=1)
        
        best_score = -1
        res_key = ""
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(MINOR_PROFILE, i))[0, 1]
            if score > best_score:
                best_score, res_key = score, NOTES[i]
        
        # Détection du BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        return res_key, int(tempo)
    except Exception as e:
        st.error(f"Erreur technique de lecture : {e}")
        return None, None

# --- INTERFACE UTILISATEUR ---
st.markdown("<h1>AMAPIANO ANALYZER</h1>", unsafe_allow_html=True)

# Accepte tout type de fichier
uploaded_file = st.file_uploader("GLISSEZ VOTRE FICHIER AUDIO (MP3, WAV, M4A, FLAC...)", type=None)

if uploaded_file:
    # Nettoyage du nom de fichier
    raw_name = uploaded_file.name
    clean_name = os.path.splitext(raw_name)[0].replace("_", " ").replace("-", " ").upper()
    
    # Titre avec animation de scan
    title_placeholder = st.empty()
    title_placeholder.markdown(f'<div class="track-title analyzing">{clean_name}</div>', unsafe_allow_html=True)

    with st.spinner("DÉCODAGE DU SIGNAL..."):
        key, bpm = analyze_audio(uploaded_file)
        
        if key and bpm:
            camelot = get_camelot(key)
            
            # Mise à jour de l'historique
            entry = {
                "time": datetime.datetime.now().strftime("%H:%M"),
                "name": clean_name,
                "key": f"{key}M",
                "camelot": camelot,
                "bpm": bpm
            }
            if not st.session_state.history or st.session_state.history[0]['name'] != entry['name']:
                st.session_state.history.insert(0, entry)

            # Fixation du titre (arrêt de l'animation)
            title_placeholder.markdown(f'<div class="track-title">{clean_name}</div>', unsafe_allow_html=True)

            # Dashboard de résultats
            c1, c2, c3 = st.columns(3)
            c1.metric("TONALITÉ", f"{key} MINOR")
            c2.metric("CAMELOT", camelot)
            c3.metric("TEMPO", f"{bpm} BPM")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.audio(uploaded_file)

# --- SECTION LOG ---
st.divider()
st.markdown("### SESSION LOG")
if st.session_state.history:
    for item in st.session_state.history:
        st.markdown(f'<div class="history-card">{item["time"]} | {item["name"]} | {item["key"]} | {item["camelot"]} | {item["bpm"]} BPM</div>', unsafe_allow_html=True)
