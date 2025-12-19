import streamlit as st
import librosa
import numpy as np
import pandas as pd
from collections import Counter
import datetime
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Amapiano Master | Universal", page_icon="⚫", layout="wide")

if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- STYLE CSS NOIR ET BLANC ---
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 200;
        letter-spacing: 5px;
        color: #FFFFFF;
        text-align: center;
        border-bottom: 1px solid #333;
        padding-bottom: 20px;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    .analyzing { animation: pulse 1.5s infinite; }

    .track-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem !important;
        font-weight: 800;
        text-transform: uppercase;
        color: #FFFFFF;
        text-align: center;
        margin: 40px 0;
        line-height: 1.2;
        border: 2px solid #FFFFFF;
        padding: 20px;
        word-wrap: break-word;
    }

    div[data-testid="stMetric"] {
        background-color: #000000;
        border: 1px solid #FFFFFF;
        border-radius: 0px;
        padding: 25px;
    }
    
    div[data-testid="stMetricLabel"] { color: #888888 !important; letter-spacing: 2px; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 2.5rem !important; }

    .history-card {
        background-color: #000000;
        border-bottom: 1px solid #222;
        padding: 15px;
        font-family: 'Courier New', monospace;
    }

    .stFileUploader { border: 1px dashed #444; border-radius: 0px; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE MUSICALE ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def get_camelot(key):
    camelot_map = {'G# minor': '1A', 'D# minor': '2A', 'Bb minor': '3A', 'F minor': '4A', 'C minor': '5A', 'G minor': '6A', 'D minor': '7A', 'A minor': '8A', 'E minor': '9A', 'B minor': '10A', 'F# minor': '11A', 'C# minor': '12A'}
    return camelot_map.get(f"{key} minor", "12A")

def analyze_audio(file):
    # Librosa charge presque tout grâce au backend 'audioread'
    y, sr = librosa.load(file, duration=45, offset=30)
    y_harm = librosa.effects.hpss(y)[0]
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    best_score = -1
    res_key = ""
    for i in range(12):
        score = np.corrcoef(chroma_avg, np.roll(MINOR_PROFILE, i))[0, 1]
        if score > best_score:
            best_score, res_key = score, NOTES[i]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return res_key, int(tempo)

# --- INTERFACE ---
st.markdown("<h1>RICARDODJ_228 KEY ANALYZER</h1>", unsafe_allow_html=True)

# Accepter tous les types de fichiers audio
uploaded_file = st.file_uploader("DROP ANY AUDIO FILE (MP3, WAV, FLAC, M4A, OGG, AIFF...)", type=None)

if uploaded_file:
    # Nettoyage automatique du titre (enlève l'extension peu importe sa longueur)
    raw_name = uploaded_file.name
    clean_name = os.path.splitext(raw_name)[0].replace("_", " ").replace("-", " ").upper()
    
    title_placeholder = st.empty()
    title_placeholder.markdown(f'<div class="track-title analyzing">{clean_name}</div>', unsafe_allow_html=True)

    try:
        with st.spinner("DECODING UNIVERSAL AUDIO..."):
            key, bpm = analyze_audio(uploaded_file)
            camelot = get_camelot(key)
            
            entry = {"time": datetime.datetime.now().strftime("%H:%M"), "name": clean_name, "key": key, "camelot": camelot, "bpm": bpm}
            if not st.session_state.history or st.session_state.history[0]['name'] != entry['name']:
                st.session_state.history.insert(0, entry)

        title_placeholder.markdown(f'<div class="track-title">{clean_name}</div>', unsafe_allow_html=True)

        # Dashboard
        c1, c2, c3 = st.columns(3)
        c1.metric("KEY", f"{key}M")
        c2.metric("CAMELOT", camelot)
        c3.metric("TEMPO", f"{bpm} BPM")

        st.markdown("<br>", unsafe_allow_html=True)
        # Lecture audio (certains navigateurs ne lisent pas nativement le FLAC ou AIFF, 
        # mais Streamlit essaiera de l'intégrer au mieux)
        st.audio(uploaded_file)

    except Exception as e:
        st.error(f"Erreur de lecture : Ce format de fichier est corrompu ou non supporté par le serveur.")

# --- HISTORIQUE ---
st.divider()
st.markdown("### SESSION LOG")
if st.session_state.history:
    for item in st.session_state.history:
        st.markdown(f"""
            <div class="history-card">
                {item['time']} | {item['name']} | {item['key']}M | {item['camelot']} | {item['bpm']} BPM
            </div>
        """, unsafe_allow_html=True)
