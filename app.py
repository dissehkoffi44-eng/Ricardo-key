import streamlit as st
import librosa
import numpy as np
import pandas as pd
from collections import Counter
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Amapiano Master | Monochrome", page_icon="⚫", layout="wide")

if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- STYLE CSS NOIR ET BLANC ---
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    
    /* Titre Application */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 200;
        letter-spacing: 5px;
        color: #FFFFFF;
        text-align: center;
        border-bottom: 1px solid #333;
        padding-bottom: 20px;
    }

    /* BLOC TITRE DE LA CHANSON (Focus de votre demande) */
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
    }

    /* Cartes de résultats */
    div[data-testid="stMetric"] {
        background-color: #000000;
        border: 1px solid #FFFFFF;
        border-radius: 0px;
        padding: 25px;
    }
    
    div[data-testid="stMetricLabel"] { color: #888888 !important; letter-spacing: 2px; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 2.5rem !important; }

    /* Historique Style Minimaliste */
    .history-card {
        background-color: #000000;
        border-bottom: 1px solid #222;
        padding: 15px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }

    /* Upload box */
    .stFileUploader { border: 1px dashed #444; border-radius: 0px; }
    
    /* Bouton Export */
    .stButton>button {
        background-color: #FFFFFF;
        color: #000000;
        border-radius: 0px;
        font-weight: bold;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE MUSICALE ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def get_camelot(key):
    camelot_map = {'G# minor': '1A', 'D# minor': '2A', 'Bb minor': '3A', 'F minor': '4A', 'C minor': '5A', 'G minor': '6A', 'D minor': '7A', 'A minor': '8A', 'E minor': '9A', 'B minor': '10A', 'F# minor': '11A', 'C# minor': '12A'}
    return camelot_map.get(f"{key} minor", "12A")

def analyze_audio(file):
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
st.markdown("<h1>RICARDO_DJ228 KEY ANALYZER</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=['mp3', 'wav', 'flac'])

if uploaded_file:
    # Nettoyage du nom de fichier pour un affichage propre
    clean_name = uploaded_file.name.replace(".mp3", "").replace(".wav", "").replace(".flac", "").replace("_", " ").upper()
    
    # AFFICHAGE DU TITRE EN GROS
    st.markdown(f'<div class="track-title">{clean_name}</div>', unsafe_allow_html=True)

    with st.spinner("ANALYSING FREQUENCIES..."):
        key, bpm = analyze_audio(uploaded_file)
        camelot = get_camelot(key)
        
        entry = {"time": datetime.datetime.now().strftime("%H:%M"), "name": clean_name, "key": key, "camelot": camelot, "bpm": bpm}
        if not st.session_state.history or st.session_state.history[0]['name'] != entry['name']:
            st.session_state.history.insert(0, entry)

    # Dashboard
    c1, c2, c3 = st.columns(3)
    c1.metric("KEY", f"{key}M")
    c2.metric("CAMELOT", camelot)
    c3.metric("TEMPO", f"{bpm} BPM")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.audio(uploaded_file)

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
