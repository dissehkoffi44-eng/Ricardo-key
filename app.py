import streamlit as st
import librosa
import numpy as np
import pandas as pd
from collections import Counter
import datetime
import io

# --- CONFIGURATION ---
st.set_page_config(page_title="Amapiano Master | Monochrome", page_icon="⚫", layout="wide")

# Initialisation de l'historique
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- STYLE CSS NOIR ET BLANC ---
st.markdown("""
    <style>
    /* Fond Noir Profond */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Titres Blanc Pur */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #FFFFFF;
        text-align: center;
    }

    /* Cartes de résultats : Fond noir, bordure blanche */
    div[data-testid="stMetric"] {
        background-color: #000000;
        border: 2px solid #FFFFFF;
        border-radius: 0px; /* Look angulaire plus moderne */
        padding: 20px;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #AAAAAA !important;
        text-transform: uppercase;
    }
    
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 800;
    }

    /* Zone d'upload */
    .stFileUploader {
        border: 1px solid #333333;
        border-radius: 0px;
        background-color: #0A0A0A;
    }

    /* Historique Style "Terminal" */
    .history-card {
        background-color: #000000;
        border-bottom: 1px solid #333333;
        padding: 15px;
        font-family: 'Courier New', Courier, monospace;
    }

    /* Boutons */
    .stButton>button {
        background-color: #FFFFFF;
        color: #000000;
        border-radius: 0px;
        font-weight: bold;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #CCCCCC;
        border: none;
    }
    
    /* Divider blanc */
    hr {
        border: 0;
        border-top: 1px solid #FFFFFF;
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
st.markdown("<h1>⚫ AMAPIANO ANALYZER PRO</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>SYSTEM READY // UPLOAD TRACK</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=['mp3', 'wav', 'flac'])

if uploaded_file:
    with st.spinner("PROCESSING..."):
        key, bpm = analyze_audio(uploaded_file)
        camelot = get_camelot(key)
        
        # Enregistrement historique
        entry = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "name": uploaded_file.name[:30],
            "key": key,
            "camelot": camelot,
            "bpm": bpm
        }
        if not st.session_state.history or st.session_state.history[0]['name'] != entry['name']:
            st.session_state.history.insert(0, entry)

    # Dashboard actuel
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("KEY", f"{key}m")
    c2.metric("CAMELOT", camelot)
    c3.metric("TEMPO", f"{bpm} BPM")

# --- HISTORIQUE TERMINAL ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("### 💾 SESSION HISTORY")

if st.session_state.history:
    # Option Export CSV
    df = pd.DataFrame(st.session_state.history)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("EXPORT LOG (.CSV)", csv, "track_log.csv", "text/csv")
    
    if st.button("CLEAR LOG"):
        st.session_state.history = []
        st.rerun()

    for item in st.session_state.history:
        st.markdown(f"""
            <div class="history-card">
                [{item['time']}] {item['name']} >> {item['key']}m // {item['camelot']} // {item['bpm']} BPM
            </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("<p style='text-align: center; color: #333;'>NO DATA LOGGED</p>", unsafe_allow_html=True)
