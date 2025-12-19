import streamlit as st
import librosa
import numpy as np
import pandas as pd
from collections import Counter
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="Amapiano Master | Wood Edition", page_icon="🪵", layout="wide")

# CSS Personnalisé : Thème Boisé & Blanc
st.markdown("""
    <style>
    /* Fond principal : Couleur crème/blanc cassé */
    .stApp {
        background-color: #F9F7F2;
    }
    
    /* Titre principal avec dégradé boisé */
    h1 {
        font-family: 'Playfair Display', serif;
        font-weight: 900;
        color: #3D2B1F; /* Brun profond */
        text-align: center;
        padding-top: 20px;
    }
    
    /* Cartes de résultats : Blanc pur sur fond crème */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0D7C6;
        border-left: 5px solid #5D4037; /* Accent bois sombre */
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Style du texte des métriques */
    div[data-testid="stMetricLabel"] {
        color: #8D6E63 !important;
        font-weight: bold;
    }
    
    div[data-testid="stMetricValue"] {
        color: #3D2B1F !important;
    }

    /* Zone d'upload */
    .stFileUploader {
        background-color: #FFFFFF;
        border: 2px dashed #D7CCC8;
        border-radius: 15px;
    }

    /* Boutons et éléments interactifs */
    .stButton>button {
        background-color: #5D4037;
        color: white;
        border-radius: 8px;
        border: none;
    }

    /* Alertes et Info */
    .stInfo {
        background-color: #EFEBE9;
        color: #4E342E;
        border: 1px solid #D7CCC8;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE MUSICALE ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def get_camelot(key, mode):
    camelot_map = {
        'G# minor': '1A', 'Ab minor': '1A', 'B major': '1B', 'D# minor': '2A', 'Eb minor': '2A', 'F# major': '2B',
        'Bb minor': '3A', 'Db major': '3B', 'F minor': '4A', 'Ab major': '4B', 'C minor': '5A', 'Eb major': '5B',
        'G minor': '6A', 'Bb major': '6B', 'D minor': '7A', 'F major': '7B', 'A minor': '8A', 'C major': '8B',
        'E minor': '9A', 'G major': '9B', 'B minor': '10A', 'D major': '10B', 'F# minor': '11A', 'A major': '11B',
        'C# minor': '12A', 'E major': '12B'
    }
    return camelot_map.get(f"{key} {mode}", "1A")

def analyze_segment(y_segment, sr):
    if len(y_segment) < sr: return None
    y_harm = librosa.effects.hpss(y_segment)[0]
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    best_score = -1
    res_key, res_mode = "", ""
    for i in range(12):
        for mode, profile in [("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)]:
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, res_key, res_mode = score, NOTES[i], mode
    return (res_key, res_mode)

# --- HEADER ---
st.markdown("<h1>RICARDO_DJ228 KEY ANALYZER </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5D1037; font-style: italic;'>Analyse organique et précise de vos productions.</p>", unsafe_allow_html=True)

# Zone de dépôt
file = st.file_uploader("", type=['mp3', 'wav', 'flac'])

if file:
    with st.spinner("Analyse des fréquences boisées..."):
        y_full, sr = librosa.load(file)
        duration_mins = int(librosa.get_duration(y=y_full, sr=sr) // 60)
        
        segment_results = []
        # Analyse minute par minute
        for m in range(min(duration_mins, 8)):
            start_sample = m * 60 * sr
            end_sample = (m + 1) * 60 * sr
            res = analyze_segment(y_full[start_sample:end_sample], sr)
            if res: segment_results.append(res)
        
        votes = [f"{k} {m}" for k, m in segment_results]
        final_res = Counter(votes).most_common(1)[0][0]
        final_key, final_mode = final_res.split()
        
        tempo, _ = librosa.beat.beat_track(y=y_full, sr=sr)
        camelot = get_camelot(final_key, final_mode)

        # AFFICHAGE RÉSULTATS
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("TONALITÉ", f"{final_key} {final_mode.upper()}")
        c2.metric("NOTATION CAMELOT", camelot)
        c3.metric("TEMPO", f"{int(tempo)} BPM")

        # --- SECTION AUDIO ---
        st.divider()
        st.markdown("<h3 style='color: #3D2B1F;'>🔊 Studio de Contrôle</h3>", unsafe_allow_html=True)
        
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("**Fichier Audio**")
            st.audio(file)
        with v2:
            st.markdown(f"**Note Témoin ({final_key})**")
            note_freqs = {'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88}
            freq = note_freqs.get(final_key, 440.0)
            t = np.linspace(0, 3, int(22050 * 3), False)
            tone = 0.5 * np.sin(2 * np.pi * freq * t)
            st.audio(tone, sample_rate=22050)

        # --- CONSEILS ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"🌿 **Conseil Harmonique :** Votre morceau en **{final_key} {final_mode}** se mariera parfaitement avec des pistes en **{camelot}**. Pour une montée en énergie douce, essayez la clé suivante sur la roue Camelot.")

else:
    st.markdown("<br><br><p style='text-align: center; color: #D7CCC8;'>En attente d'un signal audio...</p>", unsafe_allow_html=True)
