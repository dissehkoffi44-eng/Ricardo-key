import streamlit as st
import librosa
import numpy as np
import pandas as pd
from collections import Counter
import time

# --- CONFIGURATION LUXE ---
st.set_page_config(page_title="DJ Ricardo228 Key Master | Premium", page_icon="✨", layout="wide")

# CSS Personnalisé pour l'aspect Luxueux
st.markdown("""
    <style>
    /* Fond principal */
    .stApp {
        background: radial-gradient(circle at top right, #1e1e1e, #0a0a0a);
    }
    
    /* Titres */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
        background: -webkit-linear-gradient(#e2b04a, #9d762e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 20px;
    }
    
    /* Cartes de résultats */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(226, 176, 74, 0.3);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        transition: transform 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: rgba(226, 176, 74, 0.8);
    }

    /* Boutons et Upload */
    .stFileUploader {
        border: 2px dashed rgba(226, 176, 74, 0.2);
        border-radius: 15px;
        padding: 20px;
    }
    
    /* Texte info */
    .stInfo {
        background-color: rgba(226, 176, 74, 0.05);
        color: #e2b04a;
        border: 1px solid rgba(226, 176, 74, 0.2);
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
    return camelot_map.get(f"{key} {mode}", "1A") # Fallback to 1A for Amapiano logic

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

# --- HEADER LUXE ---
st.markdown("<h1>✨ AMAPIANO KEY MASTER PRO</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>L'intelligence artificielle au service de l'harmonie musicale.</p>", unsafe_allow_html=True)

# Zone de dépôt de fichier
file = st.file_uploader("", type=['mp3', 'wav', 'flac'])

if file:
    with st.spinner("Analyse Haute Fidélité en cours..."):
        # Analyse
        y_full, sr = librosa.load(file)
        duration_mins = int(librosa.get_duration(y=y_full, sr=sr) // 60)
        
        segment_results = []
        for m in range(min(duration_mins, 6)): # On analyse les 6 premières minutes max
            start_sample = m * 60 * sr
            end_sample = (m + 1) * 60 * sr
            res = analyze_segment(y_full[start_sample:end_sample], sr)
            if res: segment_results.append(res)
        
        votes = [f"{k} {m}" for k, m in segment_results]
        final_res = Counter(votes).most_common(1)[0][0]
        final_key, final_mode = final_res.split()
        
        tempo, _ = librosa.beat.beat_track(y=y_full, sr=sr)
        camelot = get_camelot(final_key, final_mode)

        # AFFICHAGE DES RÉSULTATS (Style Dashboard)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Tonalité", f"{final_key} {final_mode.capitalize()}")
        c2.metric("Notation Camelot", camelot)
        c3.metric("Tempo", f"{int(tempo)} BPM")

        # --- SECTION AUDITIVE LUXE ---
        st.markdown("<br><hr style='border: 0.5px solid rgba(226, 176, 74, 0.2);'>", unsafe_allow_html=True)
        st.subheader("🔊 Studio de Vérification")
        
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("**Morceau Original**")
            st.audio(file)
        
        with v2:
            st.markdown(f"**Référence Fréquentielle ({final_key})**")
            note_freqs = {'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88}
            freq = note_freqs.get(final_key, 440.0)
            t = np.linspace(0, 3, int(22050 * 3), False)
            tone = 0.5 * np.sin(2 * np.pi * freq * t)
            st.audio(tone, sample_rate=22050)

        # --- CONSEILS DE MIXAGE ---
        st.markdown("<br>", unsafe_allow_html=True)
        num = int(camelot[:-1])
        st.info(f"✨ **Expertise Harmonique :** Pour un enchaînement luxueux, privilégiez un titre en **{camelot}** ou effectuez une transition énergétique vers **{(num)%12+1}A**.")

else:
    # État vide élégant
    st.markdown("<br><br><p style='text-align: center; color: #444;'>Veuillez importer un fichier audio pour commencer l'analyse.</p>", unsafe_allow_html=True)
