import streamlit as st
import librosa
import numpy as np
import pandas as pd
from collections import Counter

# --- CONFIGURATION ---
st.set_page_config(page_title="Amapiano Key Master Pro", page_icon="🎼", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTES MUSICALES ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# --- FONCTIONS TECHNIQUES ---

def get_camelot(key, mode):
    camelot_map = {
        'G# minor': '1A', 'Ab minor': '1A', 'B major': '1B',
        'D# minor': '2A', 'Eb minor': '2A', 'F# major': '2B', 'Gb major': '2B',
        'Bb minor': '3A', 'A# minor': '3A', 'Db major': '3B', 'C# major': '3B',
        'F minor': '4A', 'Ab major': '4B', 'C minor': '5A', 'Eb major': '5B',
        'G minor': '6A', 'Bb major': '6B', 'D minor': '7A', 'F major': '7B',
        'A minor': '8A', 'C major': '8B', 'E minor': '9A', 'G major': '9B',
        'B minor': '10A', 'D major': '10B', 'F# minor': '11A', 'A major': '11B',
        'C# minor': '12A', 'Db minor': '12A', 'E major': '12B'
    }
    return camelot_map.get(f"{key} {mode}", "Inconnu")

def generate_tone(note_name):
    freqs = {'C': 261, 'C#': 277, 'D': 293, 'D#': 311, 'E': 329, 'F': 349, 
             'F#': 369, 'G': 392, 'G#': 415, 'A': 440, 'A#': 466, 'B': 493}
    sr = 22050
    t = np.linspace(0, 2, int(sr * 2), False)
    return 0.5 * np.sin(2 * np.pi * freqs[note_name] * t)

def analyze_segment(y_segment, sr):
    """Analyse un bloc audio spécifique avec séparation harmonique"""
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

# --- INTERFACE ---
st.title("🎼 Amapiano Key Master Pro")
st.markdown("### Analyse multi-segmentaire haute précision")

file = st.file_uploader("Charger un morceau", type=['mp3', 'wav', 'flac'])

if file:
    with st.spinner("Analyse minute par minute en cours..."):
        # 1. Chargement complet pour le calcul du BPM et découpage
        y_full, sr = librosa.load(file)
        duration_mins = int(librosa.get_duration(y=y_full, sr=sr) // 60)
        
        # 2. Analyse par blocs de 60 secondes
        segment_results = []
        for m in range(duration_mins):
            start_sample = m * 60 * sr
            end_sample = (m + 1) * 60 * sr
            res = analyze_segment(y_full[start_sample:end_sample], sr)
            if res: segment_results.append(res)
        
        # 3. Vote majoritaire pour la précision finale
        votes = [f"{k} {m}" for k, m in segment_results]
        final_res = Counter(votes).most_common(1)[0][0]
        final_key, final_mode = final_res.split()
        
        # 4. BPM
        tempo, _ = librosa.beat.beat_track(y=y_full, sr=sr)
        camelot = get_camelot(final_key, final_mode)

        # AFFICHAGE
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("TONALITÉ (VOTE MAJORITAIRE)", f"{final_key} {final_mode}")
        c2.metric("CODE CAMELOT", camelot)
        c3.metric("TEMPO MOYEN", f"{int(tempo)} BPM")

        # Détails de l'analyse par segment
        with st.expander("Voir le détail de l'analyse par minute"):
            for idx, r in enumerate(segment_results):
                st.write(f"Minute {idx+1} : **{r[0]} {r[1]}**")

        st.divider()
        
        # VÉRIFICATION
        st.subheader("🔊 Vérification Auriculaire")
        v1, v2 = st.columns(2)
        with v1:
            st.write("Le morceau :")
            st.audio(file)
        with v2:
            st.write(f"Note de référence ({final_key}) :")
            st.audio(generate_tone(final_key), sample_rate=22050)

        # CONSEILS
        st.subheader("🔀 Guide de Transition")
        num = int(camelot[:-1])
        st.info(f"Transitions parfaites pour **{camelot}** : \n"
                f"- **{num}A** (Équilibre) \n"
                f"- **{(num)%12+1}A** (Augmenter l'énergie) \n"
                f"- **{(num-2)%12+1}A** (Relâcher la pression)")
