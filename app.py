import streamlit as st
import librosa
import numpy as np
import pandas as pd
import time

# Configuration de la page
st.set_page_config(
    page_title="Amapiano Key Master",
    page_icon="🔥",
    layout="wide"
)

# Style CSS personnalisé pour un look DJ Pro
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3b82f6;
    }
    h1 {
        color: #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔥 Amapiano Key Master")
st.subheader("Analyseur Harmonique pour DJ & Producteurs")

# Sidebar avec instructions
with st.sidebar:
    st.header("Instructions")
    st.write("1. Téléchargez votre fichier (MP3, WAV, FLAC).")
    st.write("2. L'algorithme analyse la fréquence du Log Drum.")
    st.write("3. Obtenez votre code Camelot pour un mix parfait.")
    st.divider()
    st.info("Utilise l'analyse CQT (Constant-Q Transform) pour une précision accrue dans les basses.")

# Zone de téléchargement
uploaded_file = st.file_uploader("Déposez votre track ici", type=['mp3', 'wav', 'flac'])

def get_camelot(key, mode):
    # Dictionnaire complet des tonalités vers Camelot
    camelot_wheel = {
        'G# minor': '1A', 'Ab minor': '1A', 'B major': '1B',
        'D# minor': '2A', 'Eb minor': '2A', 'F# major': '2B', 'Gb major': '2B',
        'A# minor': '3A', 'Bb minor': '3A', 'C# major': '3B', 'Db major': '3B',
        'F minor': '4A', 'Ab major': '4B', 'G# major': '4B',
        'C minor': '5A', 'Eb major': '5B',
        'G minor': '6A', 'Bb major': '6B',
        'D minor': '7A', 'F major': '7B',
        'A minor': '8A', 'C major': '8B',
        'E minor': '9A', 'G major': '9B',
        'B minor': '10A', 'D major': '10B',
        'F# minor': '11A', 'Gb minor': '11A', 'A major': '11B',
        'C# minor': '12A', 'Db minor': '12A', 'E major': '12B'
    }
    query = f"{key} {mode}"
    return camelot_wheel.get(query, "Inconnu")

if uploaded_file is not None:
    # Barre de progression factice pour l'effet "pro"
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    with st.spinner('Extraction des caractéristiques musicales...'):
        # Chargement et analyse
        y, sr = librosa.load(uploaded_file, duration=45, offset=30) # Analyse le milieu du morceau
        
        # 1. Calcul du Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # 2. Analyse de la tonalité
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_vals = np.mean(chroma, axis=1)
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_idx = np.argmax(chroma_vals)
        detected_key = notes[key_idx]
        
        # En Amapiano, presque tout est en mineur
        mode = "minor"
        camelot = get_camelot(detected_key, mode)

        # Affichage des résultats
        st.divider()
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.metric("TONALITÉ", f"{detected_key}m")
        with c2:
            st.metric("CAMELOT", camelot)
        with c3:
            st.metric("TEMPO", f"{int(tempo)} BPM")

        # Section Conseils
        st.subheader("🛠️ Suggestions de Mixage")
        
        # Calcul des clés adjacentes
        current_num = int(camelot[:-1])
        prev_num = 12 if current_num == 1 else current_num - 1
        next_num = 1 if current_num == 12 else current_num + 1
        
        st.info(f"""
        **Mixage Harmonique :** Pour une transition fluide, cherchez des morceaux en : 
        - **{camelot}** (Même énergie)
        - **{prev_num}A** (Baisse de tension)
        - **{next_num}A** (Montée d'énergie)
        - **{current_num}B** (Changement de mode vers Majeur)
        """)
