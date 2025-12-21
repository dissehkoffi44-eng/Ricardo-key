import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from datetime import datetime
import io
import streamlit.components.v1 as components
from concurrent.futures import ThreadPoolExecutor
import requests  
import gc 
import os # Pour cpu_count

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | V6.2 Fast", page_icon="🎧", layout="wide")

# --- OPTIMISATION DU CHARGEMENT (90s suffisent pour la tonalité) ---
ANALYSIS_DURATION = 90 

# --- IMPORT POUR LES TAGS MP3 (MUTAGEN) ---
try:
    from mutagen.id3 import ID3, TKEY
    from mutagen.mp3 import MP3
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

# Paramètres Telegram
TELEGRAM_TOKEN = "7751365982:AAFLbeRoPsDx5OyIOlsgHcGKpI12hopzCYo"
CHAT_ID = "-1003602454394" 

if 'history' not in st.session_state: st.session_state.history = []
if 'processed_files' not in st.session_state: st.session_state.processed_files = {}
if 'order_list' not in st.session_state: st.session_state.order_list = []

def upload_to_telegram(file_buffer, filename, caption):
    try:
        file_buffer.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        files = {'document': (filename, file_buffer.read())}
        data = {'chat_id': CHAT_ID, 'caption': caption}
        response = requests.post(url, files=files, data=data, timeout=10).json()
        return response.get("ok", False)
    except: return False

# --- CSS & JS (Inchangés pour garder l'esthétique) ---
st.markdown("""<style>.stApp { background-color: #F8F9FA; } .metric-container { background: white; padding: 20px; border-radius: 15px; border: 1px solid #E0E0E0; text-align: center; height: 100%; transition: transform 0.3s; } .value-custom { font-size: 1.6em; font-weight: 800; color: #1A1A1A; } .diag-box { text-align:center; padding:10px; border-radius:10px; border:1px solid #EEE; background: white; }</style>""", unsafe_allow_html=True)

def get_sine_witness(note_mode_str, key_suffix=""):
    parts = note_mode_str.split(' ')
    note = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else "major"
    unique_id = f"playBtn_{note}_{mode}_{key_suffix}".replace("#", "sharp").replace(".", "_")
    return components.html(f"""<div style="display: flex; align-items: center; justify-content: center; gap: 10px; font-family: sans-serif;"><button id="{unique_id}" style="background: #6366F1; color: white; border: none; border-radius: 50%; width: 28px; height: 28px; cursor: pointer;">▶</button><span style="font-size: 9px; font-weight: bold; color: #666;">{note} {mode[:3].upper()}</span></div><script>const notesFreq = {{'C':261.63,'C#':277.18,'D':293.66,'D#':311.13,'E':329.63,'F':349.23,'F#':369.99,'G':392.00,'G#':415.30,'A':440.00,'A#':466.16,'B':493.88}}; const semitones = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']; let audioCtx = null; let oscillators = []; let gainNode = null; document.getElementById('{unique_id}').onclick = function() {{ if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)(); if (this.innerText === '▶') {{ this.innerText = '◼'; this.style.background = '#E74C3C'; gainNode = audioCtx.createGain(); gainNode.gain.setValueAtTime(0.05, audioCtx.currentTime); gainNode.connect(audioCtx.destination); const rootIdx = semitones.indexOf('{note}'); const intervals = ('{mode}' === 'minor') ? [0, 3, 7] : [0, 4, 7]; intervals.forEach(interval => {{ let osc = audioCtx.createOscillator(); osc.type = 'sine'; let freq = notesFreq['{note}'] * Math.pow(2, interval / 12); osc.frequency.setValueAtTime(freq, audioCtx.currentTime); osc.connect(gainNode); osc.start(); oscillators.push(osc); }}); }} else {{ oscillators.forEach(o => o.stop()); oscillators = []; this.innerText = '▶'; this.style.background = '#6366F1'; }} }};</script>""", height=40)

# --- MAPPING CAMELOT ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        return (BASE_CAMELOT_MINOR if parts[1].lower() in ['minor', 'dorian'] else BASE_CAMELOT_MAJOR).get(parts[0], "??")
    except: return "??"

def get_tagged_audio(file_buffer, key_val):
    if not MUTAGEN_AVAILABLE: return file_buffer
    try:
        file_buffer.seek(0)
        audio = MP3(io.BytesIO(file_buffer.read()))
        if audio.tags is None: audio.add_tags()
        audio.tags.add(TKEY(encoding=3, text=key_val))
        out = io.BytesIO(); audio.save(out); out.seek(0)
        return out
    except: return file_buffer

# --- ANALYSE OPTIMISÉE ---
def analyze_segment(y, sr, tuning=0.0):
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # Correction : on limite n_chroma et on utilise une méthode plus rapide
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, tuning=tuning, n_chroma=12)
    chroma_avg = np.mean(chroma, axis=1)
    
    PROFILES = {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], 
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    }
    
    res_key, best_score = "C major", -1
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, res_key = score, f"{NOTES[i]} {mode}"
    return res_key, best_score, chroma_avg

@st.cache_data(show_spinner=False)
def get_full_analysis(file_buffer):
    file_name = getattr(file_buffer, 'name', 'Unknown')
    # OPTIMISATION : On ne charge que les 90 premières secondes pour la clé
    y, sr = librosa.load(file_buffer, sr=22050, duration=ANALYSIS_DURATION, res_type='kaiser_fast')
    
    tuning_offset = librosa.estimate_tuning(y=y, sr=sr)
    
    # Séparation Percussion/Harmonique (Uniquement si nécessaire pour la clé)
    y_harm = librosa.effects.hpss(y)[0]
    
    duration = librosa.get_duration(y=y, sr=sr)
    timeline_data, votes, all_chromas = [], [], []
    
    # Analyse par blocs de 10s
    for start_t in range(0, int(duration) - 10, 10):
        y_seg = y_harm[int(start_t*sr):int((start_t+10)*sr)]
        key_seg, score_seg, chroma_vec = analyze_segment(y_seg, sr, tuning=tuning_offset)
        votes.append(key_seg)
        all_chromas.append(chroma_vec)
        timeline_data.append({"Temps": start_t, "Note": key_seg, "Confiance": round(score_seg * 100, 1)})
    
    dominante_vote = Counter(votes).most_common(1)[0][0]
    avg_chroma_global = np.mean(all_chromas, axis=0)
    
    # Synthèse globale
    best_synth_score, tonique_synth = -1, ""
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for mode, profile in {"major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]}.items():
        for i in range(12):
            score = np.corrcoef(avg_chroma_global, np.roll(profile, i))[0, 1]
            if score > best_synth_score: best_synth_score, tonique_synth = score, f"{NOTES[i]} {mode}"

    top_votes = Counter(votes).most_common(2)
    purity = int((top_votes[0][1] / len(votes)) * 100)
    
    # Tempo (sur un échantillon réduit pour la vitesse)
    tempo, _ = librosa.beat.beat_track(y=y[:int(30*sr)], sr=sr)
    
    return {
        "file_name": file_name, "vote": dominante_vote, "synthese": tonique_synth, 
        "confidence": int(best_synth_score*100), "tempo": int(float(tempo)), 
        "energy": int(np.clip(np.mean(librosa.feature.rms(y=y))*40, 1, 10)),
        "timeline": timeline_data, "purity": purity, "key_shift": (purity < 70),
        "secondary": top_votes[1][0] if len(top_votes)>1 else top_votes[0][0],
        "original_buffer": file_buffer
    }

# --- UI ---
st.markdown("<h1 style='text-align: center;'>🎧 RICARDO_DJ228 | V6.2 FAST</h1>", unsafe_allow_html=True)
files = st.file_uploader("📂 DÉPOSEZ VOS TRACKS", type=['mp3', 'wav', 'flac'], accept_multiple_files=True)

if files:
    to_proc = [f for f in files if f"{f.name}_{f.size}" not in st.session_state.processed_files]
    if to_proc:
        with st.spinner(f"Analyse ultra-rapide ({len(to_proc)} fichiers)..."):
            # Parallélisation max basée sur le CPU
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(executor.map(get_full_analysis, to_proc))
                for r in results:
                    fid = f"{r['file_name']}_{r['original_buffer'].size}"
                    cam = get_camelot_pro(r['synthese'])
                    r['saved_on_tg'] = upload_to_telegram(r['original_buffer'], f"[{cam}] {r['file_name']}", f"🔑 {cam} | 🥁 {r['tempo']} BPM")
                    st.session_state.processed_files[fid] = r
                    st.session_state.order_list.insert(0, fid)
        gc.collect()

# --- AFFICHAGE ---
for fid in st.session_state.order_list:
    res = st.session_state.processed_files[fid]
    with st.expander(f"🎵 {res['file_name']}", expanded=True):
        cam_final = get_camelot_pro(res['synthese'])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DOMINANTE", res["vote"], get_camelot_pro(res["vote"]))
        c2.metric("SYNTHÈSE", res["synthese"], cam_final)
        c3.metric("STABILITÉ", f"{res['purity']}%", "Stable" if res['purity'] > 70 else "Variable")
        c4.metric("TEMPO", f"{res['tempo']} BPM", f"E: {res['energy']}/10")
        
        st.download_button("💾 MP3 TAGGÉ", get_tagged_audio(res['original_buffer'], cam_final), f"[{cam_final}] {res['file_name']}", key=f"dl_{fid}")
        st.plotly_chart(px.line(res['timeline'], x="Temps", y="Note", title="Évolution Harmonique"), use_container_width=True)
