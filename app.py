# RCDJ228 SNIPER M3 - VERSION ULTIME "ROOT & TRIAD"
import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import os
import requests
import gc
import json
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter
from datetime import datetime
from pydub import AudioSegment

# --- FORCE FFMEG PATH (WINDOWS FIX) ---
if os.path.exists(r'C:\ffmpeg\bin'):
    os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 SNIPER M3", page_icon="🎯", layout="wide")

# Récupération des secrets
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

PROFILES = {
    "sniper_triads": {
        "major": [1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0],
        "minor": [1.0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0]
    },
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    }
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-card { 
        padding: 40px; border-radius: 30px; text-align: center; color: white; 
        border: 1px solid rgba(99, 102, 241, 0.3); box-shadow: 0 15px 45px rgba(0,0,0,0.6);
        margin-bottom: 20px;
    }
    .file-header {
        background: #1f2937; color: #10b981; padding: 10px 20px; border-radius: 10px;
        font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 10px;
        border-left: 5px solid #10b981;
    }
    .root-hint {
        background: rgba(16, 185, 129, 0.1); color: #10b981; padding: 5px 12px;
        border-radius: 20px; font-size: 0.8em; border: 1px solid #10b981;
    }
    .metric-box {
        background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d;
        height: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE CALCUL ---
def apply_sniper_filters(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    nyq = 0.5 * sr
    b, a = butter(4, [80/nyq, 5000/nyq], btype='band')
    return lfilter(b, a, y_harm)

def get_root_note_pyin(y, sr):
    """Analyse de la fréquence fondamentale (PYIN) pour identifier la note d'ancrage."""
    # On analyse un segment central de 20s pour gagner en rapidité
    start_sample = len(y) // 2
    end_sample = start_sample + (sr * 20)
    y_chunk = y[start_sample:min(end_sample, len(y))]
    
    f0, voiced_flag, voiced_probs = librosa.pyin(y_chunk, 
                                                 fmin=librosa.note_to_hz('C2'), 
                                                 fmax=librosa.note_to_hz('C5'), 
                                                 sr=sr, hop_length=1024)
    # On ne garde que les notes avec une probabilité de confiance > 80%
    valid_f0 = f0[voiced_flag & (voiced_probs > 0.8)]
    if len(valid_f0) == 0: return None
    
    notes = librosa.hz_to_note(valid_f0)
    clean_notes = [n.replace(n[-1], '') for n in notes] # On enlève l'octave
    return Counter(clean_notes).most_common(1)[0][0]

def solve_key_sniper(chroma_vector, bass_vector, root_hint=None):
    best_overall_score = -1
    best_key = "Unknown"
    
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)
    
    for p_name, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            for i in range(12):
                note_name = NOTES_LIST[i]
                reference = np.roll(p_data[mode], i)
                score = np.corrcoef(cv, reference)[0, 1]
                
                # --- BOOST PYIN (ROOT VERIFIER) ---
                if root_hint and note_name == root_hint:
                    score += 0.25 # Bonus massif si la fondamentale PYIN correspond à la tonale

                # --- BOOST SNIPER TRIAD ---
                if p_name == "sniper_triads": score *= 1.25 

                # Logique mineure
                if mode == "minor":
                    dom_idx, leading_tone = (i + 7) % 12, (i + 11) % 12
                    if cv[dom_idx] > 0.45 and cv[leading_tone] > 0.35: score *= 1.35 
                
                # Priorité Basse
                if bv[i] > 0.6: score += (bv[i] * 0.25)
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_key = f"{note_name} {mode}"
                    
    return {"key": best_key, "score": best_overall_score}

def process_audio_precision(file_bytes, file_name, _progress_callback=None):
    try:
        with io.BytesIO(file_bytes) as buf:
            y, sr = librosa.load(buf, sr=22050, mono=True)
    except: return None

    duration = librosa.get_duration(y=y, sr=sr)
    y_filt = apply_sniper_filters(y, sr)
    
    # ÉTAPE 1 : Root Sniper (PYIN)
    if _progress_callback: _progress_callback(10, "Extraction de la Root Note (PYIN)...")
    root_hint = get_root_note_pyin(y, sr)
    
    # ÉTAPE 2 : Analyse Harmonique par Segments
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    step, timeline, votes = 2, [], Counter()
    segments = list(range(0, max(1, int(duration) - step), 2))
    
    for idx, start in enumerate(segments):
        if _progress_callback:
            prog = 15 + int((idx / len(segments)) * 80)
            _progress_callback(prog, f"Analyse Sniper : {start}s")

        idx_s, idx_e = int(start * sr), int((start + step) * sr)
        seg = y_filt[idx_s:idx_e]
        if len(seg) < 1000 or np.max(np.abs(seg)) < 0.01: continue
        
        c_raw = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=24, bins_per_octave=24)
        c_avg = np.mean((c_raw[::2, :] + c_raw[1::2, :]) / 2, axis=1)
        
        # Basses (filtre passe-bas direct sur segment)
        nyq = 0.5 * sr
        b, a = butter(2, 150/nyq, btype='low')
        b_seg = np.mean(librosa.feature.chroma_cqt(y=lfilter(b, a, y[idx_s:idx_e]), sr=sr, n_chroma=12), axis=1)
        
        res = solve_key_sniper(c_avg, b_seg, root_hint=root_hint)
        weight = 3.0 if (start < 10 or start > (duration - 15)) else 1.0
        votes[res['key']] += int(res['score'] * 100 * weight)
        timeline.append({"Temps": start, "Note": res['key'], "Conf": res['score']})

    if not votes: return None

    most_common = votes.most_common(2)
    final_key = most_common[0][0]
    final_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == final_key]) * 100)
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    res_obj = {
        "key": final_key, "camelot": CAMELOT_MAP.get(final_key, "??"),
        "conf": min(final_conf, 99), "tempo": int(float(tempo)),
        "root_hint": root_hint, "name": file_name, "timeline": timeline,
        "tuning": round(440 * (2**(tuning/12)), 1),
        "chroma": np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1).tolist()
    }
    
    # Telegram 
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            msg = f"🎯 *SNIPER M3* \n📄 `{file_name}`\n🎹 *{final_key.upper()}* ({res_obj['camelot']})\n✅ Confiance: {res_obj['conf']}%"
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'})
        except: pass

    return res_obj
    def get_chord_js(button_id, key_name):
    # Mapping des notes vers les fréquences (Hz)
    freqs = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63, 'F': 349.23,
        'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
    }
    
    note, mode = key_name.split()
    root = freqs[note]
    
    # Calcul des fréquences de la triade (Fondamentale, Tierce, Quinte)
    if mode == 'major':
        chord = [root, root * 1.25, root * 1.5] # Tierce majeure
    else:
        chord = [root, root * 1.189, root * 1.5] # Tierce mineure

    return f"""
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        document.getElementById("{button_id}").addEventListener("click", () => {{
            {json.dumps(chord)}.forEach(f => {{
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.frequency.value = f;
                osc.type = "sine";
                gain.gain.setValueAtTime(0.1, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.5);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start();
                osc.stop(ctx.currentTime + 1.5);
            }});
        }});
    """

# --- INTERFACE ---
st.title("🎯 RCDJ228 SNIPER M3")
files = st.file_uploader("📂 Audio", type=['mp3','wav','m4a','flac'], accept_multiple_files=True)

if files:
    for i, f in enumerate(reversed(files)):
        with st.status(f"Analyse Sniper : {f.name}...") as status:
            prog_bar = st.progress(0)
            data = process_audio_precision(f.getvalue(), f.name, _progress_callback=lambda v, m: prog_bar.progress(v))
            status.update(label=f"✅ {f.name} Terminé", state="complete")
        
        if data:
            st.markdown(f"<div class='file-header'>📊 {data['name']}</div>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="report-card" style="background:linear-gradient(135deg, #1e293b, #0f172a);">
                    <div style="text-align:right"><span class="root-hint">Root Hint: {data['root_hint']}</span></div>
                    <h1 style="font-size:6em; margin:0; font-weight:900;">{data['key'].upper()}</h1>
                    <p style="font-size:1.5em; opacity:0.8;">{data['camelot']} | CONFIANCE: {data['conf']}%</p>
                </div>""", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2em; color:#10b981;'>{data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br><span style='font-size:2em; color:#58a6ff;'>{data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)
            with c3:
                bid = f"pl_{i}"
                components.html(f"""<button id="{bid}" style="width:100%; height:90px; background:#4F46E5; color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold;">🔊 TESTER L'ACCORD</button>
                    <script>{get_chord_js(bid, data['key'])}</script>""", height=100)
            
            # Graphiques
            g1, g2 = st.columns([2,1])
            with g1:
                fig = px.line(pd.DataFrame(data['timeline']), x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER})
                fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)
            with g2:
                fig_rd = go.Figure(data=go.Scatterpolar(r=data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
                fig_rd.update_layout(template="plotly_dark", height=250, polar=dict(radialaxis=dict(visible=False)), margin=dict(l=20,r=20,t=20,b=20))
                st.plotly_chart(fig_rd, use_container_width=True)
            st.markdown("<hr style='opacity:0.2'>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Sniper Control")
    if st.button("🗑️ Vider le cache"):
        st.cache_data.clear()
        st.rerun()
