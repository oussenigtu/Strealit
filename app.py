import io
import re
from pathlib import Path
import numpy as np
import streamlit as st
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

ALLOWED_EXT = {".wav", ".flac", ".ogg"}

# ---------------- Utils ----------------
def read_audio(path: Path, target_sr=16000, mono=True):
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if mono and x.ndim == 2:
        x = np.mean(x, axis=1)
    if target_sr and sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return x.astype(np.float32), sr

def to_wav_bytes(x: np.ndarray, sr: int) -> bytes:
    bio = io.BytesIO()
    sf.write(bio, x, sr, format="WAV", subtype="PCM_16")
    bio.seek(0)
    return bio.read()

def rms(x):  # simple RMS
    return np.sqrt(np.maximum(1e-12, np.mean(x * x, dtype=np.float64)))

def align_pair(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n]

def snr_proxy(enh, noisy):
    """SNR sans clean: signal≈enh, bruit≈noisy-enh"""
    enh, noisy = align_pair(enh, noisy)
    resid = noisy - enh
    num = np.sum(enh**2) + 1e-12
    den = np.sum(resid**2) + 1e-12
    return 10.0 * np.log10(num / den)

def snr_improvement(enh, noisy):
    """ΔSNR avec référence interne commune (noisy - enh). Purement indicatif."""
    enh, noisy = align_pair(enh, noisy)
    resid = noisy - enh
    s_noisy = 10.0 * np.log10((np.sum(noisy**2) + 1e-12) / (np.sum(resid**2) + 1e-12))
    s_enh   = 10.0 * np.log10((np.sum(enh**2)   + 1e-12) / (np.sum(resid**2) + 1e-12))
    return s_enh - s_noisy, s_noisy, s_enh

def mel_spectrogram(x, sr, n_fft=1024, hop=256, n_mels=80):
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0)
    return librosa.power_to_db(S, ref=np.max)

def list_audio_files(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in ALLOWED_EXT])

DEFAULT_TOKENS_NOISY = {"noisy", "noise", "bruite", "bruitee", "bruitees"}
DEFAULT_TOKENS_ENH   = {"enh", "enhanced", "denoise", "denoised", "rehausse", "rehaussee", "denoiser"}

def normalize_name(name: str, extra_tokens: set[str]):
    stem = Path(name).stem.lower()
    tokens = re.split(r"[_\-\s]+", stem)
    filtered = [t for t in tokens if t and t not in extra_tokens]
    norm = "_".join(filtered)
    # Compacte doublons de séparateurs résiduels
    norm = re.sub(r"_+", "_", norm).strip("_")
    return norm

def build_index(files, strategy: str, token_set: set[str]):
    """
    strategy: 'strict' => clé = stem
              'normalized' => clé = normalize_name(stem, token_set)
    Renvoie dict key -> Path
    """
    index = {}
    for p in files:
        stem = p.stem
        key = stem if strategy == "strict" else normalize_name(stem, token_set)
        if key in index:
            # collision : on garde le premier et liste les doublons
            # (on pourrait mieux gérer selon timestamp/tri, mais restons simples)
            pass
        else:
            index[key] = p
    return index

# ---------------- UI ----------------
st.set_page_config(page_title="Map & Évalue (Bruit vs Réhausse)", layout="wide")
st.title("Évaluation : Dossier **Bruit** ↔ Dossier **Réhaussé**")
st.caption("Charge deux dossiers distincts, mappe les fichiers, calcule SNR proxy & ΔSNR, affiche waveforms, mél-spectrogrammes et permet l’écoute.")

with st.sidebar:
    st.header("Paramètres")
    target_sr = st.number_input("Resample (Hz)", 8000, 48000, 16000, step=1000)
    n_fft = st.number_input("n_fft (Mél)", 256, 4096, 1024, step=256)
    hop = st.number_input("hop_length (Mél)", 64, 2048, 256, step=64)
    n_mels = st.number_input("n_mels (Mél)", 20, 256, 80, step=4)
    strategy = st.radio("Stratégie d’appariement", ["Basename strict", "Basename normalisé"], index=1)

    st.subheader("Normalisation (si 'normalisé')")
    extra_noisy = st.text_input("Tokens à retirer côté **Bruit** (séparés par virgules)", value="")
    extra_enh = st.text_input("Tokens à retirer côté **Réhausse** (séparés par virgules)", value="")
    tok_noisy = DEFAULT_TOKENS_NOISY | {t.strip().lower() for t in extra_noisy.split(",") if t.strip()}
    tok_enh   = DEFAULT_TOKENS_ENH   | {t.strip().lower() for t in extra_enh.split(",") if t.strip()}

st.subheader("Chemins des dossiers")
if "noisy_dir" not in st.session_state:
    st.session_state.noisy_dir = ""
if "enh_dir" not in st.session_state:
    st.session_state.enh_dir = ""
col1, col2 = st.columns(2)
with col1:
    noisy_dir = st.text_input("Dossier **Bruit**", value=st.session_state.noisy_dir, key="noisy_dir")
with col2:
    enh_dir = st.text_input("Dossier **Réhausse**", value=st.session_state.enh_dir, key="enh_dir")

if noisy_dir and enh_dir:
    noisy_dir = Path(noisy_dir)
    enh_dir = Path(enh_dir)
    if not noisy_dir.exists():
        st.error(f"Dossier bruit introuvable: {noisy_dir}")
    if not enh_dir.exists():
        st.error(f"Dossier réhausse introuvable: {enh_dir}")

    if noisy_dir.exists() and enh_dir.exists():
        noisy_files = list_audio_files(noisy_dir)
        enh_files = list_audio_files(enh_dir)

        st.markdown(f"- Fichiers **Bruit** trouvés: **{len(noisy_files)}**")
        st.markdown(f"- Fichiers **Réhausse** trouvés: **{len(enh_files)}**")

        strat_key = "strict" if strategy == "Basename strict" else "normalized"
        idx_noisy = build_index(noisy_files, strat_key, tok_noisy)
        idx_enh = build_index(enh_files, strat_key, tok_enh)

        keys_both = sorted(set(idx_noisy.keys()) & set(idx_enh.keys()))
        keys_only_noisy = sorted(set(idx_noisy.keys()) - set(idx_enh.keys()))
        keys_only_enh = sorted(set(idx_enh.keys()) - set(idx_noisy.keys()))

        st.markdown(f"**Paires appariées : {len(keys_both)}**")
        with st.expander("Sans correspondance côté Réhausse"):
            if keys_only_noisy:
                st.write(", ".join(keys_only_noisy))
            else:
                st.write("—")
        with st.expander("Sans correspondance côté Bruit"):
            if keys_only_enh:
                st.write(", ".join(keys_only_enh))
            else:
                st.write("—")

        # ---- Calcul SNR table
        rows = []
        for k in keys_both:
            p_noisy = idx_noisy[k]
            p_enh = idx_enh[k]
            try:
                x_noisy, sr_n = read_audio(p_noisy, target_sr=target_sr, mono=True)
                x_enh, sr_e = read_audio(p_enh, target_sr=target_sr, mono=True)
                sr = sr_n if sr_n == sr_e else target_sr
                snr_p = snr_proxy(x_enh, x_noisy)
                dsnr, s_noisy, s_enh = snr_improvement(x_enh, x_noisy)
                rows.append({
                    "clé": k,
                    "noisy": str(p_noisy.relative_to(noisy_dir)),
                    "enh": str(p_enh.relative_to(enh_dir)),
                    "sr": sr,
                    "SNR proxy [dB]": np.round(snr_p, 2),
                    "ΔSNR [dB]": np.round(dsnr, 2),
                    "SNR(noisy, ref int) [dB]": np.round(s_noisy, 2),
                    "SNR(enh, ref int) [dB]": np.round(s_enh, 2),
                })
            except Exception as e:
                st.warning(f"Erreur lecture/calcul ({k}): {e}")

        if rows:
            st.subheader("Tableau SNR")
            st.dataframe(rows, use_container_width=True)

            # Export CSV
            import pandas as pd
            df = pd.DataFrame(rows)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Export CSV", data=csv, file_name="snr_table.csv", mime="text/csv")

            # ---- Visualisation détaillée
            st.subheader("Visualisation détaillée")
            sel = st.selectbox("Choisir une paire", [r["clé"] for r in rows], index=0)
            p_noisy = idx_noisy[sel]
            p_enh = idx_enh[sel]
            x_noisy, sr_n = read_audio(p_noisy, target_sr=target_sr, mono=True)
            x_enh, sr_e = read_audio(p_enh, target_sr=target_sr, mono=True)
            sr = sr_n if sr_n == sr_e else target_sr

            dur = min(len(x_noisy), len(x_enh)) / sr
            st.write(f"Durée alignée ≈ **{dur:.2f} s** — sr = **{sr} Hz**")

            t0, t1 = st.slider("Fenêtre temporelle (zoom)", 0.0, float(dur), (0.0, float(min(3.0, dur))), step=0.01)
            s0, s1 = int(t0 * sr), int(t1 * sr)
            seg_noisy, seg_enh = x_noisy[s0:s1], x_enh[s0:s1]
            t_noisy = np.arange(len(seg_noisy)) / sr + t0
            t_enh = np.arange(len(seg_enh)) / sr + t0

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Waveform — Bruit**  \n`{p_noisy.name}`")
                fig = plt.figure(figsize=(8, 3))
                plt.plot(t_noisy, seg_noisy)
                plt.xlabel("Temps [s]"); plt.ylabel("Amp")
                plt.tight_layout()
                st.pyplot(fig)
            with c2:
                st.markdown(f"**Waveform — Réhausse**  \n`{p_enh.name}`")
                fig = plt.figure(figsize=(8, 3))
                plt.plot(t_enh, seg_enh)
                plt.xlabel("Temps [s]"); plt.ylabel("Amp")
                plt.tight_layout()
                st.pyplot(fig)

            st.markdown("**Mél-spectrogrammes (vue complète)**")
            c3, c4 = st.columns(2)
            with c3:
                S_n = mel_spectrogram(x_noisy, sr, n_fft=n_fft, hop=hop, n_mels=n_mels)
                fig = plt.figure(figsize=(8, 3.6))
                librosa.display.specshow(S_n, sr=sr, hop_length=hop, x_axis="time", y_axis="mel")
                plt.colorbar(format="%+2.0f dB")
                plt.title("Mél — Bruit")
                plt.tight_layout()
                st.pyplot(fig)
            with c4:
                S_e = mel_spectrogram(x_enh, sr, n_fft=n_fft, hop=hop, n_mels=n_mels)
                fig = plt.figure(figsize=(8, 3.6))
                librosa.display.specshow(S_e, sr=sr, hop_length=hop, x_axis="time", y_axis="mel")
                plt.colorbar(format="%+2.0f dB")
                plt.title("Mél — Réhausse")
                plt.tight_layout()
                st.pyplot(fig)

            st.markdown("**Écoute**")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.caption("Bruit")
                st.audio(to_wav_bytes(x_noisy, sr), format="audio/wav")
            with cc2:
                st.caption("Réhausse")
                st.audio(to_wav_bytes(x_enh, sr), format="audio/wav")

            st.markdown("**Mesures (paire sélectionnée)**")
            snr_p = snr_proxy(x_enh, x_noisy)
            dsnr, s_noisy, s_enh = snr_improvement(x_enh, x_noisy)
            st.subheader("Mesures SNR")
            colm1,colm2=st.columns(2)
            with colm1:
                st.metric("SNR proxy [dB]", f"{snr_p:.2f}")
            with colm2:
                st.metric("ΔSNR [dB]", f"{dsnr:.2f}")
            with st.expander("Détails du calcul"):
                st.write(
                    "Sans clean de référence, on approxime :\n"
                    "- signal ≈ **réhaussé**\n"
                    "- bruit ≈ **bruit - réhaussé**\n\n"
                    "Ces valeurs sont **indicatives** (attention aux interprétations)."
                )
        else:
            st.info("Aucune paire exploitable. Ajuste la stratégie d’appariement ou les tokens.")
else:
    st.info("Renseigne les chemins des deux dossiers pour commencer.")
