import torch
import numpy as np
import math
from transformers import AutoTokenizer

# --- Parámetros del Watermarking ---
# CLAVE SECRETA (DEBE COINCIDIR CON EL GENERADOR)
SECRET_SEED = 42
# Umbral de detección (fracción de tokens verdes para considerar watermarkeado)
DETECTION_THRESHOLD = 0.55

def build_greenlist(vocab_size: int, secret_seed: int) -> set:
    """
    Genera la Greenlist (Lista Verde) pseudoaleatoria.
    """
    rng = np.random.default_rng(secret_seed)
    indices = np.arange(vocab_size)
    rng.shuffle(indices)
    cutoff = int(0.5 * vocab_size)
    return set(indices[:cutoff])

def detect_watermark(text: str, tokenizer, greenlist: set, threshold: float) -> tuple:
    """
    Implementa la función de puntuación (ψ) para medir la evidencia de watermarking.
    """
    # 1. Tokenización
    tokens = tokenizer.encode(text, return_tensors='pt')[0].tolist()
    N = len(tokens)
    if N < 50:
        return False, 0.0, 0.0, "Longitud insuficiente para significancia estadística."

    # 2. Conteo de tokens verdes
    green_count = sum(1 for t in tokens if t in greenlist)

    # 3. Cálculo de la fracción de tokens verdes (Score)
    fraction = green_count / N

    # 4. Cálculo del Z-score
    # Desviación estándar del promedio de tokens bajo hipótesis nula (p=0.5): sqrt(0.25/N)
    std_dev_fraction = math.sqrt(0.25 / N)
    z_score = (fraction - 0.5) / std_dev_fraction

    # 5. Decisión
    is_watermarked = fraction > threshold

    result_str = "Watermarkeado" if is_watermarked else "No Watermarkeado (o Texto Humano)"

    return is_watermarked, fraction, z_score, result_str

def run_detection():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = len(tokenizer)

    # Reconstruir la Greenlist usando la clave secreta
    greenlist = build_greenlist(vocab_size, SECRET_SEED)
    print(f"🔑 Detector inicializado con clave secreta: {SECRET_SEED}")
    print("-" * 50)

    files_to_test = ["watermarked_content.txt", "normal_content.txt"]

    for file_name in files_to_test:
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                content = f.read()

            is_wm, frac, z_score, result = detect_watermark(content, tokenizer, greenlist, DETECTION_THRESHOLD)

            print(f"--- Análisis del archivo: **{file_name}** ({len(content.split())} palabras) ---")
            print(f"**Resultado de la Detección:** {result}")
            print(f"Fracción de Tokens Verdes (Score): **{frac:.4f}**")
            print(f"Z-score: **{z_score:.2f}** (Valor > 4.0 indica alta confianza)")
            print(f"Comprobación: {frac:.4f} > {DETECTION_THRESHOLD} ({is_wm})")
            print("-" * 50)

        except FileNotFoundError:
            print(f"⚠️ Archivo no encontrado: {file_name}. Ejecuta primero el Script 1 para generar el contenido.")
            print("-" * 50)

if __name__ == "__main__":
    run_detection()