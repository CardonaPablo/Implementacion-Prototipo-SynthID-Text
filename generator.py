import torch
import numpy as np
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Parámetros del Watermarking ---
SECRET_SEED = 42
WATERMARK_STRENGTH = 2.0
MAX_LENGTH = 200

def build_greenlist(vocab_size: int, secret_seed: int) -> set:
    """
    Genera la Greenlist (Lista Verde) pseudoaleatoria, 50% del vocabulario.
    """
    rng = np.random.default_rng(secret_seed)
    indices = np.arange(vocab_size)
    rng.shuffle(indices)
    cutoff = int(0.5 * vocab_size)
    return set(indices[:cutoff])

def watermark_logits(logits: torch.Tensor, greenlist: set, strength: float) -> torch.Tensor:
    """
    Modifica los logits (puntuaciones) del LLM para sesgar la selección
    hacia los tokens de la Greenlist. (Logits deben ser 1D: [VOCAB_SIZE])
    """
    # boost toma la forma de los logits [VOCAB_SIZE]
    boost = torch.zeros_like(logits)

    # Creamos un tensor con los índices de los tokens verdes
    green_idx_tensor = torch.tensor(list(greenlist), dtype=torch.long, device=logits.device)

    # Aplicar el incremento (boost) a los logits de los tokens "verdes"
    # **CORRECCIÓN DE INDEXACIÓN:** Ahora logits y boost son 1D, permitiendo la indexación directa.
    boost[green_idx_tensor] = strength
    return logits + boost

def generate_watermarked_text(prompt: str, apply_watermark: bool = True) -> str:
    """
    Genera texto utilizando GPT-2, con o sin watermarking.
    """
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    vocab_size = len(tokenizer)

    # Generación de la Greenlist
    greenlist = build_greenlist(vocab_size, SECRET_SEED)

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output_ids = []
    current_ids = input_ids

    print(f"⚙️ Generando texto (Watermark: {apply_watermark})...")

    # Bucle de generación token por token
    for _ in range(MAX_LENGTH):
        with torch.no_grad():
            # Obtener los logits del siguiente token (pLM(·|x<t))
            outputs = model(current_ids)
            # **CORRECCIÓN:** Usamos .squeeze(0) para convertir de [1, VOCAB_SIZE] a [VOCAB_SIZE]
            next_token_logits = outputs.logits[:, -1, :].squeeze(0)

        if apply_watermark:
            # Aplicar la modificación de watermarking
            watermarked_logits = watermark_logits(next_token_logits, greenlist, WATERMARK_STRENGTH)
        else:
            watermarked_logits = next_token_logits

        # Muestreo
        probs = torch.softmax(watermarked_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        output_ids.append(next_token_id.item())
        current_ids = torch.cat([current_ids, next_token_id.view(1, 1)], dim=-1)

    return tokenizer.decode(output_ids, skip_special_tokens=True)

if __name__ == "__main__":
    PROMPT = "Cierta parte de la población cree que las estaciones del año son fijas para todos los países, esto por la hegemonía multimedia que tiene Estados Unidos, donde el invierno se da en diciembre y el verano en marzo. sin embargo, las estaciones cambian con cada latitud"

    # 1. Generar texto Watermarkeado
    watermarked_text = generate_watermarked_text(PROMPT, apply_watermark=True)
    with open("watermarked_content.txt", "w", encoding="utf-8") as f:
        f.write(watermarked_text)
    print("\n--- Texto Watermarkeado Generado (watermarked_content.txt) ---")
    print(watermarked_text[:500] + "...")
    print(f"✅ Texto watermarkeado guardado en 'watermarked_content.txt'")

    # 2. Generar texto Normal (sin Watermark)
    normal_text = generate_watermarked_text(PROMPT, apply_watermark=False)
    with open("normal_content.txt", "w", encoding="utf-8") as f:
        f.write(normal_text)
    print("\n--- Texto Normal Generado (normal_content.txt) ---")
    print(normal_text[:500] + "...")
    print(f"✅ Texto normal guardado en 'normal_content.txt'")