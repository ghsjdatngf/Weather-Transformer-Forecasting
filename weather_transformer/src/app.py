import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd

from src import config
from src.tokenizer import WeatherTokenizer
from src.models.weather_transformer import WeatherTransformerLM

st.set_page_config(
    page_title="Weather Transformer",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model_config = checkpoint["config"]

    tokenizer = WeatherTokenizer.load(config.VOCAB_PATH)
    model = WeatherTransformerLM(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, device, model_config


def predict_next_day_category(model, tokenizer, device, model_config, recent_days_texts):
    context_text = " ".join(recent_days_texts)
    ids = tokenizer.encode_fixed(context_text, model_config["max_seq_len"])
    input_tensor = torch.tensor([ids], dtype=torch.long).to(device)

    with torch.no_grad():
        temp_logits, hum_logits = model.classify(input_tensor)
        temp_probs = F.softmax(temp_logits, dim=-1).squeeze(0)
        hum_probs = F.softmax(hum_logits, dim=-1).squeeze(0)

    return {
        "predicted_temperature_category": config.INV_TEMP_LABELS[temp_probs.argmax().item()],
        "temperature_confidence": {
            config.INV_TEMP_LABELS[i]: round(p.item(), 4) for i, p in enumerate(temp_probs)
        },
        "predicted_humidity_category": config.INV_HUMIDITY_LABELS[hum_probs.argmax().item()],
        "humidity_confidence": {
            config.INV_HUMIDITY_LABELS[i]: round(p.item(), 4) for i, p in enumerate(hum_probs)
        },
    }


def generate_text(model, tokenizer, device, model_config, prompt, max_new_tokens=40, temperature=1.0):
    ids = tokenizer.encode(prompt, add_special_tokens=True)[:-1]
    max_len = config.MAX_SEQ_LEN

    for _ in range(max_new_tokens):
        context = ids[-max_len:]
        input_tensor = torch.tensor([context], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
        next_token_logits = logits[0, -1, :] / max(temperature, 1e-6)
        next_id = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).item()
        ids.append(next_id)
        if next_id == tokenizer.vocab["<eos>"]:
            break

    return tokenizer.decode(ids)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/partly-cloudy-day.png", width=64)
    st.markdown("### Weather Transformer")
    st.caption("From-scratch Decoder-only Transformer")
    st.divider()

    with st.spinner("Loading model..."):
        model, tokenizer, device, model_config = load_trained_model()

    with open(config.THRESHOLDS_PATH) as f:
        thresholds = json.load(f)
    window = thresholds.get("window_days", config.WINDOW_DAYS)

    st.success(f"Model ready on **{str(device).upper()}**")
    st.markdown(f"**Lookback window:** {window} days")
    st.markdown(f"**Parameters:** {model.count_parameters():,}")
    st.divider()
    st.markdown("**Test-set performance**")
    st.markdown("| Metric | Score |")
    st.markdown("|--------|-------|")
    st.markdown("| Temp Accuracy | 87.9% |")
    st.markdown("| Humidity Accuracy | 74.0% |")
    st.markdown("| Perplexity | 2.65 |")
    st.divider()
    st.caption("Pakistan weather · 2007–2026 · ~7117 days")


st.markdown('<p class="main-header">🌤️ Weather Transformer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Next-day temperature & humidity category forecasting '
    'using a decoder-only Transformer trained from scratch</p>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Forecast", "Text Generation", "ℹAbout"])

with tab1:
    st.markdown("### Next-Day Category Forecast")
    st.write(
        f"The model reads the last **{window} days** of weather text from the "
        "held-out test set and predicts tomorrow's temperature & humidity category."
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button(" Predict Next Day", type="primary", use_container_width=True)

    if predict_clicked:
        test_df = pd.read_csv(config.TEST_LM_CSV)
        recent_texts = test_df["text_sequence"].iloc[-window:].tolist()

        with st.spinner("Running inference..."):
            result = predict_next_day_category(
                model, tokenizer, device, model_config, recent_texts
            )

        st.divider()

        # Big prediction cards
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size:0.9rem;opacity:0.85;">Temperature Category</div>'
                f'<div style="font-size:2rem;font-weight:700;margin-top:0.3rem;">'
                f'{result["predicted_temperature_category"]}</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card-green">'
                f'<div style="font-size:0.9rem;opacity:0.85;">Humidity Category</div>'
                f'<div style="font-size:2rem;font-weight:700;margin-top:0.3rem;">'
                f'{result["predicted_humidity_category"]}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.markdown("#### Confidence Breakdown")

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Temperature**")
            for label, conf in result["temperature_confidence"].items():
                st.progress(conf, text=f"{label}:  {conf:.1%}")
        with cc2:
            st.markdown("**Humidity**")
            for label, conf in result["humidity_confidence"].items():
                st.progress(conf, text=f"{label}:  {conf:.1%}")

        with st.expander(" View raw JSON"):
            st.json(result)

        with st.expander("Context days used"):
            for i, t in enumerate(recent_texts, 1):
                st.markdown(f"**Day {i}:** {t}")

with tab2:
    st.markdown("### Weather Text Generation")
    st.write(
        "The language-model head generates weather-style text "
        "continuing from your prompt (trained jointly with the forecast objective)."
    )

    prompt = st.text_input("Starting prompt", value="On July 26,")

    c1, c2 = st.columns(2)
    with c1:
        max_tokens = st.slider("Max new tokens", min_value=10, max_value=80, value=40, step=5)
    with c2:
        temperature = st.slider("Temperature (creativity)", min_value=0.5, max_value=1.5, value=1.0, step=0.1)

    if st.button("✨  Generate Text", type="primary"):
        with st.spinner("Generating..."):
            generated = generate_text(
                model, tokenizer, device, model_config,
                prompt, max_new_tokens=max_tokens, temperature=temperature,
            )
        st.markdown("#### Output")
        st.info(generated)

with tab3:
    st.markdown("### Project Overview")
    st.markdown("""
This application demonstrates a **decoder-only Transformer** built entirely from
scratch (no pretrained weights) and trained on approximately **20 years** of daily
Pakistan weather observations (2007–2026).

**Architecture**
- Token + learned positional embeddings  
- 4 causal Transformer blocks (d_model=128, 4 heads, d_ff=512)  
- Joint training: language modeling + next-day classification heads  

**Tasks**
| Head | Output |
|------|--------|
| Language Model | Next-token prediction (weather text) |
| Temperature Classifier | Cold / Moderate / Hot |
| Humidity Classifier | Dry / Moderate / Humid |

**Data safeguards**
- Chronological train / val / test split (no future leakage)
- Forecast uses only *past* days as context
- Dropout, weight decay, label smoothing, gradient clipping
    """)

    st.markdown("### How to interpret the forecast")
    st.markdown("""
The model does **not** predict exact temperature values.  
It predicts **categories** derived from fixed thresholds, using the recent
weather pattern as context — similar to how a human might say
“looking at the last few days, tomorrow should be hot.”
    """)