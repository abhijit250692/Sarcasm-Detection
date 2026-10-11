import streamlit as st
from transformers import pipeline
from pathlib import Path
from zipfile import ZipFile, BadZipFile

st.set_page_config(page_title="Sarcasm Detection App", page_icon="😏", layout="centered")

st.title("😏 Sarcasm Detection Tool")

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "distilbert_sarcasm_model"
MODEL_ARCHIVE = PROJECT_DIR / "distilbert_sarcasm_model.zip"


def model_files_exist():
    has_config = (MODEL_DIR / "config.json").is_file()
    has_weights = any(
        (MODEL_DIR / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )
    return has_config and has_weights


def extract_model_archive():
    if not MODEL_ARCHIVE.is_file() or model_files_exist():
        return

    try:
        with ZipFile(MODEL_ARCHIVE) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            if not entries:
                raise ValueError(f"Model archive is empty: {MODEL_ARCHIVE}")

            archive_has_model_folder = all(
                Path(entry.filename).parts[0] == MODEL_DIR.name
                for entry in entries
            )
            destination = PROJECT_DIR if archive_has_model_folder else MODEL_DIR
            destination.mkdir(parents=True, exist_ok=True)
            destination_root = destination.resolve()

            for entry in entries:
                extracted_path = (destination / entry.filename).resolve()
                if (
                    extracted_path != destination_root
                    and destination_root not in extracted_path.parents
                ):
                    raise ValueError(
                        f"Unsafe path in model archive: {entry.filename}"
                    )

            archive.extractall(destination)
    except BadZipFile as error:
        raise ValueError(f"Invalid model archive: {MODEL_ARCHIVE}") from error

@st.cache_resource
def load_model():
    extract_model_archive()
    if model_files_exist():
        return pipeline("text-classification", model=str(MODEL_DIR))
    else:
        # Fallback to the original hub repository if local files are missing
        return pipeline("text-classification", model="kokolamba/distilbert-base-uncased-sarcasm")

try:
    with st.spinner("Loading DistilBERT model... Please wait."):
        classifier = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# User Input
headline_input = st.text_input("Enter a headline:", placeholder="e.g., Local man wins lottery and loses ticket immediately")

if st.button("Analyze"):
    if headline_input.strip() == "":
        st.warning("Please enter a valid headline.")
    else:
        with st.spinner("Analyzing..."):
            prediction = classifier(headline_input)[0]
            label = prediction["label"]
            score = prediction["score"]
            
            # Map predicted label to user-friendly output
            is_sarcastic = (label == "LABEL_1" or "sarcastic" in label.lower())
            
        st.write("--- ")
        if is_sarcastic:
            st.subheader(f"Prediction: **Sarcastic** 😏")
            st.metric(label="Confidence", value=f"{score:.2%}")
        else:
            st.subheader(f"Prediction: **Not Sarcastic / Serious** 📰")
            st.metric(label="Confidence", value=f"{score:.2%}")