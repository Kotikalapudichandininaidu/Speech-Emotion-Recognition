# streamlit_emotion_app.py
import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import soundfile as sf
import io
import os
from pydub import AudioSegment, effects
import noisereduce as nr
import time
import sounddevice as sd

# --- Configuration from working.py (Adjust paths as needed) ---
# Assuming the model file and data file are in the same directory or accessible path
OUTPUT_DIR = '.' # Or specify the directory where files are saved
MODEL_FILE_PATH = os.path.join(OUTPUT_DIR, 'best_weights_lstm_mod.keras') # Preferred Keras format
# Alternative if using JSON/H5:
# MODEL_JSON_FILE = os.path.join(OUTPUT_DIR, 'model_lstm_mod.json')
# MODEL_H5_FILE = os.path.join(OUTPUT_DIR, 'model_lstm_mod.weights.h5')
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, 'processed_data.npz') # Needed for target_length

# Constants from working.py
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 13
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
NUM_CLASSES = len(EMOTION_LABELS)

# --- Load Target Length ---
# CRITICAL: Load the target_length used during training.
try:
    with np.load(PROCESSED_DATA_PATH, allow_pickle=True) as data:
        TARGET_LENGTH = data['target_length'].item()
        st.sidebar.success(f"Loaded target audio length: {TARGET_LENGTH} samples")
except FileNotFoundError:
    st.error(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}. Cannot determine target audio length.")
    st.stop()
except Exception as e:
    st.error(f"Error loading target length from {PROCESSED_DATA_PATH}: {e}")
    st.stop()

# --- Load Model ---
@st.cache_resource # Cache the model loading
def load_emotion_model():
    try:
        # Load Keras format model
        if os.path.exists(MODEL_FILE_PATH):
            model = keras.models.load_model(MODEL_FILE_PATH)
            st.sidebar.success("Emotion recognition model loaded successfully (Keras format).")
            return model
        # Fallback or alternative: Load from JSON and H5
        # elif os.path.exists(MODEL_JSON_FILE) and os.path.exists(MODEL_H5_FILE):
        #     with open(MODEL_JSON_FILE, 'r') as json_file:
        #         loaded_model_json = json_file.read()
        #     model = keras.models.model_from_json(loaded_model_json)
        #     model.load_weights(MODEL_H5_FILE)
        #     st.sidebar.success("Emotion recognition model loaded successfully (JSON/H5 format).")
        #     # Re-compile is often necessary when loading from JSON/H5
        #     model.compile(loss='categorical_crossentropy',
        #                   optimizer=tf.keras.optimizers.Adam(), # Use same optimizer settings if known
        #                   metrics=['categorical_accuracy'])
        #     return model
        else:
            st.error(f"Error: Model file not found at {MODEL_FILE_PATH}") # or JSON/H5 paths
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_emotion_model()

# --- Audio Preprocessing Function (Adapted from working.py) ---
def process_audio(y, sr, target_length):
    """Preprocesses raw audio data (numpy array) for the model."""
    try:
        # Normalize using pydub (requires conversion)
        y_int16 = (y * 32767).astype(np.int16)
        rawsound = AudioSegment(
            y_int16.tobytes(),
            frame_rate=sr,
            sample_width=y_int16.dtype.itemsize,
            channels=1 # Assuming mono
        )
        normalizedsound = effects.normalize(rawsound, headroom=0)
        normal_x = np.array(normalizedsound.get_array_of_samples(), dtype='float32')
        current_sr = normalizedsound.frame_rate

        # Trim silence
        xt, _ = librosa.effects.trim(normal_x, top_db=30)

        # Pad or truncate
        if len(xt) > target_length:
            xt = xt[:target_length]
        else:
            xt = np.pad(xt, (0, target_length - len(xt)), 'constant')

        # Noise reduction (optional, can be slow)
        # Consider making this optional or adjusting parameters
        try:
            final_x = nr.reduce_noise(xt, sr=current_sr, stationary=False, prop_decrease=0.8) # Adjust prop_decrease
        except Exception as nr_e:
            st.warning(f"Noise reduction failed: {nr_e}. Proceeding without it.")
            final_x = xt # Use trimmed/padded audio if NR fails

        # Extract features
        rms = librosa.feature.rms(y=final_x, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        zcr = librosa.feature.zero_crossing_rate(y=final_x, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        mfccs = librosa.feature.mfcc(y=final_x, sr=current_sr, n_mfcc=N_MFCC,
                                      n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)

        # Combine features and transpose
        features = np.vstack((zcr, rms, mfccs)).T # Shape: (timesteps, features)

        # Add batch dimension for model prediction
        features = np.expand_dims(features, axis=0) # Shape: (1, timesteps, features)

        return features

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# --- Microphone Recording Function ---
# NOTE: This uses sounddevice which might be easier to install than pyaudio sometimes.
# Install using: pip install sounddevice
# If you prefer pyaudio, replace this function accordingly.
import sounddevice as sd

RECORD_DURATION = 5 # Duration in seconds
SAMPLE_RATE = 22050 # Sample rate matching typical model training

def record_audio(duration, fs):
    """Records audio from the microphone for a specified duration."""
    st.info(f"Recording for {duration} seconds... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait() # Wait until recording is finished
    st.success("Recording finished!")
    return recording.flatten() # Return as a flat numpy array

# --- Streamlit UI ---
st.title("Real-time Emotion Recognition")
st.write("Upload an audio file or record live audio to predict the emotion.")

if model is None:
    st.error("Model could not be loaded. Please check the model file path and integrity.")
    st.stop()

input_method = st.radio("Choose input method:", ("Upload Audio File", "Live Microphone"))

predicted_emotion = None
confidence_scores = None
audio_data = None
sr = None

if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])
    if uploaded_file is not None:
        try:
            # Read audio data using soundfile
            # Use BytesIO to handle the uploaded file in memory
            file_bytes = io.BytesIO(uploaded_file.read())
            if uploaded_file.name.lower().endswith(".mp3"):
                # Need pydub to convert mp3 to wav first for librosa/soundfile
                st.info("Converting MP3 to WAV...")
                audio_segment = AudioSegment.from_mp3(file_bytes)
                # Export to a temporary WAV bytes buffer
                wav_bytes = io.BytesIO()
                audio_segment.export(wav_bytes, format="wav")
                wav_bytes.seek(0)
                audio_data, sr = sf.read(wav_bytes)
            elif uploaded_file.name.lower().endswith(".wav"):
                 audio_data, sr = sf.read(file_bytes)
            else:
                 st.error("Unsupported file format. Please upload WAV or MP3.")
                 st.stop()


            # Convert to mono if necessary
            if audio_data.ndim > 1:
                audio_data = librosa.to_mono(audio_data.T) # Transpose if needed by to_mono

            st.audio(uploaded_file, format='audio/wav') # Display the audio player

        except Exception as e:
            st.error(f"Error reading or processing uploaded file: {e}")
            audio_data = None # Ensure audio_data is None if error


elif input_method == "Live Microphone":
    if st.button("Start Recording"):
        # Check if sounddevice is available
        try:
            import sounddevice # noqa
        except ImportError:
            st.error("Microphone input requires the 'sounddevice' library. Please install it (pip install sounddevice).")
            st.stop()
        except Exception as e:
             st.error(f"Could not initialize audio device. Is a microphone connected? Error: {e}")
             st.stop()

        with st.spinner('Recording...'):
            audio_data = record_audio(RECORD_DURATION, SAMPLE_RATE)
            sr = SAMPLE_RATE # Set sample rate used for recording
            # Display recorded audio
            st.audio(audio_data, format='audio/wav', sample_rate=sr)


# --- Process and Predict ---
if audio_data is not None and sr is not None:
    st.write("Preprocessing audio and predicting emotion...")
    start_time = time.time()
    with st.spinner('Processing...'):
        features = process_audio(audio_data, sr, TARGET_LENGTH)

    if features is not None:
        try:
            prediction_start_time = time.time()
            predictions = model.predict(features)
            prediction_end_time = time.time()

            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_emotion = EMOTION_LABELS[predicted_index]
            confidence_scores = predictions[0]

            end_time = time.time()
            st.write(f"Preprocessing time: {prediction_start_time - start_time:.2f} seconds")
            st.write(f"Prediction time: {prediction_end_time - prediction_start_time:.2f} seconds")


        except Exception as e:
            st.error(f"Error during prediction: {e}")


# --- Display Results ---
if predicted_emotion is not None and confidence_scores is not None:
    st.subheader("Prediction Result:")
    st.success(f"Predicted Emotion: {predicted_emotion.capitalize()}")

    st.subheader("Confidence Scores:")
    # Create a dictionary for the bar chart
    confidence_dict = {label: score for label, score in zip(EMOTION_LABELS, confidence_scores)}
    st.bar_chart(confidence_dict)

    # Optional: Display raw scores
    # st.write("Raw Scores:")
    # for label, score in confidence_dict.items():
    #     st.write(f"- {label}: {score:.4f}")
elif audio_data is not None and sr is not None and predicted_emotion is None:
     st.warning("Could not predict emotion. Check processing errors above.")


st.sidebar.markdown("---")
st.sidebar.header("Model & Data Info")
st.sidebar.write(f"Model: {os.path.basename(MODEL_FILE_PATH)}") # Or JSON/H5 names
st.sidebar.write(f"Required Audio Length: {TARGET_LENGTH} samples")
st.sidebar.write(f"Emotion Classes: {', '.join(EMOTION_LABELS)}")