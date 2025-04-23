# Speech-Emotion-Recognition
# Emotion Recognition from Speech using LSTM

## Description

This project implements a deep learning model to recognize emotions from audio speech signals. It utilizes the TESS (Toronto Emotional Speech Set) dataset. The core of the model is a Bidirectional Long Short-Term Memory (LSTM) network built with TensorFlow/Keras. The process involves:

1.  **Preprocessing:** Loading audio files, normalizing volume, trimming silence, padding/truncating to a fixed length, applying noise reduction, and extracting features (MFCC, RMS, ZCR).
2.  **Model Training:** Defining and training a Bidirectional LSTM model on the extracted features to classify emotions.
3.  **Evaluation:** Assessing the model's performance using validation and test sets, including accuracy metrics and confusion matrices.
4.  **Prediction:** Using the trained model to predict the emotion in a new audio file.

## Dependencies

The project requires the following Python libraries. You can install them using pip:

```bash
pip install numpy json_tricks pydub librosa noisereduce tensorflow scikit-learn matplotlib seaborn pandas
numpy: For numerical operations.
pydub: For audio manipulation like normalization.
librosa: For audio analysis, feature extraction (MFCC, RMS, ZCR), loading, and effects like trimming.
noisereduce: For reducing background noise in audio signals.
tensorflow: For building and training the deep learning model (LSTM).
scikit-learn: For data splitting and evaluation metrics (confusion matrix).
matplotlib & seaborn: For plotting training history and confusion matrices.
pandas: Used for handling the confusion matrix data structure.
json_tricks: Potentially used for saving/loading data structures (though numpy.savez is used for primary data persistence).
Dataset
Source: The TESS (Toronto Emotional Speech Set) dataset is used.
Path: The code expects the dataset to be located at the path specified by the DATASET_PATH variable (e.g., /content/drive/MyDrive/special project/TESS). You MUST adjust this path to point to your dataset location.
Emotions: The model recognizes the following emotions, mapped internally to numerical labels:
neutral: 0
happy: 1
sad: 2
angry: 3
fear: 4 ('fearful' in labels)
disgust: 5
ps: 6 ('surprised' in labels)
Configuration
Key parameters and paths are defined at the beginning of the script (Working.ipynb) and may need adjustment:

DATASET_PATH: Path to the raw TESS dataset.
OUTPUT_DIR: Directory to save processed data, model files, and test data (e.g., /content/drive/My Drive/Colab Notebooks/).
PROCESSED_DATA_FILE: Path to save/load the preprocessed features and labels (.npz format).
MODEL_WEIGHTS_FILE: Path to save the best trained model weights (.keras format).
MODEL_JSON_FILE, MODEL_H5_FILE: Paths for saving model architecture (JSON) and weights (H5) separately (primarily for compatibility).
X_TEST_FILE, Y_TEST_FILE: Paths to save the test dataset features and labels.
Audio Features: FRAME_LENGTH, HOP_LENGTH, N_MFCC.
Training Hyperparameters: BATCH_SIZE, EPOCHS, LEARNING_RATE, LSTM_UNITS, DROPOUT_RATE.
Usage
The Jupyter Notebook Working.ipynb contains the full workflow:

Setup: Installs dependencies and imports necessary libraries.
Configuration: Sets paths and hyperparameters (adjust these as needed).
Data Preprocessing:
The preprocess_data function iterates through the DATASET_PATH.
For each audio file, process_audio_file performs loading, normalization, trimming, padding/truncation, noise reduction, and feature extraction (RMS, ZCR, MFCCs).
It determines a target_length based on the longest audio file after trimming (unless pre-calculated).
The extracted features (X) and corresponding emotion labels (Y) are saved to PROCESSED_DATA_FILE to avoid reprocessing on subsequent runs. If this file exists, it's loaded directly.
Data Splitting: The processed data is split into training (70%), validation (15%), and test (15%) sets. The test set is saved separately. Labels are one-hot encoded for the model.
Model Building: A Sequential Keras model is defined:
Input Layer (shape based on features)
Bidirectional LSTM layer (returns sequences)
Dropout
Bidirectional LSTM layer (returns last output)
Dropout
Dense layer (ReLU activation)
Dropout
Output Dense layer (Softmax activation for classification)
Compiled with Adam optimizer and categorical cross-entropy loss.
Training:
The model is trained using model.fit on the training data, validating against the validation set.
Callbacks are used:
ModelCheckpoint: Saves the best model based on validation accuracy to MODEL_WEIGHTS_FILE.
ReduceLROnPlateau: Reduces learning rate if validation accuracy stagnates.
EarlyStopping: Stops training if validation loss doesn't improve, restoring the best weights.
Evaluation:
Training history (loss and accuracy) is plotted using plot_history.
The evaluate_model function calculates loss and accuracy on the validation and test sets.
It also generates and plots confusion matrices for both sets and prints per-class accuracy.
Saving: The final trained model architecture is saved as JSON (MODEL_JSON_FILE) and weights as H5 (MODEL_H5_FILE). The best model is already saved in Keras format (MODEL_WEIGHTS_FILE).
Prediction:
The predict_emotion_from_file function takes an audio file path, the trained model, and the target_length used during preprocessing.
It processes the single audio file in the same way as the training data, predicts the emotion using model.predict, and prints the predicted emotion label along with confidence scores for each class.
An example prediction is run using EXAMPLE_AUDIO_FILE (adjust this path).
Running the Code
Ensure all dependencies are installed.
Modify the paths in the Configuration section of Working.ipynb to match your environment (dataset location, desired output directory).
Run the cells in the Jupyter Notebook sequentially.
The first run will perform data preprocessing, which might take time depending on the dataset size. Subsequent runs will load the saved processed_data.npz file.
Training will commence, saving the best model checkpoint.
Evaluation results and plots will be displayed.
A prediction on the example audio file will be performed.
