import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocesamiento import preprocess_and_save, TOKENIZER_PATH
from model import build_seq2seq_model
import pickle
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))


MODEL_PATH = 'models/chatbot_model.keras'
BATCH_SIZE = 64
EPOCHS     = 8

def train_model():
    enc_in, dec_in, dec_trg = preprocess_and_save()

    # Carga el tokenizer para obtener vocab_size real
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    
    vocab_size = int(len(tokenizer.word_index) + 1)
    print(f"VOCAB SIZE QUE LLEGA AL MODELO: {vocab_size}")

    model = build_seq2seq_model(
        vocab_size=vocab_size 
    )
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy'
    )
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', verbose=1)
    ]
    model.fit(
        [enc_in, dec_in],
        dec_trg[..., np.newaxis],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=callbacks
    )
    print(f"Modelo entrenado y guardado en {MODEL_PATH}")


if __name__ == '__main__':
    train_model()

