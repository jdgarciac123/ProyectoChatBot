import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocesamiento import preprocess_and_save
from model import build_seq2seq_model

MODEL_PATH = 'models/chatbot_model.h5'
BATCH_SIZE = 64
EPOCHS     = 50

def train_model():
    enc_in, dec_in, dec_trg = preprocess_and_save()
    model = build_seq2seq_model(
        vocab_size=None  # Toma VOCAB_SIZE interno
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

