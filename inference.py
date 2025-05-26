import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from preprocesamiento import load_tokenizer, MAX_SEQ_LEN, TOKENIZER_PATH
import pickle

LATENT_DIM = 256
EMBED_DIM = 100  # Debe coincidir con el usado en entrenamiento
MODEL_PATH = 'models/chatbot_model.keras'

def build_encoder(vocab_size, embed_dim, latent_dim, max_seq_len):
    encoder_inputs = Input(shape=(max_seq_len,), name='encoder_inputs')
    enc_embed = Embedding(vocab_size, embed_dim, mask_zero=True, name='encoder_embedding')(encoder_inputs)
    encoder_lstm = LSTM(
        latent_dim,
        return_state=True,
        name='encoder_lstm'
    )
    _, state_h, state_c = encoder_lstm(enc_embed)
    encoder_model = Model(encoder_inputs, [state_h, state_c])
    return encoder_model

def build_decoder(vocab_size, embed_dim, latent_dim, max_seq_len):
    decoder_inputs = Input(shape=(1,), name='decoder_inputs_infer')  # paso a paso
    dec_state_h = Input(shape=(latent_dim,), name='dec_state_h')
    dec_state_c = Input(shape=(latent_dim,), name='dec_state_c')
    dec_embed = Embedding(vocab_size, embed_dim, mask_zero=True, name='decoder_embedding')(decoder_inputs)
    decoder_lstm = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        name='decoder_lstm'
    )
    dec_outputs, out_h, out_c = decoder_lstm(dec_embed, initial_state=[dec_state_h, dec_state_c])
    decoder_dense = Dense(vocab_size, activation='softmax', name='decoder_dense')
    dec_outs = decoder_dense(dec_outputs)
    decoder_model = Model(
        [decoder_inputs, dec_state_h, dec_state_c],
        [dec_outs, out_h, out_c]
    )
    return decoder_model

def load_models():
    # Cargar tokenizer para vocab_size real
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer.word_index) + 1

    # Cargar el modelo completo solo para obtener los pesos
    full_model = load_model(MODEL_PATH)

    # Reconstruir encoder y decoder para inferencia
    encoder_model = build_encoder(vocab_size, EMBED_DIM, LATENT_DIM, MAX_SEQ_LEN)
    decoder_model = build_decoder(vocab_size, EMBED_DIM, LATENT_DIM, MAX_SEQ_LEN)

    # Cargar pesos de las capas (nombres deben coincidir con entrenamiento)
    encoder_model.get_layer('encoder_embedding').set_weights(full_model.get_layer('encoder_embedding').get_weights())
    encoder_model.get_layer('encoder_lstm').set_weights(full_model.get_layer('encoder_lstm').get_weights())

    decoder_model.get_layer('decoder_embedding').set_weights(full_model.get_layer('decoder_embedding').get_weights())
    decoder_model.get_layer('decoder_lstm').set_weights(full_model.get_layer('decoder_lstm').get_weights())
    decoder_model.get_layer('decoder_dense').set_weights(full_model.get_layer('decoder_dense').get_weights())

    return tokenizer, encoder_model, decoder_model

def generate_response(
    text, history, tokenizer, encoder_model, decoder_model, max_seq_len=MAX_SEQ_LEN
):
    context = text
    if len(history) >= 1:
        last_u, last_b = history[-1]
        context = f"{last_u} __sep__ {last_b} __sep__ {text}"
    seq = tokenizer.texts_to_sequences([context])
    seq = np.array(seq)
    seq = np.pad(seq, ((0,0), (0, max_seq_len - seq.shape[1])), 'constant') if seq.shape[1] < max_seq_len else seq[:, :max_seq_len]
    states = encoder_model.predict(seq)
    target_seq = np.array([[tokenizer.word_index['<start>']]])
    decoded = []
    stop_cond = False
    while not stop_cond:
        out_tokens, h, c = decoder_model.predict([target_seq, states[0], states[1]])
        idx = np.argmax(out_tokens[0, -1, :])
        word = tokenizer.index_word.get(idx, '<UNK>')
        if word == '<end>' or len(decoded) >= max_seq_len:
            stop_cond = True
        else:
            decoded.append(word)
            target_seq = np.array([[idx]])
            states = [h, c]
    return ' '.join(decoded)

