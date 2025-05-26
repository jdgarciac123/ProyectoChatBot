import numpy as np
from tensorflow.keras.models import load_model
from preprocesamiento import load_tokenizer, MAX_SEQ_LEN, TOKENIZER_PATH
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

LATENT_DIM = 256
MODEL_PATH = 'models/chatbot_model.h5'


def load_models():
    tokenizer = load_tokenizer()
    full_model = load_model(MODEL_PATH)
    # Encoder reconstruction
    enc_inputs = full_model.get_layer('encoder_inputs').input
    enc_embed = full_model.get_layer('encoder_embedding')(enc_inputs)
    _, st_h, st_c = full_model.get_layer('encoder_lstm')(enc_embed)
    encoder_model = Model(enc_inputs, [st_h, st_c])

    # Decoder reconstruction
    dec_state_h = Input(shape=(LATENT_DIM,), name='dec_state_h')
    dec_state_c = Input(shape=(LATENT_DIM,), name='dec_state_c')
    dec_inputs = full_model.get_layer('decoder_inputs').input
    dec_embed = full_model.get_layer('decoder_embedding')(dec_inputs)
    dec_lstm = full_model.get_layer('decoder_lstm')
    dec_outs, out_h, out_c = dec_lstm(dec_embed, initial_state=[dec_state_h, dec_state_c])
    dec_dense = full_model.get_layer('decoder_dense')
    dec_outs = dec_dense(dec_outs)
    decoder_model = Model(
        [dec_inputs, dec_state_h, dec_state_c],
        [dec_outs, out_h, out_c]
    )
    return tokenizer, encoder_model, decoder_model


def generate_response(
    text, history, tokenizer, encoder_model, decoder_model, max_seq_len=MAX_SEQ_LEN
):
    context = text
    if len(history) >= 1:
        last_u, last_b = history[-1]
        context = f"{last_u} __sep__ {last_b} __sep__ {text}"
    seq = tokenizer.texts_to_sequences([context])
    seq = pad_sequences(seq, maxlen=max_seq_len, padding='post')
    states = encoder_model.predict(seq)
    target_seq = np.array([[tokenizer.word_index['<start>']]])
    decoded = []
    stop_cond = False
    while not stop_cond:
        out_tokens, h, c = decoder_model.predict([target_seq] + states)
        idx = np.argmax(out_tokens[0, -1, :])
        word = tokenizer.index_word.get(idx, '<UNK>')
        if word == '<end>' or len(decoded) >= max_seq_len:
            stop_cond = True
        else:
            decoded.append(word)
            target_seq = np.array([[idx]])
            states = [h, c]
    return ' '.join(decoded)
