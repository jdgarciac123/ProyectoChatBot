from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense

VOCAB_SIZE = 10000
EMBED_DIM   = 100
LATENT_DIM  = 256
MAX_SEQ_LEN = 20

def build_seq2seq_model(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    latent_dim=LATENT_DIM,
    max_seq_len=MAX_SEQ_LEN
):
    # Encoder
    encoder_inputs = Input(shape=(max_seq_len,), name='encoder_inputs')
    enc_embed = Embedding(vocab_size, embed_dim, mask_zero=True, name='encoder_embedding')(encoder_inputs)
    encoder_lstm = LSTM(
        latent_dim,
        return_state=True,
        dropout=0.3,
        recurrent_dropout=0.3,
        name='encoder_lstm'
    )
    _, state_h, state_c = encoder_lstm(enc_embed)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(max_seq_len,), name='decoder_inputs')
    dec_embed = Embedding(vocab_size, embed_dim, mask_zero=True, name='decoder_embedding')(decoder_inputs)
    decoder_lstm = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        dropout=0.3,
        recurrent_dropout=0.3,
        name='decoder_lstm'
    )
    dec_outputs, _, _ = decoder_lstm(dec_embed, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(dec_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
