# Librerías
import os
import pickle
import numpy as np
import unicodedata
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_DIR = 'data/cornell/'
TOKENIZER_PATH = 'models/tokenizer.pkl'
VOCAB_SIZE = 10000
MAX_SEQ_LEN = 20

def proteger_tokens(texto):
    texto = texto.replace('<start>', 'starttoken123')
    texto = texto.replace('<end>', 'endtoken123')
    return texto

def restaurar_tokens(texto):
    texto = texto.replace('starttoken123', '<start>')
    texto = texto.replace('endtoken123', '<end>')
    return texto

def limpiar_texto(texto):
    # Proteger tokens antes de limpiar
    texto = proteger_tokens(texto)
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    texto = re.sub(r'[^a-z0-9\s\.\,\?\!]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    # Restaurar tokens después de limpiar
    texto = restaurar_tokens(texto)
    return texto


def load_cornell_data(data_dir=DATA_DIR):
    id2line = {}
    with open(os.path.join(data_dir, 'movie_lines.txt'), encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                id2line[parts[0]] = parts[4].strip()
    questions, answers = [], []
    with open(os.path.join(data_dir, 'movie_conversations.txt'), encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split(' +++$+++ ')
            if len(parts) == 4:
                conv_ids = eval(parts[3])
                for i in range(len(conv_ids) - 1):
                    q = id2line.get(conv_ids[i], '')
                    a = id2line.get(conv_ids[i+1], '')
                    if q and a:
                        questions.append(q)
                        answers.append('<start> ' + a + ' <end>')
    return questions, answers

def build_tokenizer(texts, num_words=VOCAB_SIZE):
    tokenizer = Tokenizer(num_words=num_words, oov_token='<UNK>')
    tokenizer.fit_on_texts(texts)
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    return tokenizer

def load_tokenizer(path=TOKENIZER_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)

def encode_sequences(tokenizer, texts, max_len=MAX_SEQ_LEN):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding='post')

def preprocess_and_save():
    questions, answers = load_cornell_data()

    # Muestra ejemplos antes de limpiar
    print("Ejemplo pregunta antes de limpiar:", questions[0])
    print("Ejemplo respuesta antes de limpiar:", answers[0])

    questions = [limpiar_texto(q) for q in questions]
    answers = [limpiar_texto(a) for a in answers]

    # Muestra ejemplos después de limpiar
    print("Ejemplo pregunta después de limpiar:", questions[0])
    print("Ejemplo respuesta después de limpiar:", answers[0])

    tokenizer = build_tokenizer(questions + answers)

    # Validación de los tokens especiales
    print('<start>' in tokenizer.word_index)
    print('<end>' in tokenizer.word_index)
    print('Index for <start>:', tokenizer.word_index.get('<start>'))
    print('Index for <end>:', tokenizer.word_index.get('<end>'))

    encoder_input = encode_sequences(tokenizer, questions)
    decoder_input = encode_sequences(tokenizer, answers)
    decoder_target = np.zeros_like(decoder_input)
    decoder_target[:, :-1] = decoder_input[:, 1:]
    return encoder_input, decoder_input, decoder_target
