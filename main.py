import tensorflow as tf
import numpy as np
import os
import re

# Load the Cornell Movie Dialogs Corpus dataset
path_to_dataset = "./cornell movie-dialogs corpus"
lines = open(os.path.join(path_to_dataset, "movie_lines.txt"),
             encoding="utf-8", errors="ignore").read().split("\n")
conversations = open(os.path.join(path_to_dataset, "movie_conversations.txt"),
                     encoding="utf-8", errors="ignore").read().split("\n")

# Create a dictionary to map each line's ID with its text
id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Create a list of all the conversation's lines
conversations_ids = []
for conversation in conversations:
    _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(","))

# Create the questions and answers datasets
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i + 1]])

# Preprocessing the data
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text

questions = [clean_text(question) for question in questions]
answers = [clean_text(answer) for answer in answers]

# Encoding the sentences into integers
# Encoding the sentences into integers
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000, filters='', oov_token="<OOV>")
tokenizer.fit_on_texts(questions + answers)

# Get the word index for each word in the corpus
word_index = tokenizer.word_index

# Convert each sentence into sequences of integers
questions_sequences = tokenizer.texts_to_sequences(questions)
answers_sequences = tokenizer.texts_to_sequences(answers)

# Padding the sequences to have the same length
max_length = max(len(max(questions_sequences, key=len)), len(max(answers_sequences, key=len)))
questions_sequences = tf.keras.preprocessing.sequence.pad_sequences(questions_sequences, maxlen=max_length, padding="post")
answers_sequences = tf.keras.preprocessing.sequence.pad_sequences(answers_sequences, maxlen=max_length, padding="post")

# Splitting the data into training and testing sets
training_size = int(len(questions_sequences) * 0.8)
questions_train = questions_sequences[:training_size]
answers_train = answers_sequences[:training_size]
questions_test = questions_sequences[training_size:]
answers_test = answers_sequences[training_size:]

# Building the encoder
encoder_inputs = tf.keras.layers.Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(input_dim=len(word_index) + 1, output_dim=256)(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(units=512, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Building the decoder
decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(input_dim=len(word_index) + 1, output_dim=256)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(units=512, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(units=len(word_index) + 1, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Defining the model
model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compiling the model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# Training the model
model.fit([questions_train, answers_train[:, :-1]], answers_train.reshape(answers_train.shape[0], answers_train.shape[1], 1)[:, 1:], epochs=100, batch_size=64, validation_data=([questions_test, answers_test[:, :-1]], answers_test.reshape(answers_test.shape[0], answers_test.shape[1], 1)[:, 1:]))

# Saving the model
model.save("model.h5")

# Building the encoder model
encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

# Building the decoder model
decoder_state_input_h = tf.keras.layers.Input(shape=(512,))
decoder_state_input_c = tf.keras.layers.Input(shape=(512,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Defining the function to generate answers
def generate_answer(input_sentence, encoder_model, decoder_model, word_index, max_length):
    states_value = encoder_model.predict(input_sentence)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word_index["<START>"]
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None
        for word, index in word_index.items():
            if sampled_token_index == index:
                decoded_sentence += " " + word
                sampled_word = word
        if sampled_word == "<END>" or len(decoded_sentence) > max_length:
            stop_condition = True
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return decoded_sentence.strip()

# Testing the model

