from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import datasets
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input, concatenate
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Reshape
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import StandardScaler
import random


def train_model():
    # Load the dataset
    dataset = list(datasets.load_dataset('artem9k/ai-text-detection-pile', split='train'))

    # Shuffle the dataset randomly
    random.shuffle(dataset)

    # Select a random subset of elements for training and testing
    train_size = 25000
    test_size = 5000
    train_subset = random.sample(dataset, train_size)
    test_subset = random.sample(dataset, test_size)

    # Initialize empty lists for training and testing
    train_text = []
    train_labels = []
    test_text = []
    test_labels = []
    train_sentence_lengths = []
    test_sentence_lengths = []

    # Iterate over the train subset
    for data in train_subset:
        train_text.append(data['text'])
        train_labels.append(data['source'])
        train_sentence_lengths.append(len(data['text'].split()))

    # Iterate over the test subset
    for data in test_subset:
        test_text.append(data['text'])
        test_labels.append(data['source'])
        test_sentence_lengths.append(len(data['text'].split()))

    # Convert lists to numpy arrays
    train_text = np.array(train_text)
    train_labels = np.array(train_labels)
    test_text = np.array(test_text)
    test_labels = np.array(test_labels)
    train_sentence_lengths = np.array(train_sentence_lengths)
    test_sentence_lengths = np.array(test_sentence_lengths)

    # Tokenize and convert text to sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_text)
    train_sequences = tokenizer.texts_to_sequences(train_text)
    test_sequences = tokenizer.texts_to_sequences(test_text)

    # Pad sequences to a fixed length
    max_sequence_length = max([len(text.split()) for text in np.concatenate([train_text, test_text])])
    train_data = pad_sequences(train_sequences, maxlen=max_sequence_length)
    test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)

    # Convert labels to numerical representations
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)

    # Convert labels to float type
    train_labels = train_labels.astype(float)
    test_labels = test_labels.astype(float)

    # Normalize sentence lengths
    scaler = StandardScaler()
    train_sentence_lengths = scaler.fit_transform(train_sentence_lengths.reshape(-1, 1))
    test_sentence_lengths = scaler.transform(test_sentence_lengths.reshape(-1, 1))

    embedding_dim = 50  # Dimensionality of the GloVe word embeddings
    glove_path = 'glove.6B.50d.txt'  # Path to the downloaded GloVe file

    # Load pretrained word embeddings
    word_vectors = {}
    with open(glove_path, 'r', encoding='utf8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_vectors[word] = coefs

    # Convert pretrained embeddings to a matrix
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = word_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Define the model
    text_input = Input(shape=(max_sequence_length,))
    length_input = Input(shape=(1,))
    # embedding = Embedding(len(tokenizer.word_index) + 1, 128, input_length=max_sequence_length)(text_input)
    embedding = Embedding(len(tokenizer.word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False)(text_input)

    conv = Conv1D(64, kernel_size=3, activation='relu')(embedding)
    pooling = GlobalMaxPooling1D()(conv)
    reshaped_pooling = Reshape((1, 64))(pooling)  # Reshape to add a third dimension
    lstm = Bidirectional(LSTM(64))(reshaped_pooling)
    concat = concatenate([lstm, length_input])

    dropout_rate = 0.2
    dense = Dense(128, activation='relu')(concat)
    # dropout1 = Dropout(dropout_rate)(dense)
    dense1 = Dense(64, activation='relu')(dense)
    # dropout2 = Dropout(dropout_rate)(dense1)
    dense2 = Dense(32, activation='relu')(dense1)
    dropout3 = Dropout(dropout_rate)(dense2)
    dense3 = Dense(16, activation='relu')(dropout3)
    # dropout4 = Dropout(dropout_rate)(dense3)
    dense4 = Dense(8, activation='relu')(dense3)
    # dropout5 = Dropout(dropout_rate)(dense4)
    dense5 = Dense(4, activation='relu')(dense4)
    # dropout6 = Dropout(dropout_rate)(dense5)
    dense6 = Dense(3, activation='relu')(dense5)
    dropout = Dropout(dropout_rate)(dense6)
    output_layer = Dense(1, activation='sigmoid')(dropout)
    model = Model(inputs=[text_input, length_input], outputs=output_layer)

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * 0.1 if epoch >= 5 else lr)

    # Train the model
    model.fit([train_data, train_sentence_lengths], train_labels, epochs=20, batch_size=32, validation_data=([test_data, test_sentence_lengths], test_labels), callbacks=[early_stopping, lr_scheduler])

    # Evaluate the model
    loss, accuracy = model.evaluate([test_data, test_sentence_lengths], test_labels)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')