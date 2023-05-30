from keras.utils import pad_sequences
import numpy as np
import datasets
import random
import csv

from model import train_model


def test_model():
    # Load the dataset
    dataset = list(datasets.load_dataset('artem9k/ai-text-detection-pile', split='train'))

    # Shuffle the dataset randomly
    random.shuffle(dataset)

    # Select a subset of elements for testing
    tokenizer, max_sequence_length, scaler, model, label_encoder = train_model()
    for i in range(0,1000):
        test_size = 1000
        test_subset = random.sample(dataset, test_size)

        # Initialize empty lists for testing
        test_text = []
        test_labels = []
        test_sentence_lengths = []

        # Iterate over the test subset
        for data in test_subset:
            test_text.append(data['text'])
            test_labels.append(data['source'])
            test_sentence_lengths.append(len(data['text'].split()))

        # Convert lists to numpy arrays
        test_text = np.array(test_text)
        test_sentence_lengths = np.array(test_sentence_lengths)


        # Tokenize and convert text to sequences
        # tokenizer.fit_on_texts(test_text)
        test_sequences = tokenizer.texts_to_sequences(test_text)

        # Pad sequences to a fixed length
        test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)

        # Normalize sentence lengths
        # scaler.fit(test_sentence_lengths.reshape(-1, 1))  # Fit on training data
        test_sentence_lengths = scaler.transform(test_sentence_lengths.reshape(-1, 1))

        # Save the results to a CSV file
        header = ['source', 'id', 'text', 'predicted_label']
        with open('data.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(header)
            for i in range(len(test_subset)):
                try:
                    source = test_subset[i]['source']
                    id = test_subset[i]['id']
                    text = test_subset[i]['text']
                    if len(text) < 1000:
                        continue
                    prediction = model.predict([test_data[i:i + 1], test_sentence_lengths[i:i + 1]])[0][0]
                    predicted_label = label_encoder.inverse_transform(prediction.flatten().round().astype(int))[0]
                    row = [source, id, text, predicted_label]
                    writer.writerow(row)
                except:
                    continue

    print("Testing complete. Results saved to data.csv")


if __name__ == '__main__':
    test_model()
