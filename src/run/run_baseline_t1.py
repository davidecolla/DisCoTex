from statistics import mean, median
import argparse
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import jaccard_score
from numpy import array, argmax
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy.spatial.distance import hamming
from tqdm import tqdm
from prettytable import PrettyTable
from Levenshtein import distance
from os.path import exists
import pandas as pd

def main(train_file, test_file):

    train_data_dict, train_all_sentences = load_dataset_t1(train_file)
    test_data_dict, test_all_sentences = load_dataset_t1(test_file)

    # Build the One-Hot Encoder on the training set vocabulary
    encoder = build_one_hot_encoder(train_all_sentences)

    # -------------------------------------------------------------------------- #
    #                              Compute Threshold                             #
    # -------------------------------------------------------------------------- #
    threshold_list_hamming = []
    threshold_list_jaccard = []
    threshold_list_edit = []
    for entry in tqdm(train_data_dict, desc='Estimating Thresholds'):

        # Estimate the threshold on entries with label == 1 only
        if entry['label'] == 1:
            # Retrieve instance prompt
            prompt_sentences = entry['prompt']
            distances_hamming = []
            distances_jaccard = []
            distances_edit = []

            # Encode the target sentence
            target_sentence_encoding = encode_sentence(encoder, entry['target'])

            for prompt_sentence in prompt_sentences:
                # Build one-hot encoding for each sentence within the prompt and compute prompt sentence-to-target distance
                prompt_sentence_encoding = encode_sentence(encoder, prompt_sentence)

                distance_prompt_sentence_target_sentence_hamming = hamming(prompt_sentence_encoding, target_sentence_encoding)
                distance_prompt_sentence_target_sentence_jaccard = jaccard_score(prompt_sentence_encoding, target_sentence_encoding)
                distance_prompt_sentence_target_sentence_edit = distance(prompt_sentence_encoding.tolist(), target_sentence_encoding.tolist())

                distances_hamming.append(distance_prompt_sentence_target_sentence_hamming)
                distances_jaccard.append(distance_prompt_sentence_target_sentence_jaccard)
                distances_edit.append(distance_prompt_sentence_target_sentence_edit)

            # Compute average distance between prompt sentences and target sentence
            average_distance_hamming = mean(distances_hamming)
            average_distance_jaccard = mean(distances_jaccard)
            average_distance_edit = mean(distances_edit)

            # Store each threshold
            threshold_list_hamming.append(average_distance_hamming)
            threshold_list_jaccard.append(average_distance_jaccard)
            threshold_list_edit.append(average_distance_edit)

    # Define the threshold as the median distance from training set
    threshold_hamming = median(threshold_list_hamming)
    threshold_jaccard = median(threshold_list_jaccard)
    threshold_edit = median(threshold_list_edit)

    # -------------------------------------------------------------------------- #
    #                                 Run Test                                   #
    # -------------------------------------------------------------------------- #
    hits_hamming = 0.0
    hits_jaccard = 0.0
    hits_edit = 0.0
    counts = 0.0

    for entry in tqdm(test_data_dict, 'Testing'):

        # Retrieve instance prompt
        prompt_sentences = entry['prompt']

        distances_hamming = []
        distances_jaccard = []
        distances_edit = []

        # Encode the target sentence
        target_sentence_encoding = encode_sentence(encoder, entry['target'])

        # Build one-hot encoding for each sentence within the prompt and compute prompt sentence-to-target distance
        for prompt_sentence in prompt_sentences:
            prompt_sentence_encoding = encode_sentence(encoder, prompt_sentence)

            distance_prompt_sentence_target_sentence_hamming = hamming(prompt_sentence_encoding, target_sentence_encoding)
            distance_prompt_sentence_target_sentence_jaccard = jaccard_score(prompt_sentence_encoding,target_sentence_encoding)
            distance_prompt_sentence_target_sentence_edit = distance(prompt_sentence_encoding.tolist(), target_sentence_encoding.tolist())
            # Add distance scores to the corresponding list
            distances_hamming.append(distance_prompt_sentence_target_sentence_hamming)
            distances_jaccard.append(distance_prompt_sentence_target_sentence_jaccard)
            distances_edit.append(distance_prompt_sentence_target_sentence_edit)

        # Compute average distance
        average_distance_hamming = mean(distances_hamming)
        average_distance_jaccard = mean(distances_jaccard)
        average_distance_edit = mean(distances_edit)

        # Set entry prediction
        # Hamming (distance)
        if average_distance_hamming <= threshold_hamming:
            entry['prediction_hamming'] = 1
        else:
            entry['prediction_hamming'] = 0
        # Jaccard (similarity)
        if average_distance_jaccard >= threshold_jaccard:
            entry['prediction_jaccard'] = 1
        else:
            entry['prediction_jaccard'] = 0
        # Edit (distance)
        if average_distance_edit <= threshold_edit:
            entry['prediction_edit'] = 1
        else:
            entry['prediction_edit'] = 0


        if entry['prediction_hamming'] == entry['label']:
            hits_hamming += 1.0
        if entry['prediction_jaccard'] == entry['label']:
            hits_jaccard += 1.0
        if entry['prediction_edit'] == entry['label']:
            hits_edit += 1.0

        counts += 1.0

    # Compute accuracy scores
    accuracy_hamming = hits_hamming/counts
    accuracy_jaccard = hits_jaccard/counts
    accuracy_edit = hits_edit/counts

    # Print results
    table = PrettyTable()
    table.field_names = ['|Dataset|', 'Hamming', 'Jaccard', 'Edit']
    table.title = 'T1 Accuracy One-Hot Encodings'
    table.add_row([len(test_data_dict), "{:.2f}".format(accuracy_hamming), "{:.2f}".format(accuracy_jaccard), "{:.2f}".format(accuracy_edit)])

    print(table.get_string())

def encode_sentence(encoder, sentence):
    words = word_tokenize(sentence.lower())
    encoding = encoder.transform(array(words).reshape(len(words),1))
    encoding = sum(encoding)
    encoding[encoding > 1] = 1.0

    return encoding

def build_one_hot_encoder(sentences):
    # creating instance of one-hot-encoder
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    tokenized_sentences = []
    unique_words = set()
    for sentence in sentences:
        sentence = sentence.lower()
        tokenized_sentences.append(word_tokenize(sentence))
        for word in word_tokenize(sentence):
            unique_words.add(word)

    unique_words_list = list(unique_words)
    values = array(unique_words_list)
    encoder.fit(values.reshape(len(values), 1))

    return encoder

def load_dataset_t1(file_path):

    data = pd.read_csv(file_path, sep='\t', header=0)
    sentences = []
    data_dict = []
    for index, row in data.iterrows():
        instance_id = row[0]
        prompt = row[1]
        target = row[2]
        label = row[3]

        entry_sentences = sent_tokenize(prompt)

        data_dict.append({
            'instance_prompt': instance_id,
            'prompt': entry_sentences,
            'target': target,
            'label': int(label)
        })

        sentences.extend(entry_sentences)
        sentences.append(target)
    print(len(data_dict))


    return data_dict, sentences

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=Path)
    parser.add_argument("--test_file", type=Path)
    p = parser.parse_args()

    main(p.train_file, p.test_file)