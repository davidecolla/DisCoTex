from statistics import mean
import argparse
from pathlib import Path
from nltk import word_tokenize
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from sklearn.metrics import jaccard_score
from tqdm import tqdm
from prettytable import PrettyTable
import pandas as pd

def main(train_data_file_path, test_data_file_path):
    _ , train_sentences = load_dataset_t2(train_data_file_path)
    test_entries, _ = load_dataset_t2(test_data_file_path)

    # Build the One-Hot Encoder on the training set vocabulary
    encoder = build_one_hot_encoder(train_sentences)

    human_ratings = []
    baseline_jaccard_ratings = []

    dataset_scores = {'human_ratings': [], 'jaccard_similarities': []}

    for entry in tqdm(test_entries):
        entry_sentences = entry['sentences']

        # Build vectors for each sentence
        sentences_accumulator = [entry_sentences[0]]
        vectors_accumulator = [encode_sentence(encoder, entry_sentences[0])]
        for index, entry_sentence in enumerate(entry_sentences[1:]):
            sentence = entry_sentence
            sentences_accumulator.append(sentence)
            vectors_accumulator.append(encode_sentence(encoder, sentence))

        # Compute vector pairs similarity
        jaccard_similarities = []

        for index, vector in enumerate(vectors_accumulator):
            if (index + 1) < len(vectors_accumulator):
                jaccard_similarities.append(jaccard_score(vector, vectors_accumulator[index + 1]))

        # Add data to global variables
        human_ratings.append(entry['avg_human_score'])
        baseline_jaccard_ratings.append(mean(jaccard_similarities))

        dataset_scores['human_ratings'].append(entry['avg_human_score'])
        dataset_scores['jaccard_similarities'].append(mean(jaccard_similarities))

    # Compute correlations
    pearson_jaccard = pearsonr(human_ratings, baseline_jaccard_ratings).statistic
    spearman_jaccard = spearmanr(human_ratings, baseline_jaccard_ratings).correlation

    # Build table
    table = PrettyTable()
    table.field_names = ['|Dataset|', 'Jaccard Pearson', 'Jaccard Spearman']
    table.title = 'T2 Correlations One-Hot Encodings'
    table.add_row([str(len(dataset_scores['human_ratings'])), "{:.2f}".format(pearson_jaccard), "{:.2f}".format(spearman_jaccard)])

    print(table.get_string())


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


def encode_sentence(encoder, sentence):
    # Encode input sentence with the passed encoder
    words = word_tokenize(sentence.lower())
    encoding = encoder.transform(array(words).reshape(len(words),1))
    encoding = sum(encoding)
    encoding[encoding > 1] = 1.0

    return encoding


def load_dataset_t2(data_file_path):

    data_df = pd.read_csv(data_file_path, sep='\t', header=0)

    all_sentences = []
    entries = []
    for index, row in data_df.iterrows():
        entry_id = row[0]
        text = row[1]
        avg_coherence = row[2]
        sentences = [x + '.' for x in text.split('.') if x != '']
        if len(sentences) == 1:
            sentences = [x + '?' for x in text.split('?') if x != '']
        all_sentences.extend(sentences)

        entries.append({
            'id': entry_id,
            'sentences': sentences,
            'avg_human_score': avg_coherence
        })

    return entries, all_sentences

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=Path)
    parser.add_argument("--test_file", type=Path)
    p = parser.parse_args()

    main(p.train_file, p.test_file)