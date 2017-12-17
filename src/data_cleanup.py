import pprint
import os
import json
import copy
from main import Network


current_path = os.path.dirname(os.path.realpath(__file__))


def create_test_data(trainfile, testfile):

    train_data = json.load(open(trainfile, 'r'))

    words = {}

    # Count the occurences
    for sentence in train_data:
        for word in sentence['words']:
            occurences = words.get(word['form'])
            if occurences is None:
                words[word['form']] = 1
            else:
                words[word['form']] = occurences + 1

    multi_words = []

    for word in words.keys():
        if words[word] > 1:
            multi_words.append(word)

    test_data = json.load(open(testfile, 'r'))

    # Create new data
    for sentence in test_data:
        for word in sentence['words']:
            if word['form'] not in multi_words:
                word['form'] = '<unk>'

    labels_in_trainset = json.load(open('../data/labels-en.json', 'r'))

    test_data_with_existing_labels = []

    for obj in test_data:
        add = True
        for word in obj['words']:
            if word['form'] == 'Administrator':
                add = False
                break

            if word['deprel'] not in labels_in_trainset.keys():
                add = False
                break
        if add:
            test_data_with_existing_labels.append(obj)

    with open('../data/toy_data_unk.json', 'w+') as f:
        f.write(json.dumps(test_data_with_existing_labels, indent=4))


def unknown_words_handler(filepath):
    '''
    Replace all words that occur only once in the dataset to <unk>
    '''
    data = json.load(open(filepath, 'r'))

    words = {}

    # Count the occurences
    for sentence in data:
        for word in sentence['words']:
            occurences = words.get(word['form'])
            if occurences is None:
                words[word['form']] = 1
            else:
                words[word['form']] = occurences + 1

    single_words = []

    for word in words.keys():
        if words[word] == 1:
            single_words.append(word)

    # Create new data
    for sentence in data:
        for word in sentence['words']:
            if word['form'] in single_words:
                word['form'] = '<unk>'

    with open('../data/en-ud-train-short_unk.json', 'w+') as f:
        f.write(json.dumps(data, indent=4))


def conllu_to_json(filepath=None):
    '''
    converts a conllu file to json
    '''
    # initialize
    text = []

    # Read conllu file
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    sentences = source.strip().split('\n\n')

    for sentence in sentences:
        temp_lines = sentence.strip().split('\n')

        sent = {}
        lines = []
        for line in temp_lines:
            words = line.split(' ')
            # Drop all lines beginning with #
            if words[0] == '#':
                if words[1] == 'sent_id':
                    sent['sent_id'] = words[3]
            else:
                lines.append(line)

        reject_sentence = False
        words = []
        for line in lines:
            words_list = line.split('\t')

            try:
                int(words_list[0])
            except ValueError:
                reject_sentence = True
                break

            word = {
                "id": words_list[0],
                "form": words_list[1],
                "lemma": words_list[2],
                "upostag": words_list[3],
                "xpostag": words_list[4],
                "feats": words_list[5],
                "head": words_list[6],
                "deprel": words_list[7],
                "deps": words_list[8],
                "misc": words_list[9]
            }
            words.append(word)

            if word['deprel'] == 'root':
                words.append({
                    "id": "0",
                    "form": "<ROOT> ",
                    "lemma": "<ROOT>",
                    "upostag": "ROOT",
                    "xpostag": "ROOT",
                    "feats": "_",
                    "head": "-1",
                    "deprel": "_",
                    "deps": "_",
                    "misc": "_"
                })

        if reject_sentence:
            continue

        sent['words'] = words
        text.append(sent)

    with open(filepath.replace('conllu', 'json'), 'w+') as f:
        f.write(json.dumps(text, indent=4))


def clean_conllu_file():
    with open('../data/conllu/hi-ud-train.conllu', 'r') as f:
        data = f.readlines()

    output = ''

    for line in data:
        if line == '\n':
            continue

        if line.startswith('# sent_id'):
            output += '\n'
        output += line

    with open('new-hi.conllu', 'w+') as f:
        f.write(output)


def shorten_data_set():
    data = json.load(open('../data/en-ud-train.json'))

    max_sentence_length = 12

    short_data = []

    for obj in data:
        if len(obj['words']) <= max_sentence_length:
            short_data.append(obj)

    with open('../data/en-ud-train-short.json', 'w+') as f:
        f.write(json.dumps(short_data, indent=4))


def create_test_data_new(w2i, testfile):

    test_data = json.load(open(testfile, 'r'))

    # Create new data
    for sentence in test_data:
        for word in sentence['words']:
            if word['form'] not in w2i.keys():
                word['form'] = '<unk>'

    labels_in_trainset = json.load(open('../data/labels.json', 'r'))

    test_data_with_existing_labels = []

    for obj in test_data:
        add = True
        for word in obj['words']:
            if word['deprel'] not in labels_in_trainset.keys():
                add = False
                break
        if add:
            test_data_with_existing_labels.append(obj)

    with open('../data/new-hi-ud-test_unk.json', 'w+') as f:
        f.write(json.dumps(test_data_with_existing_labels))


if __name__ == '__main__':
    pass
#     # preprocess_all_files()
#     data_path = current_path + '/../data/'

#     conllu_files = os.listdir(data_path + 'conllu/')
#     json_files = os.listdir(data_path + 'json/')

#     # Don't convert files that are already pre-processed
#     for json_file in json_files:
#         converted = json_file.replace('.json', '.conllu')
#         if converted in conllu_files:
#             conllu_files.remove(converted)

#     for conllu_file in conllu_files:
#         try:
#             conllu_to_json(data_path + 'conllu/' + conllu_file)
#             print('Converted: ', conllu_file)
#         except:
#             print('Failed to convert: ', conllu_file)

    # shorten_data_set()
    # clean_conllu_file()
    # conllu_to_json('../data/conllu/en-ud-test.conllu')
    # unknown_words_handler('../data/en-ud-train-short.json')
    # create_test_data('../data/en-ud-train-short_unk.json', '../data/json/toy_data.json')
