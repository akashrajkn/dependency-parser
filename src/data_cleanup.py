import pprint
import os
import json
import copy


current_path = os.path.dirname(os.path.realpath(__file__))

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

    with open('../data/en-ud-train_unk.json', 'w+') as f:
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

    # pprint.pformat(text)

    with open(filepath.replace('conllu', 'json'), 'w+') as f:
        f.write(json.dumps(text, indent=4))


if __name__ == '__main__':
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

    unknown_words_handler('../data/en-ud-train.json')
