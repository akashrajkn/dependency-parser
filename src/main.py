def preprocess():
	pass


def word_embedding(corpus):

	return word



def dependency_parser():

	word_embeddings = word_embedding(corpus)
	pos_embeddings = word_embedding(corpus_pos_tags)

    # initialize U_1

    for each sentence:
        x = sentence
        for each word:
            # word_embedding = get_word_embeddings
            word_embedding = word_embeddings[word]

            # pos_embedding = get_pos_embeddings()
            pos_embedding = word_embedding[pos]
            x[i] = concatenate()

        r = LSTM_forward(x) + LSTM_backward(x)

        # score
        h_head = MLP_head(r)
        h_dep = MLP_dep(r)

        s[i] = h_head.T * U_1 * h_dep[i] + h_head.T * u_2
