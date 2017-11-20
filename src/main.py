
def dependency_parser():

    for each sentence:
        x = sentence
        for each word:
            word_embedding = get_word_embeddings
            pos_embedding = get_pos_embeddings()
            x[i] = concatenate()
        
        r = LSTM_forward(x) + LSTM_backward(x)

        h_head = MLP_head(r)
        h_dep = MLP_dep(r)

        # score
        s[i] = h_head.T * U_1 * h_dep[i] + h_head.T * u_2 
