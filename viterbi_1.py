"""
MP7: Viterbi 1
Implement an HMM-based spelling corrector using the basic Viterbi algorithm.
"""

import math
from collections import defaultdict, Counter

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial character probabilities, emission characters and transition character-to-character probabilities
    :param pairs:
    :return: intitial character probs, emission characters given character probs, transition of character to character probs
    """
    init_counts = Counter()
    emit_counts = defaultdict(Counter)
    trans_counts = defaultdict(Counter)

    if sentences is None:
        return {}, {}, {}

    states = set()

    for observed, hidden in sentences:
        if not hidden:
            continue

        states.update(hidden)

        init_counts[hidden[0]] += 1

        for obs_char, hid_char in zip(observed, hidden):
            emit_counts[hid_char][obs_char] += 1

        for prev_char, curr_char in zip(hidden[:-1], hidden[1:]):
            trans_counts[prev_char][curr_char] += 1

    if not states:
        return {}, {}, {}

    states = sorted(states)
    alpha_init = 0.01
    alpha_trans = 0.01
    alpha_emit = 0.01

    total_init = sum(init_counts.values())
    num_states = len(states)
    init_prob = {}
    denominator_init = total_init + alpha_init * num_states
    for state in states:
        init_prob[state] = (init_counts[state] + alpha_init) / denominator_init if denominator_init > 0 else 1.0 / num_states

    trans_prob = {}
    for prev_state in states:
        total = sum(trans_counts[prev_state].values())
        denominator = total + alpha_trans * num_states
        trans_prob[prev_state] = {}
        for curr_state in states:
            count = trans_counts[prev_state][curr_state]
            trans_prob[prev_state][curr_state] = (count + alpha_trans) / denominator if denominator > 0 else 1.0 / num_states

    emit_prob = {}
    for state in states:
        counts = emit_counts[state]
        total = sum(counts.values())
        unique = len(counts)
        denominator = total + alpha_emit * (unique + 1)
        emit_prob[state] = {}
        for obs_char, count in counts.items():
            emit_prob[state][obs_char] = (count + alpha_emit) / denominator if denominator > 0 else emit_epsilon
        emit_prob[state]['UNSEEN'] = alpha_emit / denominator if denominator > 0 else emit_epsilon

    return init_prob, emit_prob, trans_prob


def viterbi_stepforward(i, char, prev_prob, prev_predict_char_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param char: The i'th observed char
    :param prev_prob: A dictionary of chars to probs representing the max probability of getting to each character at in the
    previous column of the lattice
    :param prev_predict_char_seq: A dictionary representing the predicted character sequences leading up to the previous column
    of the lattice for each character in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each character, and the respective predicted character sequences
    """
    log_prob = {}
    predict_char_seq = {}

    states = list(emit_prob.keys())

    if i == 0:
        for state in states:
            emission_prob = emit_prob[state].get(char, emit_prob[state].get('UNSEEN', emit_epsilon))
            emission_log = math.log(emission_prob if emission_prob > 0 else emit_epsilon)
            prev_log = prev_prob.get(state, math.log(epsilon_for_pt))
            log_prob[state] = prev_log + emission_log
            prev_seq = prev_predict_char_seq.get(state, [])
            predict_char_seq[state] = prev_seq + [state]
        return log_prob, predict_char_seq

    for curr_state in states:
        emission_prob = emit_prob[curr_state].get(char, emit_prob[curr_state].get('UNSEEN', emit_epsilon))
        emission_log = math.log(emission_prob if emission_prob > 0 else emit_epsilon)
        best_log = -math.inf
        best_seq = []
        for prev_state, prev_log in prev_prob.items():
            transition_prob = trans_prob.get(prev_state, {}).get(curr_state, epsilon_for_pt)
            transition_log = math.log(transition_prob if transition_prob > 0 else epsilon_for_pt)
            candidate_log = prev_log + transition_log + emission_log
            if candidate_log > best_log:
                best_log = candidate_log
                prev_seq = prev_predict_char_seq.get(prev_state, [])
                best_seq = prev_seq + [curr_state]
        log_prob[curr_state] = best_log
        predict_char_seq[curr_state] = best_seq

    return log_prob, predict_char_seq


def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of pairs of typos and their correct spelling). E.g.,  [(typo1, word1), (typo2, word2), (typo3, word3)]
            test data (list of typos). E.g.,  [typo1, typo2, typo3]
    output: list of corrected typos.
            E.g., [corrected_typo1, corrected_typo2, corrected_typo3]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)

    predicts = []

    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_char_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = math.log(init_prob[t])
            else:
                log_prob[t] = math.log(epsilon_for_pt)
            predict_char_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_char_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_char_seq, emit_prob,trans_prob)

        if not log_prob:
            predicts.append('')
            continue

        best_state = max(log_prob, key=log_prob.get)
        predicts.append(''.join(predict_char_seq[best_state]))

    return predicts
