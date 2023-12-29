import json
import math
import random

import numpy as np


def calculate_dcg(relevance_scores, alpha, perspectives):
    """
    Compute the DCG given a list of "relevance-scores" and "perspectives".
    :param relevance_scores: a list of binary scores (1/0). Each element indicates whether the argument at that rank
    was relevant (for the question) or not.
    :param alpha: a float between 0 and 1, that penalizes 'redundancy'.
     We would simplfiy redundancy to "i already have seen an argument from that perspective".
    :param perspectives: a list that contains the 'perspective' for each argument at that rank. E.g. we can define perspective
    as the political spectrum where the argument was coming from (or party or any other socio-demo variable).
    :return: dcg score
    """
    dcg = 0.0
    seen_perspectives = set()
    for i, (relevance, perspective) in enumerate(zip(relevance_scores, perspectives)):
        # Apply penalty for redundancy
        penalty = (1 - alpha) if perspective in seen_perspectives else 1
        seen_perspectives.add(perspective)
        # Calculate DCG
        dcg += (relevance * penalty) / math.log2(i + 2)  # i+2 because the rank starts counting at 1 and not at 0
    return dcg


# Function to calculate IDCG, assuming the ideal case where there's no redundancy, that is all top-k arguments come from
# different perspectives
def calculate_idcg(relevance_scores):
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores))
    return idcg


def alpha_ndcg(relevance_scores, perspectives, alpha):
    """
    Compute the alpha-nDCG given a list of "relevance-scores" and "perspectives".
    :param relevance_scores: a list of binary scores (1/0). Each element indicates whether the argument at that rank
    was relevant (for the question) or not.
    :param alpha: a float between 0 and 1, that penalizes 'redundancy'.
     We would simplfiy redundancy to "i already have seen an argument from that perspective".
    :param perspectives: a list that contains the 'perspective' for each argument at that rank. E.g. we can define perspective
    as the political spectrum where the argument was coming from (or party or any other socio-demo variable).
    :return: alpha-nDCG score
    """
    # Calculate DCG
    dcg = calculate_dcg(relevance_scores, alpha, perspectives)
    # Calculate IDCG
    idcg = calculate_idcg(relevance_scores)
    # Calculate alpha-nDCG
    alpha_ndcg = dcg / idcg if idcg > 0 else 0
    return alpha_ndcg


def s_precision_at_k(relevance_scores, perspectives, k):
    """
    Compute the S-Precision@k given a list that defines wether the argument was relevant to the question and
    a list of labels that define different "perspectives" (e.g. which political spectrum the author of the argument has).
    :param relevance_scores: a list of binary scores, where each element indicates whether the argument at that rank
    was relevant (for the question) or not.
    :param perspectives: a list of labels (e.g. political spectrum) for each argument at that rank.
    :param k: the cut-off point
    :return: precision@k which is a float between 0.0 and 1.0
    """
    # Get the top-k relevance scores and perspectives
    relevance_scores = relevance_scores[:k]
    perspectives = perspectives[:k]
    # Get the perspectives of those that are relevant
    perspectives = [perspective for relevance, perspective in zip(relevance_scores, perspectives) if relevance == 1]
    unique_perspectives = set(perspectives)
    # Calculate S-Precision@k
    s_precision = len(unique_perspectives) / k
    return s_precision


def example_perfect_scores():
    # a minin example that computes the three scores for a perfect ranking (all relevant, all diverse)
    alpha = 0.5

    # Hypothetical binary relevance scores for top 5 and top 32
    binary_relevance_scores_5 = [1] * 5
    binary_relevance_scores_32 = [1] * 32

    # Hypothetical unique perspectives for top 5 and top 32 (assuming no redundancy within the top-k)
    unique_perspectives_5 = list(range(1, 6))  # Just a placeholder for unique perspectives
    unique_perspectives_32 = list(range(1, 33))  # Same here

    print("Alpha-nDCG@5: ", alpha_ndcg(binary_relevance_scores_5, unique_perspectives_5, alpha))
    print("Alpha-nDCG@32: ", alpha_ndcg(binary_relevance_scores_32, unique_perspectives_32, alpha))
    print("S-Precision@5: ", s_precision_at_k(binary_relevance_scores_5, unique_perspectives_5, 5))


def get_relevance_and_perspectives(items, question_id):
    """
    Given a list of items (rank = position in list) return two lists that indicate whether that item was relevant to the
    question and the perspective of each item
    :param items:
    :return:
    """
    relevance_list = [1 if item["question_id"] == question_id else 0 for item in items]
    perspectives_list = [item["political_spectrum"] for item in items]
    return relevance_list, perspectives_list


def random_example_real_data(path_to_training_data):
    average_sprecision_at_10 = []
    average_sprecision_at_5 = []
    average_ndgc = []
    with open(path_to_training_data, "r") as f:
        # read in the list of items, each item is an argument that belongs to a unique question, with a unique stance
        # and a unique socio-demo profile
        data = [json.loads(line) for line in f.readlines()]
    # for each question, sample 70% of arguments that fit to that question and 30% random
    all_question_ids = set([item["question_id"] for item in data])
    question2sample = {}
    for question_id in all_question_ids:
        question_data = [item for item in data if item["question_id"] == question_id]
        # get the number of unique diverse perspectives that are relevant for that question id
        unique_possible_persepectives = len(set([item["political_spectrum"] for item in question_data]))
        non_question_data = [item for item in data if item["question_id"] != question_id]
        total_size_relevant = len(question_data)
        if total_size_relevant >= 70:
            sample_current_question = random.sample(question_data, 70) + random.sample(non_question_data, 30)

        else:
            left_to_sample = 100 - total_size_relevant
            sample_current_question = question_data + random.sample(non_question_data, left_to_sample)
        # shuffle the sample
        random.shuffle(sample_current_question)
        question2sample[question_id] = sample_current_question
        relevance_list, perspective_list = get_relevance_and_perspectives(sample_current_question, question_id)
        alpha = 0.5
        ndgc = alpha_ndcg(relevance_list, perspective_list, alpha)
        sprecision_at_10 = s_precision_at_k(relevance_list, perspective_list, 10)
        sprecision_at_5 = s_precision_at_k(relevance_list, perspective_list, 5)
        max_sprecision_at_10 = unique_possible_persepectives / 10
        # normalize sprecision by max_sprecision
        normalized_sprecision_at_10 = sprecision_at_10 / max_sprecision_at_10
        normalized_sprecision_at_5 = sprecision_at_5 / (unique_possible_persepectives / 5)
        average_ndgc.append(ndgc)
        average_sprecision_at_10.append(normalized_sprecision_at_10)
        average_sprecision_at_5.append(normalized_sprecision_at_5)
    print("mean ndgc: %.2f" % np.mean(average_ndgc))
    print("mean S-Precision @ 10: %.2f" % np.mean(average_sprecision_at_10))
    print("mean S-Precision @ 5: %.2f" % np.mean(average_sprecision_at_5))


if __name__ == '__main__':
    random_example_real_data("./train_with_all_socios.jsonl")
