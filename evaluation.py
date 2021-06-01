import nltk
import subprocess
import os
from rouge import Rouge
from collections import Counter
import numpy as np
import itertools
import numpy as np

def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _split_into_words(sentences):
    return list(itertools.chain(*[list(_.replace(' ', '')) for _ in sentences]))



def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    words = _split_into_words(sentences)
    return _get_ngrams(n, words)


def _len_lcs(x, y):
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _recon_lcs(x, y):
    i, j = len(x), len(y)
    table = _lcs(x, y)

    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
    return recon_tuple


def rouge_n(evaluated_sentences, reference_sentences, n=2):
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    # Handle edge case. This isn't mathematically correct, but it's good enough
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    # return overlapping_count / reference_count
    return f1_score, precision, recall


def _f_p_r_lcs(llcs, m, n):
    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / (r_lcs + 1e-12)
    num = (1 + (beta**2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta**2) * p_lcs)
    f_lcs = num / (denom + 1e-12)
    return f_lcs, p_lcs, r_lcs


def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")
    reference_words = _split_into_words(reference_sentences)
    evaluated_words = _split_into_words(evaluated_sentences)
    m = len(reference_words)
    n = len(evaluated_words)
    lcs = _len_lcs(evaluated_words, reference_words)
    return _f_p_r_lcs(lcs, m, n)


def _union_lcs(evaluated_sentences, reference_sentence):
    if len(evaluated_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    lcs_union = set()
    reference_words = _split_into_words([reference_sentence])
    combined_lcs_length = 0
    for eval_s in evaluated_sentences:
        evaluated_words = _split_into_words([eval_s])
        lcs = set(_recon_lcs(reference_words, evaluated_words))
        combined_lcs_length += len(lcs)
        lcs_union = lcs_union.union(lcs)

    union_lcs_count = len(lcs_union)
    union_lcs_value = union_lcs_count / combined_lcs_length
    return union_lcs_value


def rouge_l_summary_level(evaluated_sentences, reference_sentences):
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    # total number of words in reference sentences
    m = len(_split_into_words(reference_sentences))

    # total number of words in evaluated sentences
    n = len(_split_into_words(evaluated_sentences))

    union_lcs_sum_across_all_references = 0
    for ref_s in reference_sentences:
        union_lcs_sum_across_all_references += _union_lcs(evaluated_sentences,
                                                      ref_s)
    return _f_p_r_lcs(union_lcs_sum_across_all_references, m, n)


def rouge(hypotheses, references):
    """Calculates average rouge scores for a list of hypotheses and
    references"""

    # Filter out hyps that are of 0 length
    # hyps_and_refs = zip(hypotheses, references)
    # hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0]
    # hypotheses, references = zip(*hyps_and_refs)

    # Calculate ROUGE-1 F1, precision, recall scores
    rouge_1 = [
      rouge_n([hyp], [ref], 1) for hyp, ref in zip(hypotheses, references)
    ]
    rouge_1_f, rouge_1_p, rouge_1_r = map(np.mean, zip(*rouge_1))

    # Calculate ROUGE-2 F1, precision, recall scores
    rouge_2 = [
      rouge_n([hyp], [ref], 2) for hyp, ref in zip(hypotheses, references)
    ]
    rouge_2_f, rouge_2_p, rouge_2_r = map(np.mean, zip(*rouge_2))

    # Calculate ROUGE-L F1, precision, recall scores
    rouge_l = [
      rouge_l_sentence_level([hyp], [ref])
      for hyp, ref in zip(hypotheses, references)
    ]
    rouge_l_f, rouge_l_p, rouge_l_r = map(np.mean, zip(*rouge_l))

    return {
      "rouge_1/f_score": rouge_1_f,
      "rouge_1/r_score": rouge_1_r,
      "rouge_1/p_score": rouge_1_p,
      "rouge_2/f_score": rouge_2_f,
      "rouge_2/r_score": rouge_2_r,
      "rouge_2/p_score": rouge_2_p,
      "rouge_l/f_score": rouge_l_f,
      "rouge_l/r_score": rouge_l_r,
      "rouge_l/p_score": rouge_l_p,
    }

def rouge_wrapper(prediction_file, ref_file):
    hypotheses = [_ for _ in open(prediction_file, 'r', encoding='utf-8')]
    references = [_ for _ in open(ref_file, 'r', encoding='utf-8')]
    return rouge(hypotheses, references)

def Bleu(cand_file, ref_file, lang="de", bpe=False):
    print(ref_file)
    temp = "result.txt"
    command = "perl multi-bleu.perl " + \
        ref_file + "<" + cand_file + "> " + temp
    print(command)
    try:
        subprocess.call(command, shell=True)
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)

    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
   # print(len(result))
   # print the score
   # print(result)

    return float(result.split()[2][:-1])

def div_distinct(cand):
    list_hyp = list()
    with open(cand, 'r') as f1:
        for i in f1.readlines():
            list_hyp.append(np.array(i.split()))
    list_hyp = np.array(list_hyp)
    batch_size = len(list_hyp)
    n_unigrams, n_bigrams, n_unigrams_total , n_bigrams_total = 0. ,0., 0., 0.
    unigrams_all, bigrams_all = Counter(), Counter()
    for b in range(batch_size):
        buf_hyp = list_hyp[b]
        unigrams_all.update([tuple(buf_hyp[i:i+1]) for i in range(len(list_hyp[b]))])
        bigrams_all.update([tuple(buf_hyp[i:i+2]) for i in range(len(list_hyp[b])-1)])
        n_unigrams_total += len(list_hyp[b])
        n_bigrams_total += max(0, len(list_hyp[b])-1)
    inter_dist1 = (len(unigrams_all.items())+1e-12)/(n_unigrams_total+1e-5)
    inter_dist2 = (len(bigrams_all.items())+1e-12)/(n_bigrams_total+1e-5)
    return inter_dist1, inter_dist2


cand_file = 'semeval.result2'      # abstract.transformer.step80000.top1'
ref_file = 'dataset/semeval.title'
print('calculating bleu...')
bleu = Bleu(cand_file,ref_file)
print('calculating rouge...')
rouge = rouge_wrapper(cand_file, ref_file)
print('calculating distinct...')
distinct1, distinct2 = div_distinct(cand_file)

print('bleu', bleu)
print('rouge', rouge)
#print('distinct1', distinct1)
#print('distinct2', distinct2)
