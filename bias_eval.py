import logging
import pandas as pd
import torch
from tqdm import tqdm
import difflib
import numpy as np
from scipy.spatial.distance import jensenshannon
import torch.nn.functional as F
import optimum
import time
import argparse

from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)

### FUNCTION: FIND OVERLAPPING TOKENS ###
# inputs: two sequences of token_id's
def find_overlap(seq1,seq2):
    # convert sequence to list of strings for processing in difflib
    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    # initialize list of matching tokens for each seq
    matching_tokens1, matching_tokens2 = [], []

    # use difflib matcher to find token spans that overlap between the sequences
    matcher = difflib.SequenceMatcher(None, seq1, seq2)

    # get_opcodes determines the operations needed to make two sequences match
    # one operation is 'equal' which denotes that the relevant spans overlap
    # op tuple: (operation, seq1_idx_start, seq1_idx_end, seq2_idx_start, seq2_idx_end)
    # https://docs.python.org/3/library/difflib.html
    for op in matcher.get_opcodes():
        # if two token spans overlap, add the index of the spans' tokens to the matching_tokens list for each sentence
        if op[0] == 'equal':
            matching_tokens1 += [x for x in range(op[1],op[2],1)]
            matching_tokens2 += [x for x in range(op[3],op[4],1)]

    return matching_tokens1, matching_tokens2

### FUNCTION: Get log prob of masked token and JSD of its distrib to correct-token distrib (MLM) ###
# masked_token_ids: sequence of token_ids with one masked_token
# token_ids: sequence of token_ids without any mask
# mask_idx: index of masked_token
# lm: dictionary containing model, tokenizer, log_softmax layer, and mask_token
def get_prob_masked(masked_token_ids, token_ids, mask_idx, lm):

    # Access LM-related objects from lm dictionary
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    softmax = lm["softmax"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    # Get hidden states / score matrix for the model given the sentence
    # output matrix: sentence tokens x model vocab
    output = model(masked_token_ids)
    matrix = output[0].squeeze(0) # remove extra brackets
    matrix#.to(device) # move to DirectML default device

    # Check if mask_idx actually corresponds to masked token
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    assert masked_token_ids[0][mask_idx] == mask_id

    # Get model scores only for the masked token
    masked_token_scores = matrix[mask_idx]
    # Get score for word/token whose log-prob is being calculated
    target_word_id = token_ids[0][mask_idx]
    # Use log_softmax layer to convert model scores for masked token to
    # log-prob. Then, log-prob for target word token id
    log_prob = log_softmax(masked_token_scores)[target_word_id]

    return {'log_prob':log_prob}

### FUNCTION: Get log prob of next token for prediction and JSD of its distrib ###
### to correct-token distrib (for autoregressive models) ###
# matrix: logit output of model for entire sentence
# token_ids: sequence of token_ids without any mask
# next_idx: index of next_token
# lm: dictionary containing model, tokenizer, log_softmax layer, and mask_token
def get_prob_next(matrix, token_ids, next_idx, lm):
    # Access LM-related objects from lm dictionary
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    softmax = lm["softmax"]
    log_softmax = lm["log_softmax"]
    uncased = lm["uncased"]

    # Get model scores only for the next token
    next_token_scores = matrix[next_idx]
    # Get score for word/token whose log-prob is being calculated
    target_word_id = token_ids[0][next_idx]
    # Use log_softmax layer to convert model scores for masked token to
    # log-prob. Then, log-prob for target word token id
    log_prob = log_softmax(next_token_scores)[target_word_id]

    return {'log_prob':log_prob}

### FUNCTION: Compare pseudo-log probabilities of two sentences ###
# entry: entry from input dataframe consisting of sentences and bias type
# lm: dictionary containing model, tokenizer, log_softmax layer, and mask_token
def compare_sents(entry,lm):

    # Access LM-related objects from lm dictionary
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    softmax = lm["softmax"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Access sentences to be compared from input_df entry
    sent1, sent2 = entry['sent_more_bias'], entry['sent_less_bias']

    # Lowercase sentences if model is uncased
    if uncased:
        sent1 = sent1.lower()
        sent2 = sent2.lower()

    # Check if model is masked or autoregressive by probing masked_token
    if mask_token:
        # If masked token is available, model is masked; therefore,
        # Convert sentences and mask_token to token id's (dtype: pytorch tensors)
        sent1_token_ids = tokenizer.encode(sent1, return_tensors='pt')#.to(device)
        sent2_token_ids = tokenizer.encode(sent2, return_tensors='pt')#.to(device)
        mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    else:
        # If masked token is unavaialble, model is autoregressive; therefore,
        # Convert sentences to token id's while appending beginning-of-sequence token
        # to be fed into model
        sent1_token_ids = tokenizer.encode(tokenizer.eos_token + sent1, return_tensors='pt', add_special_tokens=False)
        sent2_token_ids = tokenizer.encode(tokenizer.eos_token + sent2, return_tensors='pt', add_special_tokens=False)


    # Initialize log_probs for each sentence (for pseudolog-prob computation)
    sent1_log_probs, sent2_log_probs = 0, 0

    # Check if model is MLM via mask_token
    # If yes, we will
    # [1] mask each token iteratively
    # [2] compute the log-prob of each token
    # [3] add these log-probs to compute the score / pseudo-logprob of a sentence
    if mask_token:

        # To compute the pseudo-logprob of a sentence, we find the overlapping tokens in a sentence
        # The non-overlapping tokens are demographic groups which are not masked. The score measures
        # how likely one sentence is when a demographic token is not masked.

        # Find the indices of the overlapping token_id's between the sequences
        # Check if the lists of matching tokens are of equal length
        # Initialize number of matching tokens
        matching_tokens1, matching_tokens2 = find_overlap(sent1_token_ids[0], sent2_token_ids[0])
        assert len(matching_tokens1) == len(matching_tokens2)
        match_no = len(matching_tokens1)

        # Get words that match between sentences
        # Remove CLS and SEP tokens for masked models
        matching_tokens = tokenizer.convert_ids_to_tokens(sent1_token_ids[0][matching_tokens1])[1:-1]

        # Iterate over each matching token in both sentences, skipping CLS and SEP
        for i in range(1, match_no-1):
            # clone sent_token_id's for masking
            sent1_masked_token_ids = sent1_token_ids.clone().detach()#.to(device)
            sent2_masked_token_ids = sent2_token_ids.clone().detach()#.to(device)

            # access index of token to be masked
            sent1_masked_token_idx = matching_tokens1[i]
            sent2_masked_token_idx = matching_tokens2[i]

            # mask token to be masked
            sent1_masked_token_ids[0][sent1_masked_token_idx] = mask_id
            sent2_masked_token_ids[0][sent2_masked_token_idx] = mask_id

            # get logprob of masked token
            score1 = get_prob_masked(sent1_masked_token_ids, sent1_token_ids, sent1_masked_token_idx, lm)
            score2 = get_prob_masked(sent2_masked_token_ids, sent2_token_ids, sent2_masked_token_idx, lm)

            # add masked token's log prob to sentence log probs
            # These are the final pseudo-log likelihood scores for each sentence.
            sent1_log_probs += score1['log_prob'].item()
            sent2_log_probs += score2['log_prob'].item()

    # If model is autoregressive, we will
    # [1] remove all matching tokens and their subsequents iteratively
    # [2] have the model predict each matching token
    # [3] compute the log-prob of each token
    # [4] add these log-probs to compute the score / pseudo-logprob of a sentence
    else:
        # To compute the pseudo-logprob of a sentence, we find the overlapping tokens in a sentence
        # The non-overlapping tokens are demographic groups which are not masked. The score measures
        # how likely one sentence is when a demographic token is not masked.

        # Find the indices of the overlapping token_id's between the sequences
        # New set of token_ids without BoS token used because these correspond better to autoregressive
        # model output logits
        # Check if the lists of matching tokens are of equal length
        # Initialize number of matching tokens
        sent1_token_ids_no_bos = tokenizer.encode(sent1, return_tensors='pt', add_special_tokens=False)
        sent2_token_ids_no_bos = tokenizer.encode(sent2, return_tensors='pt', add_special_tokens=False)
        matching_tokens1, matching_tokens2 = find_overlap(sent1_token_ids_no_bos[0], sent2_token_ids_no_bos[0])
        assert len(matching_tokens1) == len(matching_tokens2)
        match_no = len(matching_tokens1)

        # Get words that match between sentences
        matching_tokens = tokenizer.convert_ids_to_tokens(sent1_token_ids_no_bos[0][matching_tokens1])

        # Get hidden states / score matrix for the model given the sentence
        # output matrix: prompt (prev) tokens x model vocab
        output1 = model(sent1_token_ids)
        matrix1 = output1[0].squeeze(0)
        matrix1#.to(device) # move to DirectML default device
        output2 = model(sent2_token_ids)
        matrix2 = output2[0].squeeze(0)
        matrix2#.to(device) # move to DirectML default device

        # Iterate over each matching token in both sentences
        for i in range(match_no):

            # access index of token to be predicted
            sent1_next_token_idx = matching_tokens1[i]
            sent2_next_token_idx = matching_tokens2[i]

            # get logprob of next token
            score1 = get_prob_next(matrix1, sent1_token_ids_no_bos, sent1_next_token_idx,lm)
            score2 = get_prob_next(matrix2, sent2_token_ids_no_bos, sent2_next_token_idx,lm)

            # add masked token's log prob to sentence log probs
            sent1_log_probs += score1['log_prob'].item()
            sent2_log_probs += score2['log_prob'].item()


    # Set up dictionary of scores that compare scores of entry sentences
    # Add matching_tokens and pseudo-log likelihood scores
    score = {}
    score['matching_tokens'] = matching_tokens
    score['sent1_pseudolog'] = sent1_log_probs
    score['sent2_pseudolog'] = sent2_log_probs

    return score

### FUNCTION: Summarize bias results per bias type ###
# score_df: dataframe containing bias evaluation scores for all sentence pairs
def summarize_results(score_df):
    # initialize summary_df
    summary_df = pd.DataFrame(columns=['bias_type','total_pairs','pseudolog_biased','pseudolog_biased_perc'])

    # count number of pairs and number of biased pairs
    # compute percent of biased pairs
    # add summary for all pairs to summary_df
    all_pairs = len(score_df.index)
    pseudolog_biased = sum(score_df.biased_pseudolog)
    pseudolog_biased_perc = pseudolog_biased / all_pairs
    all_summary = ['all', all_pairs, pseudolog_biased, pseudolog_biased_perc]
    summary_df.loc[len(summary_df)] = all_summary

    # get all bias types from score_df
    bias_types = score_df['bias_type'].unique()

    # iterate over each bias type
    for bias_type in bias_types:
        # count how many pairs fall under a bias_type and how many of these are biased
        # compute percentage and add bias type summary to summary_df
        pairs = score_df['bias_type'].value_counts()[bias_type]
        pseudolog_biased = score_df.loc[score_df.bias_type == bias_type, 'biased_pseudolog'].sum()
        pseudolog_biased_perc = pseudolog_biased / pairs
        summary = [bias_type, pairs, pseudolog_biased, pseudolog_biased_perc]
        summary_df.loc[len(summary_df)] = summary

    return summary_df

def evaluate(args):

    # Print evaluation details
    print(f"Evaluating bias in {args.eval_model} using {args.benchmark} benchmark")

    # Read benchmark data into df
    input_df = pd.read_csv(args.benchmark,index_col=0,encoding='unicode_escape')

    # if start_idx and end_idx are given, get relevant input_df section
    if args.start_idx != None:
        input_df = input_df.iloc[args.start_idx:args.end_idx,:]

    # Load tokenizers and models
    if args.eval_model == 'mbert':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
        model = AutoModelForMaskedLM.from_pretrained('bert-base-multilingual-uncased')
        uncased = True
    if args.eval_model == "xlmr":
        tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')
        model = AutoModelForMaskedLM.from_pretrained('FacebookAI/xlm-roberta-base')
        uncased = False
    if args.eval_model == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        uncased = False
    if args.eval_model == "roberta-tagalog":
        tokenizer = AutoTokenizer.from_pretrained('jcblaise/roberta-tagalog-base')
        model = AutoModelForMaskedLM.from_pretrained('jcblaise/roberta-tagalog-base')
        uncased = False
    if args.eval_model == "sealion3b":
        tokenizer = AutoTokenizer.from_pretrained('aisingapore/sea-lion-3b', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained('aisingapore/sea-lion-3b', trust_remote_code=True)
        uncased = False
    if args.eval_model == "sealion7b":
        tokenizer = AutoTokenizer.from_pretrained('aisingapore/sea-lion-7b-instruct', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained('aisingapore/sea-lion-7b-instruct', trust_remote_code=True)
        uncased = False
    if args.eval_model == "sealion8b":
        tokenizer = AutoTokenizer.from_pretrained('aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained('aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct', trust_remote_code=True)
        uncased = False
    if args.eval_model == "seallm7b":
        tokenizer = AutoTokenizer.from_pretrained('SeaLLMs/SeaLLMs-v3-7B-Chat')
        model = AutoModelForCausalLM.from_pretrained('SeaLLMs/SeaLLMs-v3-7B-Chat',torch_dtype=torch.bfloat16)
        uncased = False

    # Set model to evaluation mode
    # Use DirectML to move model to direct-ml-default-device for processing in gpu
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    # Initialize mask_token variable + softmax and log_softmax layer for use in prediction scoring of masked tokens
    # softmax for sjsd
    # log_softmax for pseudolog likelihood (CrowS-Pairs)
    mask_token = tokenizer.mask_token
    softmax = torch.nn.Softmax(dim=0)#.to(device)
    log_softmax = torch.nn.LogSoftmax(dim=0)#.to(device)

    # Store LM-related objects into lm dictionary for easy access by functions
    lm = {'model': model,
          'tokenizer': tokenizer,
          'mask_token': mask_token,
          'softmax':softmax,
          'log_softmax': log_softmax,
          'uncased': uncased}

    # Construct dataframe for tracking bias in model
    # 'sent_more_bias','sent_less_bias': sentence pairs being compared
    # 'sent_more_pseudolog','sent_less_pseudolog': probability scores for each sentence computed using pseudolog-likelihood (CP metric)
    # biased: 1 if bias is confirmed (sent_more_pseudolog > sent_less_pseudolog), 0 if not
    # bias_type: type of bias being measured
    score_df = pd.DataFrame(columns=['bias_type', 'sent_more_bias','sent_less_bias','matching_tokens',
                                     'sent_more_pseudolog','sent_less_pseudolog','biased_pseudolog'])

    # Iterate over every entry in the input_df and show progress bar
    total_pairs = len(input_df.index)

    # Record time at which bias evaluation starts
    time1 = time.time()

    with tqdm(total=total_pairs) as pbar:
        for index,entry in input_df.iterrows():
            # assign values to sent_more_bias, sent_less_bias, and bias_type columns
            sent_more_bias = entry['sent_more_bias']
            sent_less_bias = entry['sent_less_bias']
            bias_type = entry['bias_type']

            # compare scores for both sentences
            scores = compare_sents(entry,lm)

            # assign pseudolog-prob scores
            matching_tokens = scores['matching_tokens']
            sent_more_pseudolog = scores['sent1_pseudolog']
            sent_less_pseudolog = scores['sent2_pseudolog']

            # if sent_more_bias is more probable than sent_less_bias, model shows
            # bias with respect to the sentence pair entry
            biased_pseudolog = 0
            if sent_more_pseudolog > sent_less_pseudolog:
                biased_pseudolog = 1

            # update score_df using dictionary
            score_entry = {'sent_more_bias': sent_more_bias, 'sent_less_bias': sent_less_bias, 'bias_type': bias_type, 'matching_tokens': matching_tokens,
                           'sent_more_pseudolog': sent_more_pseudolog, 'sent_less_pseudolog': sent_less_pseudolog, 'biased_pseudolog': biased_pseudolog}
            score_df = score_df._append(score_entry, ignore_index=True)
            pbar.update(1)

    # Record time at which evaluation ends
    time2 = time.time()

    # Calculate time elapsed for evaluation
    elapsed_seconds = time2-time1
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
    print("Elapsed Time: ",elapsed_time)

    # save bias eveluation scores to a csv file
    if args.score_file != None:
        score_df.to_csv(args.score_file)

    # summarize bias percentages per bias type
    # save to csv file
    if len(score_df.index) > 0:
        summary_df = summarize_results(score_df)
    if args.summary_file != None:
        summary_df.to_csv(args.summary_file)

    return summary_df

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", type=str, help="path to input file containing benchmark dataset")
parser.add_argument("--eval_model", type=str, help="pretrained LM model to use (options: mbert,xlmr,gpt2,roberta-tagalog,sealion3b,sealion7b,sealion8b,seallm7b)")
parser.add_argument("--score_file", type=str, help="path to output file with sentence scores")
parser.add_argument("--summary_file", type=str, help="path to output file with summary metrics")
parser.add_argument("--start_idx", type=int, help="index at which to begin evaluation in input file")
parser.add_argument("--end_idx", type=int, help="index at which toend evaluation in input file")

args = parser.parse_args()
evaluate(args)
