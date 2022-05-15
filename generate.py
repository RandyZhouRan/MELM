import torch
import time
import numpy as np
import random
import json
import argparse
import os

from torch.utils.data import DataLoader

from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM

from data_gen import Data

label_map = {"PAD":0, "O": 1, "B-PER":2, "I-PER":3, "B-ORG":4, "I-ORG":5,
             "B-LOC":6, "I-LOC":7, "B-MISC":8, "I-MISC":9}

def aug(entity_model, o_model, iterator, k, sub_idx):

    print("Augmenting sentences with MELM checkpoint ...")

    assert sub_idx <= k
    entity_model.eval()
    o_model.eval()

    batches_of_entity_aug = []
    batches_of_o_aug = []
    batches_of_total_aug = [] # Combine both entity aug and O aug
    batches_of_input = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            batch_start = time.time()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids, entity_masked_ids, entity_mask, o_masked_ids, o_mask = batch

            # Entity aug
            entity_outputs = entity_model(entity_masked_ids, input_mask, labels=input_ids)
            entity_logits = entity_outputs.logits

            top_logits, topk = torch.topk(entity_logits, k=k, dim=-1)
            #torch. set_printoptions(profile="full")
            #print("indices \n", topk[:,:30,:5])
            if sub_idx != -1:
                sub = topk[:, :, sub_idx]
            else: # random picking from topk
                gather_idx = torch.randint(1, k, (topk.shape[0], topk.shape[1], 1)).to(device)
                sub = torch.gather(topk, -1, gather_idx).squeeze(-1)

            entity_aug = torch.where(entity_mask == 1, sub, entity_masked_ids)
            batches_of_entity_aug.append(entity_aug)

            # O aug
            o_outputs = o_model(o_masked_ids, input_mask, labels=input_ids)
            o_logits = o_outputs.logits

            top_logits, topk = torch.topk(o_logits, k=k, dim=-1)
            sub = topk[:, :, 0] # Always use top prediction for O-masks
            o_aug = torch.where(o_mask == 1, sub, o_masked_ids)
            batches_of_o_aug.append(o_aug)
            
            # Combine both entity aug and O aug
            total_aug = torch.where(o_mask == 1, sub, entity_aug)
            batches_of_total_aug.append(total_aug)

            batches_of_input.append(input_ids)

    entity_aug_tensor = torch.cat(batches_of_entity_aug, dim=0)
    o_aug_tensor = torch.cat(batches_of_o_aug, dim=0)
    total_aug_tensor = torch.cat(batches_of_total_aug, dim=0)
    input_tensor = torch.cat(batches_of_input, dim=0)
    
    assert entity_aug_tensor.shape == input_tensor.shape
    assert o_aug_tensor.shape == input_tensor.shape
    assert total_aug_tensor.shape == input_tensor.shape
    
    return entity_aug_tensor, o_aug_tensor, total_aug_tensor, input_tensor

def decode(entity_aug_tensor, o_aug_tensor, total_aug_tensor, input_tensor, tokenizer):

    print("Decoding augmented ids ...")
    entity_aug_text = []
    o_aug_text = []
    total_aug_text = []

    for entity_aug_ids, o_aug_ids, total_aug_ids, input_ids in zip(entity_aug_tensor.tolist(), o_aug_tensor.tolist(), total_aug_tensor.tolist(), input_tensor.tolist()):
        input_subs = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)
        entity_aug_subs = tokenizer.convert_ids_to_tokens(entity_aug_ids, skip_special_tokens=False)[1:len(input_subs)+1] # Cater for cases when last token is predicted as EOS and thus wrongly removed
        o_aug_subs = tokenizer.convert_ids_to_tokens(o_aug_ids, skip_special_tokens=False)[1:len(input_subs)+1]
        total_aug_subs = tokenizer.convert_ids_to_tokens(total_aug_ids, skip_special_tokens=False)[1:len(input_subs)+1]

        assert len(entity_aug_subs) == len(input_subs), f"input {input_subs} \n {input_ids}\n entity_aug {entity_aug_subs}\n{entity_aug_ids}"
        entity_word, o_word, total_word = '', '', ''
        entity_aug_sent, o_aug_sent, total_aug_sent = [], [], []

        special_masks = ['<B-PER>', '<I-PER>', '<B-ORG>', '<I-ORG>', '<B-LOC>', '<I-LOC>', '<B-MISC>', '<I-MISC>', '<En>', '<De>', '<Es>', '<Nl>']
        for entity_aug_sub, o_aug_sub, total_aug_sub, input_sub in zip(entity_aug_subs, o_aug_subs, total_aug_subs, input_subs):
            if input_sub[0] == '▁' or  input_sub in special_masks:
                if entity_word != '':
                    entity_aug_sent.append(entity_word)
                entity_word = entity_aug_sub
                if o_word != '':
                    o_aug_sent.append(o_word)
                o_word = o_aug_sub
                if total_word != '':
                    total_aug_sent.append(total_word)
                total_word = total_aug_sub
            else:
                entity_word += entity_aug_sub
                o_word += o_aug_sub
                total_word += total_aug_sub
        # Cater for last word in the sentence
        entity_aug_sent.append(entity_word)
        o_aug_sent.append(o_word)
        total_aug_sent.append(total_word)

        entity_aug_text.append(entity_aug_sent)
        o_aug_text.append(o_aug_sent)
        total_aug_text.append(total_aug_sent)

    return entity_aug_text, o_aug_text, total_aug_text

parser = argparse.ArgumentParser()
parser.add_argument('--seed', required=True, type=int)
parser.add_argument('--bsize', required=True, type=int)
parser.add_argument('--mu_ratio', required=True, type=float)
parser.add_argument('--sigma', required=True, type=float)
parser.add_argument('--o_mask_rate', default=0.0, type=float)
parser.add_argument('--k', required=True, type=int)
parser.add_argument('--sub_idx', required=True, type=int)
parser.add_argument('--ckpt_dir', required=True, type=str)
parser.add_argument('--in_dir', required=True, type=str)
parser.add_argument('--load_bert', required=True, type=str)
parser.add_argument('--panx_out', default=0, type=int)

args = parser.parse_args()

if True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on ", device)

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    BSIZE = args.bsize
    MU_RATIO = args.mu_ratio
    SIGMA = args.sigma
    O_MASK_RATE = args.o_mask_rate
    K = args.k
    SUB_IDX = args.sub_idx
    CKPT_DIR = args.ckpt_dir
    IN_DIR = args.in_dir + 'train.txt'
    OUT_DIR = args.in_dir + 'aug'

    config = {}
    config["load_bert"] = args.load_bert

    entity_model = XLMRobertaForMaskedLM.from_pretrained(config["load_bert"], return_dict=True).to(device)
    o_model = XLMRobertaForMaskedLM.from_pretrained(config["load_bert"], return_dict=True).to(device)
    tokenizer = XLMRobertaTokenizer.from_pretrained(config["load_bert"], do_lower_case=False)
    
    # Add entity labels as special tokens
    tokenizer.add_tokens(['<En>', '<De>', '<Es>', '<Nl>'], special_tokens=True)
    tokenizer.add_tokens(['<B-PER>', '<I-PER>', '<B-ORG>', '<I-ORG>', '<B-LOC>', '<I-LOC>', '<B-MISC>', '<I-MISC>', '<O>'],
                         special_tokens=False) # False so that they are not removed during decoding
    entity_model.resize_token_embeddings(len(tokenizer))
    o_model.resize_token_embeddings(len(tokenizer))

    entity_model.load_state_dict(torch.load(CKPT_DIR))

    dataset = Data(tokenizer, BSIZE, label_map, IN_DIR, MU_RATIO, SIGMA, O_MASK_RATE).dataset
    
    dataloader = DataLoader(dataset, batch_size=BSIZE)

    entity_aug_tensor, o_aug_tensor, total_aug_tensor, input_tensor = aug(entity_model, o_model, dataloader, K, SUB_IDX)
    entity_aug_text, o_aug_text, total_aug_text = decode(entity_aug_tensor, o_aug_tensor, total_aug_tensor, input_tensor, tokenizer)


    num_skip_lines = 0

    for aug_text, ext in zip([entity_aug_text], ['entity']):
        with open(OUT_DIR + '.tmp', 'w') as filehandle:
            for sent in aug_text:
                for word in sent:
                    filehandle.write('%s\n' % word) #.lstrip('▁'))
                filehandle.write('\n')

        with open(OUT_DIR + '.tmp', 'r') as filehandle, open(OUT_DIR+'.'+ext , 'w+') as outfile, open(IN_DIR, 'r') as infile:
            
            tmplines = filehandle.readlines()
            inlines = infile.readlines()
            in_idx = 0
            for tmp_idx in range(len(tmplines)):
                special_masks = ['<B-PER>', '<I-PER>', '<B-ORG>', '<I-ORG>', '<B-LOC>', '<I-LOC>', '<B-MISC>', '<I-MISC>', '<En>', '<De>', '<Es>', '<Nl>']
                if tmplines[tmp_idx].rstrip('\n') in special_masks: # remove special masks
                    continue
                if tmplines[tmp_idx] != '\n':
                    assert inlines[in_idx] != '\n'
                    outline = tmplines[tmp_idx].rstrip('\n').split()[0][1:] + '\t' + inlines[in_idx].rstrip('\n').split()[-1]
                    outfile.write(outline + '\n')
                else:
                    while inlines[in_idx] != '\n':
                        in_idx += 1
                        num_skip_lines += 1
                    outfile.write('\n')
                in_idx += 1

        # Remove tmp file
        os.remove(OUT_DIR + '.tmp')














