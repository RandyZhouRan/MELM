import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
import random

def trunc_gauss(mu, sigma, bottom, top):
    a = round(random.gauss(mu,sigma))
    while (bottom <= a <= top) == False:
        a = round(random.gauss(mu,sigma))
    return a

class InputExample(object):

    def __init__(self, guid, text, label):

        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):

    def __init__(self, input_ids, input_mask, label_ids, entity_masked_ids, entity_mask, o_masked_ids, o_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_ids = label_ids
        self.entity_masked_ids = entity_masked_ids
        self.entity_mask = entity_mask
        self.o_masked_ids = o_masked_ids
        self.o_mask = o_mask

class MultiLabelTextProcessor():

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = None

    def _create_examples(self, df):
        """Creates examples for the training and dev sets."""
        examples = []
        text = []
        label = []
        guid = 0

        for i,row in enumerate(df.values):
            if not pd.isna(row[-1]):
                # Add leading label special token
                if row[-1] != 'O':
                    if row[1] in ['En', 'De', 'Es', 'Nl']: # Add language prefix if they are present
                        text.append('<' + row[1] + '>')
                        label.append('O')
                    text.append('<' + row[-1] + '>')
                    label.append('O')
                text.append(row[0])
                label.append(row[-1])
                # Add trailing label special token
                if row[-1] != 'O':
                    text.append('<' + row[-1] + '>')
                    label.append('O')
            elif text != []:
                examples.append(
                    InputExample(guid=guid, text=text, label=label))
                guid += 1
                text = []
                label = []
        return examples

    def get_examples(self, filename):

        data_df = pd.read_csv(filename,
                              sep=" |\t", header=None, skip_blank_lines=False,
                              engine='python', error_bad_lines=False, quoting=3,
                              keep_default_na = False,
                              na_values=['']) 

        return self._create_examples(data_df)

class Data():
    def __init__(self, tokenizer, b_size, label_map, filename, mu_ratio, sigma, o_mask_rate):

        self.tokenizer = tokenizer
        self.b_size = b_size
        self.label_map = label_map
        self.filename = filename
        self.mu_ratio = mu_ratio
        self.sigma = sigma
        self.o_mask_rate = o_mask_rate

        self.dataset = self.create_dataset_files()


    def create_dataset_files(self):
        processor = MultiLabelTextProcessor('data')
        for filename in [self.filename]:
            print(f"Generating dataloader for {filename}")
            examples = processor.get_examples(filename)
            features = self.convert_examples_to_features(examples, self.tokenizer)
            input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
            entity_masked_ids = torch.tensor([f.entity_masked_ids for f in features], dtype=torch.long)
            entity_mask = torch.tensor([f.entity_mask for f in features], dtype=torch.long)
            o_masked_ids = torch.tensor([f.o_masked_ids for f in features], dtype=torch.long)
            o_mask = torch.tensor([f.o_mask for f in features], dtype=torch.long)

            dataset = TensorDataset(input_ids, input_mask, label_ids, entity_masked_ids, entity_mask, o_masked_ids, o_mask)

        return dataset

    def convert_examples_to_features(self, examples, tokenizer, max_seq_length=128):
        """Loads a data file into a list of `InputBatch`s."""

        features = []

        for example in examples:
            encoded = tokenizer(example.text,
                                padding = 'max_length',
                                truncation = True,
                                max_length=max_seq_length,
                                is_split_into_words = True)
            input_ids = encoded["input_ids"]
            input_mask = encoded["attention_mask"]

            # Insert X label for non-leading sub-word tokens
            subword_len = []
            for word in example.text:
                subword_len.append(len(tokenizer.tokenize(word)))

            subword_start = [0]
            subword_start.extend(np.cumsum(subword_len))
            subword_start = [x+1 for x in subword_start]
            entity_masked_ids = input_ids.copy()
            o_masked_ids = input_ids.copy()
            entity_mask = [0]
            o_mask = [0]
            label_ids = [0]

            for i, label in enumerate(example.label):
                label_ids.append(self.label_map[label])
                label_ids.extend([0] * (subword_len[i]-1))

                # Mask named entities in sentence, and generate entity mask
                if label != "O":
                    o_mask.extend([0] * subword_len[i])

                    mask_len = trunc_gauss(subword_len[i] * self.mu_ratio, self.sigma, 1, subword_len[i])
                    mask_pos = random.sample(list(range(subword_len[i])), mask_len)

                    for count in range(subword_len[i]):
                        if subword_start[i]+count >= max_seq_length:
                            break
                        if count in mask_pos:
                            entity_masked_ids[subword_start[i]+count] = self.tokenizer.convert_tokens_to_ids('<mask>')
                            entity_mask.append(1)
                        else:
                            entity_mask.append(0)
                else:
                    entity_mask.extend([0] * subword_len[i])
                    for count in range(subword_len[i]):
                        if subword_start[i]+count >= max_seq_length:
                            break
                        if random.random() < self.o_mask_rate:
                            o_masked_ids[subword_start[i]+count] = self.tokenizer.convert_tokens_to_ids('<mask>')
                            o_mask.append(1)
                        else:
                            o_mask.append(0)
     
            # Pad short sentence and truncate long sentence
            if len(label_ids) > max_seq_length:
                label_ids = label_ids[:max_seq_length]
                entity_masked_ids = entity_masked_ids[:max_seq_length]
                entity_mask = entity_mask[:max_seq_length]
                o_masked_ids = o_masked_ids[:max_seq_length]
                o_mask = o_mask[:max_seq_length]
            else:
                label_ids.extend([0] * (max_seq_length - len(label_ids)))
                entity_masked_ids.extend([0] * (max_seq_length - len(entity_masked_ids)))
                entity_mask.extend([0] * (max_seq_length - len(entity_mask)))
                o_masked_ids.extend([0] * (max_seq_length - len(o_masked_ids)))
                o_mask.extend([0] * (max_seq_length - len(o_mask)))

            features.append(
                    InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        label_ids=label_ids,
                        entity_masked_ids=entity_masked_ids,
                        entity_mask=entity_mask,
                        o_masked_ids = o_masked_ids,
                        o_mask=o_mask
                        ))

        return features

    def label_to_token_id(self,label):
        label = '<' + label + '>'
        assert label in ['<B-PER>', '<I-PER>', '<B-ORG>', '<I-ORG>', '<B-LOC>', '<I-LOC>', '<B-MISC>', '<I-MISC>']

        return self.tokenizer.convert_tokens_to_ids(label)


