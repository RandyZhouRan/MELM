import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
import random


class InputExample(object):

    def __init__(self, guid, text, label):

        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):

    def __init__(self, input_ids, input_mask, label_ids, masked_ids, entity_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_ids = label_ids
        self.masked_ids = masked_ids
        self.entity_mask = entity_mask

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
                    label.append('O') # Use O as pseudo label for entity special token
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

    def get_examples(self, dsplit):

        data_df = pd.read_csv(os.path.join(self.data_dir, dsplit + ".txt"),
                              sep=" |\t", header=None, skip_blank_lines=False,
                              engine='python', error_bad_lines=False, quoting=3,
                              keep_default_na = False,
                              na_values=[''])
        return self._create_examples(data_df)

class Data():
    def __init__(self, tokenizer, b_size, label_map, file_dir, mask_rate):

        self.tokenizer = tokenizer
        self.b_size = b_size
        self.label_map = label_map
        self.mask_rate = mask_rate

        self.datasets = self.create_dataset_files(file_dir)

    def create_dataset_files(self, file_dir):
        processor = MultiLabelTextProcessor(file_dir)
        datasets = []
        for dsplit in ['train', 'dev']:
            print(f"Generating dataloader for {dsplit}")
            examples = processor.get_examples(dsplit)
            features = self.convert_examples_to_features(examples, self.tokenizer, is_train=(dsplit == 'train'))
            input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
            masked_ids = torch.tensor([f.masked_ids for f in features], dtype=torch.long)
            entity_mask = torch.tensor([f.entity_mask for f in features], dtype=torch.long)

            dataset = TensorDataset(input_ids, input_mask, label_ids, masked_ids, entity_mask)
            datasets.append(dataset)

        return datasets

    def convert_examples_to_features(self, examples, tokenizer, max_seq_length=128, is_train=True):
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
            masked_ids = [input_ids.copy() for i in range(30)]  # Use different masking for diff epoch, [list]*20 is bug coz it refer to the same list

            entity_mask = [[0] for i in range(30)]
            label_ids = [0]

            for i, label in enumerate(example.label):
                label_ids.append(self.label_map[label])
                label_ids.extend([0] * (subword_len[i]-1))

                # Mask named entities in sentence, and generate entity mask
                if label != "O":
                    same_class_labels = ['B-'+label[2:], 'I-'+label[2:]]
                    diff_class_labels = [l for l in ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC'] if l not in same_class_labels]
                    assert len(diff_class_labels) == 7

                    for count in range(subword_len[i]):
                        if subword_start[i]+count >= max_seq_length:
                            break
                        for k in range(30):
                            random.shuffle(diff_class_labels) #Diff subs for each epoch

                            if random.random() < self.mask_rate:
                                #masked_ids[k][subword_start[i]+count] = self.label_to_token_id(label)
                                masked_ids[k][subword_start[i]+count] = self.tokenizer.convert_tokens_to_ids('<mask>')
                                entity_mask[k].append(1)
                            else:
                                entity_mask[k].append(0)

                else:
                    for count in range(subword_len[i]):
                        if subword_start[i]+count >= max_seq_length:
                            break
                        for k in range(30):
                            if is_train and random.random() < 0:
                                masked_ids[k][subword_start[i]+count] = self.tokenizer.convert_tokens_to_ids('<mask>') 
                            entity_mask[k].append(0)

            # Pad short sentence and truncate long sentence
            if len(label_ids) > max_seq_length:
                #print("Truncating...")
                label_ids = label_ids[:max_seq_length]
                for k in range(30):
                    masked_ids[k] = masked_ids[k][:max_seq_length]
                    entity_mask[k] = entity_mask[k][:max_seq_length]
            else:
                #print("Padding...")
                label_ids.extend([0] * (max_seq_length - len(label_ids)))
                for k in range(30):
                    masked_ids[k].extend([0] * (max_seq_length - len(masked_ids[k])))
                    entity_mask[k].extend([0] * (max_seq_length - len(entity_mask[k])))

            features.append(
                    InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        label_ids=label_ids,
                        masked_ids=masked_ids,
                        entity_mask=entity_mask,
                        ))

        return features

    def label_to_token_id(self,label):
        label = '<' + label + '>'
        assert label in ['<O>', '<B-PER>', '<I-PER>', '<B-ORG>', '<I-ORG>', '<B-LOC>', '<I-LOC>', '<B-MISC>', '<I-MISC>']

        return self.tokenizer.convert_tokens_to_ids(label)

