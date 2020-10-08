import torch
import pickle
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

class DataProcessor:
  def __init__(self, path, lang):
    self.path = path
    self.lang = lang
    self.multi_label = False

  def read_data(self):
    data, labels = self._read_and_split_data(self.path)
    labels = [torch.LongTensor(d) for d in labels]
    data_ids, data_mask = self._tokenize(data, lang=self.lang)
    data = (*data_ids, *data_mask, *labels)
    
    return data

  def _read_and_split_data(self, path):
    raise NotImplementedError()

  def _tokenize(self, data, lang='en'):
    if lang == 'en':
      tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    elif lang == 'ja':
      tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese', do_lower_case=False)
    elif lang == 'zh':
      tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', do_lower_case=False)
    else:
      raise ValueError('unknown language: {}'.format(lang))
    
    data_padded = [torch.LongTensor(tokenizer(text, padding='max_length', max_length=128)['input_ids']) for text in data]
    data_mask = [(d != 0).type(torch.LongTensor) for d in data_padded]
    
    return data_padded, data_mask
  
  
  @staticmethod
  def _add_sp_tokens(data):
    return ['[CLS] ' + sent + ' [SEP]' for sent in data]

class ntcirProcessor(DataProcessor):
  def __init__(self, path, lang):
    super(ntcirProcessor, self).__init__(path, lang)
    self.multi_label = True

  def _read_and_split_data(self, path):
    train_df = pd.read_csv(os.path.join(path, 'train.tsv'), sep='\t', header=None, encoding='utf-8', skiprows=[0])
    dev_df = pd.read_csv(os.path.join(path, 'dev.tsv'), sep='\t', header=None, encoding='utf-8', skiprows=[0])
    test_df = pd.read_csv(os.path.join(path, 'test.tsv'), sep='\t', header=None, encoding='utf-8', skiprows=[0])
    train_text = train_df.iloc[:, 1].values
    dev_text = dev_df.iloc[:, 1].values
    test_text = test_df.iloc[:, 1].values
    
    train_label = [[int(s == 'p') for s in l] for l in train_df.iloc[:, 2:].values]
    dev_label = [[int(s == 'p') for s in l] for l in dev_df.iloc[:, 2:].values]
    test_label = [[int(s == 'p') for s in l] for l in test_df.iloc[:, 2:].values]
    
    return [train_text, dev_text, test_text], [train_label, dev_label, test_label]

class sst5Processor(DataProcessor):
  def _read_and_split_data(self, path):
    with open(os.path.join(path, 'sentiment-train'), encoding='utf-8') as f:
      train_data = f.readlines()
      train_text, train_label = self._split_label_and_text(train_data)
      train_text = self._add_sp_tokens(train_text)

    with open(os.path.join(path, 'sentiment-dev'), encoding='utf-8') as f:
      dev_data = f.readlines()
      dev_text, dev_label = self._split_label_and_text(dev_data)
      dev_text = self._add_sp_tokens(dev_text)

    with open(os.path.join(path, 'sentiment-test'), encoding='utf-8') as f:
      test_data = f.readlines()
      test_text, test_label = self._split_label_and_text(test_data)
      test_text = self._add_sp_tokens(test_text)
    
    train_label = [np.eye(5)[int(s)] for s in train_label]
    dev_label = [np.eye(5)[int(s)] for s in dev_label]
    test_label = [np.eye(5)[int(s)] for s in test_label]
    
    return [train_text, dev_text, test_text], [train_label, dev_label, test_label]


  @staticmethod
  def _split_label_and_text(data):
    text = [line.strip().split(' ', 1)[-1] for line in data]
    labels = [line.strip().split(' ', 1)[0] for line in data]

    return text, labels

class sst2Processor(DataProcessor):
  def _read_and_split_data(self, path):
    with open(os.path.join(path, 'sentiment-train'), encoding='utf-8') as f:
      train_data = f.readlines()
      train_text, train_label = self._split_label_and_text(train_data)

    with open(os.path.join(path, 'sentiment-dev'), encoding='utf-8') as f:
      dev_data = f.readlines()
      dev_text, dev_label = self._split_label_and_text(dev_data)

    with open(os.path.join(path, 'sentiment-test'), encoding='utf-8') as f:
      test_data = f.readlines()
      test_text, test_label = self._split_label_and_text(test_data)
    
    train_label = [np.eye(2)[int(s)] for s in train_label]
    dev_label = [np.eye(2)[int(s)] for s in dev_label]
    test_label = [np.eye(2)[int(s)] for s in test_label]
    
    return [train_text, dev_text, test_text], [train_label, dev_label, test_label]

  @staticmethod
  def _split_label_and_text(data):
    text = [line.strip().split('\t', 1)[0] for line in data]
    labels = [line.strip().split('\t', 1)[-1] for line in data]

    return text, labels


class mrProcessor(DataProcessor):
  def _read_and_split_data(self, path):
    with open(os.path.join(path, 'rt-polarity.pos'), encoding='utf-8') as f:
      positives = [line.strip() for line in f]
    with open(os.path.join(path, 'rt-polarity.neg'), encoding='utf-8') as f:
      negatives = [line.strip() for line in f]
    data = positives + negatives
    labels = [np.eye(2)[0],]*len(positives) + [np.eye(2)[1],]*len(negatives)
    train_text, test_text, train_label, test_label = train_test_split(data,
								      labels,
								      shuffle=True,
								      stratify=labels,
								      test_size=0.2,
								      random_state=77777)
    train_text, dev_text, train_label, dev_label = train_test_split(train_text,
							    train_label,
							    shuffle=True,
							    stratify=train_label,
							    test_size=0.2,
							    random_state=77777)
    dataset = {
              'train_text': train_text,
              'dev_text': dev_text,
              'test_text': test_text,
              'train_label': train_label,
              'dev_label': dev_label,
              'test_label': test_label}

    return [train_text, dev_text, test_text], [train_label, dev_label, test_label]

  
class crProcessor(DataProcessor):
  def _read_and_split_data(self, path):
    with open(os.path.join(path, 'custrev.pos'), encoding='utf-8') as f:
      positives = [line.strip() for line in f]
    with open(os.path.join(path, 'custrev.neg'), encoding='utf-8') as f:
      negatives = [line.strip() for line in f]
    data = positives + negatives
    labels = [np.eye(2)[0],]*len(positives) + [np.eye(2)[1],]*len(negatives)
    train_text, test_text, train_label, test_label = train_test_split(data,
								      labels,
								      shuffle=True,
								      stratify=labels,
								      test_size=0.2,
								      random_state=77777)
    train_text, dev_text, train_label, dev_label = train_test_split(train_text,
							    train_label,
							    shuffle=True,
							    stratify=train_label,
							    test_size=0.2,
							    random_state=77777)
    dataset = {
            	'train_text': train_text,
            	'dev_text': dev_text,
            	'test_text': test_text,
            	'train_label': train_label,
            	'dev_label': dev_label,
            	'test_label': test_label}

    return [train_text, dev_text, test_text], [train_label, dev_label, test_label]

class subjProcessor(DataProcessor):
  def _read_and_split_data(self, path):
    with open(os.path.join(path, 'subj.objective'), encoding='utf-8') as f:
      positives = [line.strip() for line in f]
    with open(os.path.join(path, 'subj.subjective'), encoding='utf-8') as f:
      negatives = [line.strip() for line in f]
    data = self._add_sp_tokens(positives + negatives)
    labels = [np.eye(2)[0],]*len(positives) + [np.eye(2)[1],]*len(negatives)
    train_text, test_text, train_label, test_label = train_test_split(data,
								      labels,
								      shuffle=True,
								      stratify=labels,
								      test_size=0.2,
								      random_state=77777)
    train_text, dev_text, train_label, dev_label = train_test_split(train_text,
							    train_label,
							    shuffle=True,
							    stratify=train_label,
							    test_size=0.2,
							    random_state=77777)
    dataset = {
            	'train_text': train_text,
            	'dev_text': dev_text,
            	'test_text': test_text,
            	'train_label': train_label,
            	'dev_label': dev_label,
            	'test_label': test_label}

    return [train_text, dev_text, test_text], [train_label, dev_label, test_label]


class trecProcessor(DataProcessor):
  def _read_and_split_data(self, path):
    with open(os.path.join(path, 'train_5500.label'), encoding='utf-8') as f:
      data = [line.strip() for line in f]
      text = [line.split(' ', 1)[-1] for line in data]
      label = [line.split(' ', 1)[0].split(':')[0] for line in data]
      label_dict = {l: i for i, l in enumerate(sorted(list(set(label))))}
    with open(os.path.join(path, 'TREC_10.label'), encoding='utf-8') as f:
      data = [line.strip() for line in f]
      test_text = [line.split(' ', 1)[-1] for line in data]
      test_label = [line.split(' ', 1)[0].split(':')[0] for line in data]

    train_text, dev_text, train_label, dev_label = train_test_split(text,
								    label,
								    shuffle=True,
								    stratify=label,
								    test_size=0.2,
								    random_state=77777)
    num_labels = len(label_dict)
    train_label = [np.eye(num_labels)[label_dict[l]] for l in train_label]
    dev_label = [np.eye(num_labels)[label_dict[l]] for l in dev_label]
    test_label = [np.eye(num_labels)[label_dict[l]] for l in test_label]

    return [train_text, dev_text, test_text], [train_label, dev_label, test_label]

