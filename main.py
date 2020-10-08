import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_processor import *
import pandas as pd
import numpy as np
import io
import random
import os
import time
import pickle
import torch
import argparse
from pprint import pprint
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUX_LOSS_TYPES = ['cos', 'cos_margin']
AUX_SAMPLING = ['uniform']
TASKS = ['ntcir_ja', 'ntcir_en', 'ntcir_zh', 'sst2', 'sst5', 'mr', 'cr', 'subj', 'trec']

def cos_sim(va, vb):
  return sum(va * vb) / ((sum(va ** 2) ** 0.5) * (sum(vb ** 2) ** 0.5))


class Bert(torch.nn.Module):
  def __init__(self, lr=5e-5, aux_loss_type='sim', lang='en', cls_mode='single', num_labels=2):
    super().__init__()
    self.aux_loss_type = aux_loss_type
    print(cls_mode)
    if lang == 'en':
      self.bert = BertModel.from_pretrained('bert-base-uncased')
    elif lang == 'ja':
      self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
    elif lang == 'zh':
      self.bert = BertModel.from_pretrained('bert-base-chinese')
      
    self.linear = torch.nn.Linear(768, num_labels)
    self.bert.cuda()
    self.linear.cuda()
    self.cls_mode = cls_mode
    self.num_labels = num_labels

      
  def calc_loss(self, main_x, aux_x, main_mask, aux_mask, main_y, aux_y, all_neg, use_aux=False):
    _, encoded_main = self.bert(input_ids=main_x, attention_mask=main_mask)
    logits = self.linear(encoded_main)
    if self.cls_mode == 'multi':
      prob_pos = torch.sigmoid(logits)
      prob_neg = torch.sigmoid(-logits)
      log_prob_pos = torch.log(prob_pos + 1e-9)
      log_prob_neg = torch.log(prob_neg + 1e-9)
      loss_pos = torch.sum(-main_y * log_prob_pos, 1)
      loss_neg = torch.sum(-(1 - main_y) * log_prob_neg, 1)
      main_loss = loss_pos + loss_neg
    else:
      prob = F.softmax(logits, dim=-1)
      log_prob = torch.log(prob + 1e-9)
      main_loss = torch.sum(-log_prob * main_y, 1)
    if not use_aux:
      return torch.mean(main_loss), main_loss

    _, encoded_aux = self.bert(input_ids=aux_x, attention_mask=aux_mask)
    aux_batch_size = encoded_aux.shape[0] // encoded_main.shape[0] 
    encoded_main = torch.reshape(encoded_main, (encoded_main.shape[0], 1, encoded_main.shape[1]))
    encoded_aux = torch.reshape(encoded_aux, (encoded_aux.shape[0] // aux_batch_size, aux_batch_size, encoded_aux.shape[1]))
    
    elif 'cos' in self.aux_loss_type:
      norm_main = torch.sum(encoded_main ** 2, dim=2) ** 0.5
      norm_aux = torch.sum(encoded_aux ** 2, dim=2) ** 0.5
      cos_sim = torch.sum(encoded_main * encoded_aux, 2) / (norm_main * norm_aux) + 1

      if all_neg:
        y_onehot = torch.Tensor(()).new_zeros((aux_y.shape[0], self.aux_batch_size)).cuda()
      else:
        y_onehot = F.one_hot(aux_y, num_classes=encoded_aux.shape[1])
      aux_loss = torch.sum(-y_onehot * cos_sim, dim=1) + torch.mean((1 - y_onehot) * cos_sim, dim=1)

      if 'margin' in self.aux_loss_type:
        aux_loss += 0.4
    else:
      raise ValueError('Invalid auxiliary loss type: {}'.format(self.aux_loss_type))

    return torch.mean(aux_loss) + torch.mean(main_loss), main_loss, torch.mean(aux_loss)

  def train_model(self,
                  train_x,
                  train_y,
                  train_mask,
                  dev_x,
                  dev_y,
                  dev_mask,
                  test_x,
                  test_y,
                  test_mask,
                  lr=1e-5,
                  batch_size=16,
                  aux_batch_size=4,
                  use_aux=False,
                  sampling='uniform',
                  all_neg=False,
                  model_path='models'):
    model_name = "model_{}".format("ns" if use_aux else "base")
    if use_aux:
      model_name += "_{}".format("allneg" if all_neg else "normal")
    all_params = [p for _, p in self.bert.named_parameters()] + [self.linear.weight, self.linear.bias]
    num_train_steps = len(train_x) // batch_size * 50
    optimizer = AdamW(all_params, lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer,
					      num_warmup_steps=int(num_train_steps * 0.1),
					      num_training_steps=num_train_steps)
    train_y_numpy = [l.numpy() for l in train_y]
    tor = 0
    max_score = -1
    label_group = make_label_group(train_y_numpy)
    stack = []
    count = 1
    steps = 0
    train_losses_main = []
    train_losses_aux = []
    dev_losses_main = []
    dev_losses_aux = []
    all_idx = set(list(range(train_x.shape[0])))

    while tor < 10:
      if count > 50:
          break
      st = time.time()
      print('epoch ', count)
      count += 1
      self.train()
      for bx, by, bmask, prg, bidx in self.data_generator(train_x, train_y, train_mask):
        bx = bx.to(device)
        by = by.to(device)
        bmask = bmask.to(device)

        if use_aux:
          aux_x, aux_y, aux_label, aux_idx = self.aux_task_sampling(
									train_x,
									train_y,
									by,
									bidx,
									label_group,
									batch=aux_batch_size,
									at_random=('rand' in sampling),
									all_neg=all_neg)
          aux_mask = torch.cuda.LongTensor([train_mask[i].numpy() for i in aux_idx])
          aux_x = torch.stack(aux_x).type(torch.LongTensor).cuda()
          aux_y = torch.cuda.LongTensor(aux_y)
          all_loss, main_loss, aux_loss = self.calc_loss(bx, aux_x, bmask, aux_mask, by, aux_y, all_neg, use_aux=True)
        else:
          all_loss, main_loss = self.calc_loss(bx, None, bmask, None, by, None, False, use_aux=False)

        optimizer.zero_grad()
        all_loss.backward()
        loss_value = all_loss.detach().cpu().numpy()
        optimizer.step()
        scheduler.step()
        print('progress: {:.2f}%, loss = {:.5f}\r'.format(prg * 100, loss_value), end='', flush=True)
        steps += 1
      print('')
      self.eval()
      with torch.no_grad():
        score, _, losses = self.evaluate(dev_x, dev_y, dev_mask, all_neg=all_neg)
        score_test, _, _ = self.evaluate(test_x, test_y, test_mask, all_neg=all_neg)
      print('dev exact match = ', score, flush=True)
      print('test exact match = ', score_test, flush=True)

      if max_score < score:
        max_score = score
        tor = 0
        with torch.no_grad():
          max_score_test, preds_test, _ = self.evaluate(test_x, test_y, test_mask)
        torch.save(self.state_dict(), os.path.join(model_path, model_name) + "_{}".format(len(os.listdir(model_path))))
      else:
        tor += 1
      ed = time.time()
      print('time = ', ed - st)
      self.train()

    print('finish')
    all_losses = {
        'main_losses_train': train_losses_main,
        'aux_losses_train': train_losses_aux,
        'main_losses_dev': dev_losses_main,
        'aux_losses_dev': dev_losses_aux}
    return max_score_test, max_score, all_losses


  def predict(self, x, mask):
    _, encoded = self.bert(input_ids=x, attention_mask=mask)
    logits = self.linear(encoded)
    if self.cls_mode == 'single':
      preds = F.softmax(logits, dim=-1)
    else:
      preds = torch.sigmoid(logits)
    return preds

  def encode(self, inputs, masks):
    _, encoded = self.bert(input_ids=inputs, attention_mask=masks)
    return encoded
  
  def evaluate(self, data_x, data_y, data_mask, use_aux=False, all_neg=False):
    all_preds = []
    all_label = []
    dev_y_numpy = [l.numpy() for l in data_y]
    label_group_dev = make_label_group(dev_y_numpy)
    losses = {'main_losses': [], 'aux_losses': []}

    for bx, by, bmask, _, _ in self.data_generator(data_x, data_y, data_mask, shuffle=False, batch=64):
      bx = bx.cuda()
      by = by.cuda()
      bmask = bmask.cuda()
      preds = self.predict(bx, bmask)
      if self.cls_mode == 'single':
        preds_bin = [np.eye(self.num_labels)[np.argmax(l.cpu())] for l in preds]
      else:
        preds_bin = [[int(p >= 0.5) for p in pred] for pred in preds]
      all_preds.extend(preds_bin)
      all_label.extend([l.cpu().numpy() for l in by])

    
    return accuracy_score(np.array(all_label).astype(np.int32), np.array(all_preds)), all_preds, losses

  def aux_task_sampling(self, xs, ys, by, bidx, label_group, at_random=True, batch=4, all_neg=False):
    all_aux_inputs = []
    all_aux_y = []
    all_aux_label = []
    all_aux_index = []

    for tar_y, id_ in zip(by, bidx):
      tar_y = tar_y.cpu().numpy()
      label_eq = tuple(tar_y)
      label_noteq = list(label_group.keys())
      label_noteq.remove(label_eq)
      idx_list = list(range(batch))
      random.shuffle(idx_list)
      random.shuffle(label_noteq)
      if len(label_noteq) >= batch:
        aux_label = label_noteq[:batch]
      else:
        aux_label = [random.choice(label_noteq) for _ in range(batch)]
      index_eq = idx_list[0]
      if not all_neg:
        aux_y = [1 if i == index_eq else 0 for i in range(batch)]
        aux_label[index_eq] = label_eq
      #aux_label = list(label_group.keys())

      aux_index = []
      chosen = []
      for l in aux_label:
        ix = random.choice(label_group[tuple(l)])
        chosen.append(l)
        aux_index.append(ix)

      aux_inputs = [xs[i] for i in aux_index]
      all_aux_inputs.extend(aux_inputs)
      if not all_neg:
        all_aux_y.append(next(i for i, l in enumerate(aux_y) if l == 1))
      else:
        all_aux_y.append(0)
      all_aux_label.extend(chosen)
      all_aux_index.extend(aux_index)
   
    return all_aux_inputs, all_aux_y, all_aux_label, all_aux_index


  @staticmethod
  def data_generator(train_x, train_y, train_mask, batch=16, shuffle=True):
    idx = list(range(len(train_x)))
    num_examples = len(idx)
    if shuffle:
      random.shuffle(idx)
    num_step = train_x.shape[0] // batch + 1
    for i in range(num_step):
      bx = [train_x[i] for i in idx[i*batch:(i + 1)*batch]]
      by = [train_y[i] for i in idx[i*batch:(i + 1)*batch]]
      bmask = [train_mask[i] for i in idx[i*batch:(i + 1)*batch]]
      bidx = idx[i*batch:(i + 1)*batch]
      if len(bx) == 0:
        break
      bx = torch.nn.utils.rnn.pad_sequence(bx).transpose(0, 1).type(torch.LongTensor)
      bmask = (bx != 0).type(torch.LongTensor)
      
      prg = ((i + 1)*batch/num_examples)

      yield bx, torch.stack(by, 0), bmask, prg, bidx


def make_label_group(y):
    label_group = {}
    for i, label in enumerate(y):
        try:
          label_group[tuple(label)].append(i)
        except KeyError:
          label_group[tuple(label)] = [i]

    return label_group
  
def check_args(args):
  pprint(args)
  assert args.task in TASKS, 'unknown task name: {}'.format(args.task)
  assert args.aux_loss_type in AUX_LOSS_TYPES, 'unknown aux loss type: {}'.format(args.aux_loss_type)
  assert args.aux_sampling in AUX_SAMPLING, 'unknown sampling method: {}'.format(args.aux_sampling)


def main(args):
  check_args(args)
  
  datasets = {
    'ntcir_en': ntcirProcessor('../data/ntcir_en/', lang='en'),
    'ntcir_ja': ntcirProcessor('../data/ntcir_ja/', lang='ja'),
    'ntcir_zh': ntcirProcessor('../data/ntcir_zh', lang='zh'),
    'sst2': sst2Processor('../data/downstream/SST/binary', lang='en'),
    'sst5': sst5Processor('../data/downstream/SST/fine', lang='en'),
    'mr': mrProcessor('../data/downstream/MR/', lang='en'),
    'cr': crProcessor('../data/downstream/CR/', lang='en'),
    'subj': subjProcessor('../data/downstream/SUBJ/', lang='en'),
    'trec': trecProcessor('../data/downstream/TREC/', lang='en')
  }
  train_x, dev_x, test_x, train_mask, dev_mask, test_mask, train_y, dev_y, test_y = datasets[args.task].read_data()
  if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)

  dev_scores = []
  test_scores = []
  for i in range(0, 10):
    bert_model = Bert(aux_loss_type=args.aux_loss_type,
                      lang=datasets[args.task].lang,
		      cls_mode=('multi' if datasets[args.task].multi_label else 'single'),
		      num_labels=train_y.shape[-1])
    
    score_test, score, all_losses = bert_model.train_model(train_x,
       			       train_y,
			       train_mask,
			       dev_x,
			       dev_y,
			       dev_mask,
			       test_x,
			       test_y,
			       test_mask,
			       lr=args.lr,
			       batch_size=args.batch_size,
			       aux_batch_size=args.aux_batch_size,
			       use_aux=args.use_aux,
			       sampling=args.aux_sampling,
			       all_neg=args.all_neg,
			       model_path=args.model_dir)
    
    dev_scores.append(score)
    test_scores.append(score_test)
    del bert_model
  print('finish experiments')
  dev_scores = sorted(dev_scores)[1:-1]
  test_scores = sorted(test_scores)[1:-1]
  print('mean_dev_score = ', sum(dev_scores) / len(dev_scores))
  print('mean_test_score = ', sum(test_scores) / len(test_scores))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, help="Name of task (dataset) defined in data_processor.py")
  parser.add_argument('--lr', type=float, default=1e-5, help="learning rate")
  parser.add_argument('--model_dir', type=str, default="models", help="Path to directory that models are saved in")
  parser.add_argument('--aux_loss_type', type=str, default="cos", help="Type of loss function of auxiliary task.")
  parser.add_argument('--aux_sampling', type=str, default="uniform", help="The way of sampling instences for the auxiliary task")
  parser.add_argument('--use_aux', action='store_true', help="Whether use the auxiliary task")
  parser.add_argument('--all_neg', action='store_true', help="Whether inputs of the auxiliary is all negative examples")
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--aux_batch_size', type=int, default=4)
  args = parser.parse_args()
  start = time.time()
  main(args)
  end = time.time()
  print('spend time = ', end - start)


