# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import copy
import time
import json
import sys
import shutil
import numpy as np
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F

from .nltClassifier import *
# .. means in the parent directory
from ..optim import get_optimizer
from ..utils import concat_batches, truncate, to_cuda
from ..data.dataset import ParallelDataset
from ..data.loader import load_binarized, set_dico_parameters

# phraseTT stands for: phrase translation technique (this script is an adaptation from hmt.py)
# conduct feature engineering to fine-tune XLM, on phrase-level translation technique classification

logger = logging.getLogger()

class phraseTT:
    def __init__(self, embedder, scores, params):
        """
        Initialize trainer / evaluator.
        Initial `embedder` should be on CPU to save memory.
        """
        self._embedder = embedder
        # scores is a dictionary
        self.scores = scores
        self.params = params

    def get_iterator(self, splt):
        """
        Get a data iterator.
        """
        assert splt in ['train', 'valid', 'test']
        # shuffle = False,
        return self.data[splt]['x'].get_iterator(
            shuffle = (splt == 'train'),
            group_by_size=self.params.group_by_size,
            return_indices=True
        )

    def save_checkpoint(self, state, is_best, ckt_folder):
        """
        For each epoch, saves model and training parameters at checkpoint/ + 'last.pth.tar'
        If is_best==True, also saves checkpoint/ + 'best.pth.tar'
        Args:
           state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer's state_dict, etc.
           is_best: (bool) True if it is the best model seen till now
           ckt_folder: (string) the folder where checkpoints are to be saved
        """
        # todo to save the final model
        filepath = os.path.join(ckt_folder, 'balanced.final.last.pth.tar')
        if not os.path.exists(ckt_folder):
            print("Checkpoint directory doesn't exist! Making directory %s" % ckt_folder)
            os.mkdir(ckt_folder)
        torch.save(state, filepath)
        if is_best:
            logger.info("Saving best model of epoch %i" % state['epoch'])
            shutil.copyfile(filepath, os.path.join(ckt_folder, self.params.bestModel))

    def load_checkpoint(self, checkpoint_path, classifierModel, optimizer_e=None, optimizer_p=None):
        """Loads model and optimizer parameters (state_dict) from checkpoint_path.
        Args:
            checkpoint_path: (string) FILENAME which needs to be loaded
            classifierModel: (torch.nn.Module) model for which the parameters are loaded
            optimizer: (torch.optim): resume optimizer from checkpoint
        """
        if not os.path.exists(checkpoint_path):
            logger.error("This checkpoint file doesn't exist: %s" % checkpoint_path)
        loaded = torch.load(checkpoint_path)
        classifierModel.load_state_dict(loaded['model_state_dict'])
        if optimizer_e and optimizer_p:
            optimizer_e.load_state_dict(loaded['optim_e_dict'])
            optimizer_p.load_state_dict(loaded['optim_p_dict'])
        logger.info("Load the best model which was saved after training epoch %i" % loaded['epoch'])
        if not "final" in checkpoint_path:
            logger.info("The best accuracy obtained was %.1f%%" % loaded['accuracy'])
        return loaded

    # todo better understand here
    def set_seed(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        # If you are working with a multi-GPU model, this function is insufficient to get determinism.
        # To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        random.seed(0)

    def load_phrase_aligned_data(self, file):
        """
        load dataset of phrase-level alignments
        """
        i = 0
        dico_phrase_data = {}
        with open(file, "r") as file:
            for l in file:
                dico_phrase_data[i] = l.strip()
                i += 1
        return dico_phrase_data

    def train_epoch(self, params, classifierModel):
        """
        for each epoch, train, validate and save checkpoints
        """
        # trace best validation accuracy outside of loop
        best_val_acc = 0.0
        # train and validate the model
        for epoch in range(params.n_epochs):
            self.set_seed()
            # update epoch, begin from 1 instead of 0
            self.epoch = epoch + 1
            logger.info("Training epoch %i/%i ..." % (self.epoch, params.n_epochs))
            self.train(classifierModel)

            logger.info("Validating epoch %i/%i ..." % (self.epoch, params.n_epochs))
            ### todo during cross-validation, use this:
            # deactivate the autograd engine, don't backprop
            # with torch.no_grad():
            #     scores = self.validate(classifierModel)
            #     self.scores.update(scores)
            # val_acc = self.scores['valid_acc']
            # is_best = val_acc >= best_val_acc
            # self.save_checkpoint({'epoch': self.epoch,
            #                       'accuracy': val_acc,
            #                       'model_state_dict': classifierModel.state_dict(),
            #                       'optim_e_dict': self.optimizer_e.state_dict(),
            #                       'optim_p_dict': self.optimizer_p.state_dict()},
            #                      is_best=is_best,
            #                      ckt_folder=params.checkpoint_dir)
            # if is_best:
            #     logger.info("The new best validation accuracy: %.1f%%" % val_acc)
            #     # update the best_val_acc
            #     best_val_acc = val_acc
            #     # print('Saved best model\'s state dict: \n', classifierModel.state_dict())
            ### todo to save the final model using all data, don't validate:
            if epoch == params.n_epochs - 1 :
                self.save_checkpoint({'epoch': self.epoch,
                                  'model_state_dict': classifierModel.state_dict(),
                                  'optim_e_dict': self.optimizer_e.state_dict(),
                                  'optim_p_dict': self.optimizer_p.state_dict()},
                                  is_best=False,
                                  ckt_folder=params.checkpoint_dir)

    def run(self):
        """
        Run training / validation / evaluation
        """
        params = self.params
        self.set_seed()
        # load data (OK)
        # sentence pair level
        self.data = self.load_data()
        # phrase alignment level
        phrase_data_path = params.data_path
        trainName = params.trainCorpus
        validName = params.devCorpus
        testName = params.testCorpus
        fold_index = str(params.fold)
        # self.train_phrase_aligned_data = self.load_phrase_aligned_data(phrase_data_path + trainName + ".data")
        # self.train_phrase_aligned_data = self.load_phrase_aligned_data(phrase_data_path + trainName + ".train" + fold_index + ".data")
        # self.valid_phrase_aligned_data = self.load_phrase_aligned_data(phrase_data_path + validName + ".valid" + fold_index + ".data")
        self.test_phrase_aligned_data = self.load_phrase_aligned_data(phrase_data_path + testName + ".data")

        if not self.data['dico'] == self._embedder.dico:
            raise Exception(("Dictionary in evaluation data (%i words) seems different than the one " +
                             "in the pretrained model (%i words). Please verify you used the same dictionary, " +
                             "and the same values for max_vocab and min_count.") % (len(self.data['dico']), len(self._embedder.dico)))

        self.embedder = copy.deepcopy(self._embedder)
        # initialize the classifier model
        classifierModel = nltClassifier(self.embedder, params)
        classifierModel.cuda()

        ### check the model and whether the components are trainable
        # print(classifierModel)
        # for name, parameters in classifierModel.named_parameters():
        #     print(name, parameters.requires_grad)

        # todo understand the code about optimizers
        self.optimizer_e = get_optimizer(list(classifierModel.embedder.get_parameters(params.finetune_layers)),
                                         params.optimizer_e)
        self.optimizer_p = get_optimizer(classifierModel.proj.parameters(), params.optimizer_p)

        if params.inference:
            restore_path = os.path.join(params.checkpoint_dir, params.bestModel)
            logger.info("Restoring parameters from %s" % restore_path)
            self.load_checkpoint(restore_path, classifierModel)
            logger.info("Inference mode...")
            # print("Loaded model for testing: \n", classifierModel.state_dict())
            with torch.no_grad():
                # evaluate the saved best model on an unseen test set
                self.test(classifierModel)
        elif params.resume_training:
            restore_path = os.path.join(params.checkpoint_dir, params.bestModel)
            logger.info("Restoring parameters from %s" % restore_path)
            self.load_checkpoint(restore_path, classifierModel, self.optimizer_e, self.optimizer_p)
            logger.info("Resume training mode...")
            # e.g. train first on Europarl, then train on OpenSubtitles
            self.train_epoch(params, classifierModel)
        else:
            logger.info("Training from scratch mode...")
            self.train_epoch(params, classifierModel)

    def get_batch_x_data(self, batch, params, en_id, fr_id):
        (sent1, len1), (sent2, len2), idx = batch
        sent1, len1 = truncate(sent1, len1, params.max_len, params.eos_index)
        sent2, len2 = truncate(sent2, len2, params.max_len, params.eos_index)
        # 30/03 modify reset_positions=True, before it was False
        x, lengths, positions, langs = concat_batches(
            sent1, len1, en_id,
            sent2, len2, fr_id,
            params.pad_index,
            params.eos_index,
            reset_positions=True
        )
        # x.size(): (max(len1+len2),batch_size)
        return x, lengths, positions, langs, idx, len1

    def train(self, classifierModel):
        """
        Finetune for one epoch on the training set.
        """
        params = self.params
        # the train() method comes from nn.Module
        # first train the XLM pre-trained model, then the output projection layer
        classifierModel.train()
        # training variables
        losses = []
        ns = 0  # number of sentences
        nw = 0  # number of words
        t = time.time()
        # iterator of the training dataset (OK)
        iterator = self.get_iterator('train')
        en_id = params.lang2id['en']
        fr_id = params.lang2id['fr']
        while True:
            try:
                # iterate on the yield generator 'iterator'
                batch = next(iterator)
            except StopIteration:
                break
            # idx: sentence ids in this batch
            # x: the tensor containing two sentences (concatenated) in each column
            # positions tensor is also very useful
            # len1 is a list: sentence lengths of each source sentence (contains already 2 </s>)
            # lengths: list, each: len_ENsent + len_FRsent + 4</s>
            x, lengths, positions, langs, idx, len1 = self.get_batch_x_data(batch, params, en_id, fr_id)
            # todo this y is sentence-level label, see this problem later
            # y = self.data['train']['y'][idx]   # self.data['train']['y'] is a torch LongTensor(list)
            ### x, y, lengths, positions, langs = to_cuda(x, y, lengths, positions, langs)

            x, lengths, positions, langs = to_cuda(x, lengths, positions, langs)
            bs = len(len1)  # batchsize

            """here we add the phrase alignment pairs batch which corresponds to each sentence batch"""
            # idx e.g. when without shuffle: [0 1 2 3]
            # a list containing batch_size (e.g. 4) strings, e.g. one of them could be [3, 4]:[1, 2]:literal [5]:[3, 4]:literal [6]:[5, 6]:literal
            phrase_data_batch = [self.train_phrase_aligned_data[id] for id in idx]   # so even with shuffle, there's no problem

            ### gold labels:
            batch_gold = []
            for group_pairs in phrase_data_batch:   # correspond to one sentence pair
                tabs = group_pairs.split("\t")    # "\t" as delimiter
                tmp_list = []
                for tab in tabs:
                    label = tab.split(':')[2]
                    if label in (["literal", "equivalence", "lexical_shift"]):
                        label_id = 0
                    elif label in (["transposition", "generalization", "particularization", "modulation", "modulation_transposition", "figurative"]):
                        label_id = 1
                    tmp_list.append(label_id)
                batch_gold.append(tmp_list)

            #### model's predictions:
            # shape of list: batch_size, current_nb_phrase_pairs, 2
            batch_prediction = classifierModel(x, lengths, positions, langs, phrase_data_batch, len1)

            batch_loss = []
            for i in range(bs):
                prediction = batch_prediction[i]
                target = torch.tensor(batch_gold[i])
                prediction, target = to_cuda(prediction, target)
                loss_per_sent = F.cross_entropy(prediction, target)
                batch_loss.append(loss_per_sent)
            sum_batch_loss = torch.sum(torch.stack(batch_loss))

            # backward / optimization
            self.optimizer_e.zero_grad()
            self.optimizer_p.zero_grad()
            sum_batch_loss.backward()
            self.optimizer_e.step()
            self.optimizer_p.step()

            # update statistics
            ns += bs
            nw += lengths.sum().item()
            losses.append(sum_batch_loss.item())

            # log
            # ns: multiple of batch size. so every 100*bs sentences, show the average loss
            if ns % (100 * bs) < bs:
                logger.info("Epoch %i - Train nb sentences %7i - %.1f words/s - Loss: %.4f" %
                            (self.epoch, ns, nw / (time.time() - t), sum(losses) / len(losses)))   # average loss
                nw, t = 0, time.time()
                losses = []

            # number of sentences to train on in one epoch
            # otherwise, train on all sentences, until there is no batch to iterate
            if params.epoch_size != -1 and ns >= params.epoch_size:
                break

    def validate(self, classifierModel):
        params = self.params
        classifierModel.eval()
        scores = OrderedDict({'epoch': self.epoch})
        en_id = params.lang2id['en']
        fr_id = params.lang2id['fr']
        valid = 0
        total = 0
        # load sentence-level data
        for batch in self.get_iterator('valid'):
            x, lengths, positions, langs, idx, len1 = self.get_batch_x_data(batch, params, en_id, fr_id)
            # todo this y is sentence-level label, see this problem later
            # y = self.data['valid']['y'][idx]
            ### x, y, lengths, positions, langs = to_cuda(x, y, lengths, positions, langs)
            x, lengths, positions, langs = to_cuda(x, lengths, positions, langs)
            bs = len(len1)

            phrase_data_batch = [self.valid_phrase_aligned_data[id] for id in idx]
            batch_gold = []
            for group_pairs in phrase_data_batch:
                tabs = group_pairs.split("\t")
                tmp_list = []
                for tab in tabs:
                    label = tab.split(':')[2]
                    if label in (["literal", "equivalence", "lexical_shift"]):
                        label_id = 0
                    elif label in (["transposition", "generalization", "particularization", "modulation", "modulation_transposition", "figurative"]):
                        label_id = 1
                    tmp_list.append(label_id)
                batch_gold.append(tmp_list)
            batch_prediction = classifierModel(x, lengths, positions, langs, phrase_data_batch, len1)

            for i in range(bs):
                prediction = batch_prediction[i]
                predictions = prediction.data.max(1)[1]
                target = torch.tensor(batch_gold[i])
                predictions, target = to_cuda(predictions, target)
                valid += predictions.eq(target).sum().item()
                total += len(target)
        acc = 100.0 * valid / total
        scores['valid_acc'] = acc
        logger.info("correctly predicted dev data %i/%i" % (valid, total))
        logger.info("valid - Epoch %i - Acc: %.1f%%" % (self.epoch, acc))
        logger.info("__log__:%s" % json.dumps(scores))
        return scores

    def test(self, classifierModel):
        params = self.params
        classifierModel.eval()
        en_id = params.lang2id['en']
        fr_id = params.lang2id['fr']
        inference_file = open(params.clf_result, "w")
        test = 0
        total = 0
        for batch in self.get_iterator('test'):
            x, lengths, positions, langs, idx, len1 = self.get_batch_x_data(batch, params, en_id, fr_id)
            x, lengths, positions, langs = to_cuda(x, lengths, positions, langs)
            bs = len(len1)

            phrase_data_batch = [self.test_phrase_aligned_data[id] for id in idx]
            batch_gold = []
            for group_pairs in phrase_data_batch:
                tabs = group_pairs.split("\t")
                tmp_list = []
                for tab in tabs:
                    label = tab.split(':')[2]
                    # if label in (["literal", "equivalence", "lexical_shift"]):
                    #     label_id = 0
                    # elif label in (["transposition", "generalization", "particularization", "modulation", "modulation_transposition", "figurative"]):
                    #     label_id = 1
                    # todo for rosetta test
                    if label in (["literal"]):
                        label_id = 0
                    elif label in (["modulation", "localization"]):
                        label_id = 1
                    tmp_list.append(label_id)
                batch_gold.append(tmp_list)
            batch_prediction = classifierModel(x, lengths, positions, langs, phrase_data_batch, len1)

            for i in range(bs):
                prediction = batch_prediction[i]
                predictions = prediction.data.max(1)[1]
                target = torch.tensor(batch_gold[i])
                predictions, target = to_cuda(predictions, target)
                test += predictions.eq(target).sum().item()
                total += len(target)
                print(predictions, file=inference_file)
                print(target, file=inference_file)
                print("correctly predicted " + str(predictions.eq(target).sum().item()) + " / total " + str(len(target)), file=inference_file)

        inference_file.close()
        acc = 100.0 * test / total
        logger.info("correct prediction %i, total %i" % (test, total))
        logger.info("test accuracy: %.1f%%" % acc)
        logger.info("The classifier has finished its prediction!")

    # now I have nearly understood all about data processing
    def load_data(self):
        """
        Load bilingual classification data.
        check evernote for more information about dataset preparation
        """
        params = self.params
        data = {splt: {} for splt in ['train', 'valid', 'test']}
        #### sentence-level label
        # human vs machine translation
        # label2id = {'machine': 0, 'human': 1}
        # non-literal translation is absent or present
        # label2id = {'absent': 0, 'present': 1}
        dpath = params.data_path
        fold_index = str(params.fold)
        trainCorpus = params.trainCorpus
        devCorpus = params.devCorpus
        testCorpus = params.testCorpus
        # for splt in ['train','valid', 'test']:
        for splt in ['test']:
            # load data and dictionary
            # if splt == 'train':
            #     data_en = load_binarized(os.path.join(dpath, trainCorpus + '.en.bpe.pth'), params)
            #     data_fr = load_binarized(os.path.join(dpath, trainCorpus + '.fr.bpe.pth'), params)
                # data_en = load_binarized(os.path.join(dpath, (trainCorpus + '.%s' + fold_index + '.en.bpe.pth') % splt), params)
                # data_fr = load_binarized(os.path.join(dpath, (trainCorpus + '.%s' + fold_index + '.fr.bpe.pth') % splt), params)
            # elif splt == 'valid':
            #     data_en = load_binarized(os.path.join(dpath, (devCorpus + '.%s' + fold_index + '.en.bpe.pth') % splt), params)
            #     data_fr = load_binarized(os.path.join(dpath, (devCorpus + '.%s' + fold_index + '.fr.bpe.pth') % splt), params)
            if splt == 'test':
                data_en = load_binarized(os.path.join(dpath, testCorpus + '.en.bpe.pth'), params)
                data_fr = load_binarized(os.path.join(dpath, testCorpus + '.fr.bpe.pth'), params)
            data['dico'] = data.get('dico', data_en['dico'])

            # set (or update) dictionary parameters (from loader.py)
            set_dico_parameters(params, data, data_en['dico'])
            set_dico_parameters(params, data, data_fr['dico'])

            data[splt]['x'] = ParallelDataset(
                data_en['sentences'], data_en['positions'],
                data_fr['sentences'], data_fr['positions'],
                params
            )

            #### load sentence-level label
            # if splt == 'train':
            #     label = 'cv_data/' + trainCorpus + '/label_%s_' + fold_index + '.txt'
            # elif splt == 'valid':
            #     label = 'cv_data/' + devCorpus + '/label_%s_' + fold_index + '.txt'
            # if splt != 'test':
            #     with open(os.path.join(dpath, label % splt), 'r') as f:
            #         labels = [label2id[l.rstrip()] for l in f]
            #     data[splt]['y'] = torch.LongTensor(labels)
            #     # ParallelDataset has a method: __len__(self) to show the number of sentences in the dataset.
            #     assert len(data[splt]['x']) == len(data[splt]['y'])
        return data
