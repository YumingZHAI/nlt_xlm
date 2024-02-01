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

from .hmtClassifier import *
# .. means in the parent directory
from ..optim import get_optimizer
from ..utils import concat_batches, truncate, to_cuda
from ..data.dataset import ParallelDataset
from ..data.loader import load_binarized, set_dico_parameters

logger = logging.getLogger()

class HMT:
    def __init__(self, embedder, scores, params):
        """
        Initialize HMT trainer / evaluator.
        Initial `embedder` should be on CPU to save memory.
        """
        self._embedder = embedder
        # scores = {}   (initialized in finetune-hmt.py)
        self.scores = scores
        self.params = params

    def get_iterator(self, splt):
        """
        Get a data iterator.
        """
        assert splt in ['train', 'valid', 'test']
        # shuffle=False,
        return self.data[splt]['x'].get_iterator(
            shuffle = (splt == 'train'),
            group_by_size=self.params.group_by_size,
            return_indices=True
        )

    def save_checkpoint(self, state, is_best, ckt_folder):
        """
           Saves model and training parameters at checkpoint/ + 'last.pth.tar'.
           If is_best==True, also saves checkpoint/ + 'best.pth.tar'
           Args:
               state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer's state_dict
               is_best: (bool) True if it is the best model seen till now
               ckt_folder: (string) folder where parameters are to be saved
        """
        # todo when use all_data, after the last given epoch, save a X.final.last.pth.tar, then well save it
        # filepath = os.path.join(ckt_folder, self.params.trainCorpus + '.final.last.pth.tar')
        # otherwise, just save '.last.pth.tar' after each epoch
        filepath = os.path.join(ckt_folder, self.params.trainCorpus + '.last.pth.tar')
        if not os.path.exists(ckt_folder):
            print("Checkpoint directory doesn't exist! Making directory %s" % ckt_folder)
            os.mkdir(ckt_folder)
        torch.save(state, filepath)
        if is_best:
            logger.info("Saving best model of epoch %i" % state['epoch'])
            shutil.copyfile(filepath, os.path.join(ckt_folder, self.params.trainCorpus + 'f' + str(self.params.fold) + '.best.pth.tar'))

    def load_checkpoint(self, checkpoint_path, classifierModel, optimizer_e=None, optimizer_p=None):
        """Loads model and optimizer parameters (state_dict) from checkpoint_path.
        Args:
            checkpoint_path: (string) FILENAME which needs to be loaded
            model: (torch.nn.Module) model for which the parameters are loaded
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
        # attention: accuracy information is not saved for the final model
        if not "final" in checkpoint_path:
            logger.info("The best accuracy obtained was %.1f%%" % loaded['accuracy'])
        return loaded

    def set_seed(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        ## If you are working with a multi-GPU model, this function is insufficient to get determinism.
        ## To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        random.seed(0)

    def train_epoch(self, params, classifierModel):
        # track best validation accuracy outside of loop
        best_val_acc = 0.0
        # train and validate the model
        for epoch in range(params.n_epochs):
            self.set_seed()
            # update epoch, begin from 1 instead of 0
            self.epoch = epoch + 1
            logger.info("HMT - Training epoch %i/%i ..." % (self.epoch, params.n_epochs))
            self.train(classifierModel)

            logger.info("HMT - Validating epoch %i/%i ..." % (self.epoch, params.n_epochs))
            # deactivate the autograd engine, don't backprop
            ### todo during cross-validation, use this:
            with torch.no_grad():
                scores = self.validate(classifierModel)
                self.scores.update(scores)
            val_acc = self.scores['hmt_valid_acc']
            is_best = val_acc >= best_val_acc
            self.save_checkpoint({'epoch': self.epoch,
                                  'accuracy': val_acc,
                                  'model_state_dict': classifierModel.state_dict(),
                                  'optim_e_dict': self.optimizer_e.state_dict(),
                                  'optim_p_dict': self.optimizer_p.state_dict()},
                                 is_best=is_best,
                                 ckt_folder=params.checkpoint_dir)
            if is_best:
                logger.info("The new best validation accuracy: %.1f%%" % val_acc)
                best_val_acc = val_acc
            ### todo to save the final model using all data, don't validate:
            # if epoch == params.n_epochs - 1 :
            #     self.save_checkpoint({'epoch': self.epoch,
            #                       'model_state_dict': classifierModel.state_dict(),
            #                       'optim_e_dict': self.optimizer_e.state_dict(),
            #                       'optim_p_dict': self.optimizer_p.state_dict()},
            #                       is_best=False,
            #                       ckt_folder=params.checkpoint_dir)

    def run(self):
        """
        Run HMT training / validation / evaluation
        """
        params = self.params
        self.set_seed()

        # load data
        self.data = self.load_data()

        if not self.data['dico'] == self._embedder.dico:
            raise Exception(("Dictionary in evaluation data (%i words) seems different than the one " +
                             "in the pretrained model (%i words). Please verify you used the same dictionary, " +
                             "and the same values for max_vocab and min_count.") % (len(self.data['dico']), len(self._embedder.dico)))

        self.embedder = copy.deepcopy(self._embedder)
        # initialize the classifier model
        classifierModel = hmtClassifier(self.embedder, params)
        classifierModel.cuda()

        # optimizers
        self.optimizer_e = get_optimizer(list(classifierModel.embedder.get_parameters(params.finetune_layers)),
                                         params.optimizer_e)
        self.optimizer_p = get_optimizer(classifierModel.proj.parameters(), params.optimizer_p)

        # useful during cross validation
        # best_savedModel = params.trainCorpus + 'f' + str(params.fold) + '.best.pth.tar'
        # todo for using the final model:
        best_savedModel = "books.final.last.pth.tar"  # checkpoint_dir should be sent-finalModel/ !!
        if params.inference:
            restore_path = os.path.join(params.checkpoint_dir, best_savedModel)
            logger.info("Restoring parameters from %s" % restore_path)
            self.load_checkpoint(restore_path, classifierModel)
            logger.info("Inference mode...")
            with torch.no_grad():
                # evaluate the saved best model on an unseen test set
                self.test(classifierModel)
        elif params.resume_training:
            restore_path = os.path.join(params.checkpoint_dir, best_savedModel)
            logger.info("Restoring parameters from %s" % restore_path)
            self.load_checkpoint(restore_path, classifierModel, self.optimizer_e, self.optimizer_p)
            logger.info("Resume training mode...")
            # e.g. train first on Europarl, then train on tedannote
            self.train_epoch(params, classifierModel)
        else:
            logger.info("Training from scratch mode...")
            self.train_epoch(params, classifierModel)

    def get_batch_x_data(self, batch, params, en_id, fr_id):
        (sent1, len1), (sent2, len2), idx = batch
        # sent1: by column: each sentence, by line: each word piece id?
        sent1, len1 = truncate(sent1, len1, params.max_len, params.eos_index)
        sent2, len2 = truncate(sent2, len2, params.max_len, params.eos_index)
        x, lengths, positions, langs = concat_batches(
            sent1, len1, en_id,
            sent2, len2, fr_id,
            params.pad_index,
            params.eos_index,
            reset_positions=False
        )
        # print(x)  # combine two sentences
        return x, lengths, positions, langs, idx, len1

    def train(self, classifierModel):
        """
        Finetune for one epoch on the HMT training set.
        """
        params = self.params
        # the train() method comes from nn.Module
        # first train the XLM model, then the output projection layer
        classifierModel.train()

        # training variables
        losses = []
        ns = 0  # number of sentences
        nw = 0  # number of words
        t = time.time()

        # iterator of the training dataset
        iterator = self.get_iterator('train')
        en_id = params.lang2id['en']
        fr_id = params.lang2id['fr']

        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                break

            x, lengths, positions, langs, idx, len1 = self.get_batch_x_data(batch, params, en_id, fr_id)
            # size: batch_size
            y = self.data['train']['y'][idx]
            x, y, lengths, positions, langs = to_cuda(x, y, lengths, positions, langs)
            bs = len(len1)  # batchsize

            # size: batch_size, 2
            output = classifierModel(x, lengths, positions, langs)
            # loss (for all sentences of each batch)
            loss = F.cross_entropy(output, y)

            # backward / optimization
            # must first call .zero_grad()
            self.optimizer_e.zero_grad()
            self.optimizer_p.zero_grad()
            loss.backward()
            self.optimizer_e.step()
            self.optimizer_p.step()

            # update statistics
            ns += bs
            nw += lengths.sum().item()
            # print(type(lengths))  <class 'torch.Tensor'>
            losses.append(loss.item())

            # log
            # ns: multiple of batch size. so every 100*bs sentences, print the average loss
            if ns % (100 * bs) < bs:
                logger.info("HMT - Epoch %i - Train nb sentences %7i - %.1f words/s - Loss: %.4f" %
                            (self.epoch, ns, nw / (time.time() - t), sum(losses) / len(losses)))   # average loss
                nw, t = 0, time.time()
                losses = []

            # number of sentences to train on in one epoch
            if params.epoch_size != -1 and ns >= params.epoch_size:
                break

    def validate(self, classifierModel):
        """
        Validate on HMT validation sets
        """
        params = self.params
        classifierModel.eval()

        scores = OrderedDict({'epoch': self.epoch})

        en_id = params.lang2id['en']
        fr_id = params.lang2id['fr']
        valid = 0
        total = 0

        for batch in self.get_iterator('valid'):
            x, lengths, positions, langs, idx, len1 = self.get_batch_x_data(batch, params, en_id, fr_id)
            y = self.data['valid']['y'][idx]
            x, y, lengths, positions, langs = to_cuda(x, y, lengths, positions, langs)
            output = classifierModel(x, lengths, positions, langs)
            # get the indice of the maximum prediction value
            predictions = output.data.max(1)[1]
            # update statistics
            # count the correct predictions:
            valid += predictions.eq(y).sum().item()
            total += len(len1)

        # compute accuracy
        acc = 100.0 * valid / total
        scores['hmt_valid_acc'] = acc
        logger.info("HMT - valid - Epoch %i - Acc: %.1f%%" % (self.epoch, acc))
        logger.info("__log__:%s" % json.dumps(scores))
        return scores

    def test(self, classifierModel):
        """
        Test on a TED Talks annotated test set
        todo: batch size need to be 1
        """
        params = self.params
        classifierModel.eval()
        # print(classifierModel)

        en_id = params.lang2id['en']
        fr_id = params.lang2id['fr']

        proba_list = []
        output_file = open(params.clf_result, "w")

        for batch in self.get_iterator('test'):
            x, lengths, positions, langs, idx, len1 = self.get_batch_x_data(batch, params, en_id, fr_id)
            # test set: not label about human vs machine, instead we want to check
            # whether there's correlation between
            # the classifier's human translation prediction probability
            # and non-literal translation's percentage (on token level)
            x, lengths, positions, langs = to_cuda(x, lengths, positions, langs)
            output = classifierModel(x, lengths, positions, langs)
            # output.data: tensor([[ 4.9024, -0.4637]], device='cuda:0')
            # softmax, dimension=1: values of all columns sum to 1
            # ----get the value for the prediction of human translation, first row, second column
            proba_list.append(F.softmax(output.data, dim=1)[0][1].item())
            # ----0 or 1, predictions: tensor([0], device='cuda:0')
            # predictions = output.data.max(1)[1]
            # print(predictions.item(), file=output_file)
            # ----without softmax, direct output value of the predicting the human translation
            # proba_list.append(output.data[0][1].item())
        for proba in proba_list:
            print(proba, file=output_file)
        output_file.close()
        logger.info("The classifier has finished its prediction!")

    def load_data(self):
        """
        Load HMT bilingual classification data.
        """
        params = self.params
        # (dataset size) half machine, half human
        data = {splt: {} for splt in ['train', 'valid', 'test']}

        if params.trainCorpus != "tedannote":
            # human vs machine translation (sent-level)
            label2id = {'machine': 0, 'human': 1}
        else:
            # the sentence contains or not non-literal translation
            label2id = {'absent': 0, 'present': 1}
        # params.data_path: ./hmt/data
        dpath = params.data_path
        fold_index = str(params.fold)
        trainCorpus = params.trainCorpus
        devCorpus = params.devCorpus
        for splt in ['train', 'valid', 'test']:
            # load data and dictionary
            if splt == 'train':
                # todo to save the final model, after combining train and dev data (comment Valid and test part)
                # data_en = load_binarized(os.path.join(dpath, 'all_data/' + trainCorpus + '/source_all.bpe.pth'), params)
                # data_fr = load_binarized(os.path.join(dpath, 'all_data/' + trainCorpus + '/target_all.bpe.pth'), params)
                data_en = load_binarized(os.path.join(dpath, ('cv_data/' + trainCorpus + '/source_%s_' + fold_index + '.bpe.pth') % splt), params)
                data_fr = load_binarized(os.path.join(dpath, ('cv_data/' + trainCorpus + '/target_%s_' + fold_index + '.bpe.pth') % splt), params)
            elif splt == 'valid':
                data_en = load_binarized(os.path.join(dpath, ('cv_data/' + devCorpus + '/source_%s_' + fold_index + '.bpe.pth') % splt), params)
                data_fr = load_binarized(os.path.join(dpath, ('cv_data/' + devCorpus + '/target_%s_' + fold_index + '.bpe.pth') % splt), params)
            # elif splt == 'test':
            #     data_en = load_binarized(os.path.join(dpath, 'test/all.en.bpe.pth'), params)
            #     data_fr = load_binarized(os.path.join(dpath, 'test/all.fr.bpe.pth'), params)
            data['dico'] = data.get('dico', data_en['dico'])

            # set (or update) dictionary parameters
            set_dico_parameters(params, data, data_en['dico'])
            set_dico_parameters(params, data, data_fr['dico'])

            # check evernote to understand data_en['sentences'], data_en['positions']
            # create dataset, e.g. data[train]['x'] = ParallelDataset(____)
            data[splt]['x'] = ParallelDataset(
                data_en['sentences'], data_en['positions'],
                data_fr['sentences'], data_fr['positions'],
                params
            )

            # load labels
            if splt == 'train':
                # todo to save the final model, after combining train and dev data (comment Valid part)
                # label = 'all_data/' + trainCorpus + '/label_all.txt'
                label = 'cv_data/' + trainCorpus + '/label_%s_' + fold_index + '.txt'
            elif splt == 'valid':
                label = 'cv_data/' + devCorpus + '/label_%s_' + fold_index + '.txt'
            if splt != 'test':
                # todo to save the final model, after combining train and dev data
                # with open(os.path.join(dpath, label), 'r') as f:
                with open(os.path.join(dpath, label % splt), 'r') as f:
                    labels = [label2id[l.rstrip()] for l in f]
                data[splt]['y'] = torch.LongTensor(labels)
                assert len(data[splt]['x']) == len(data[splt]['y'])
        return data
