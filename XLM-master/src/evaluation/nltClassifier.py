import torch.nn as nn
import torch
import ast

class nltClassifier(nn.Module):
    def __init__(self, embedder, params):
        super(nltClassifier, self).__init__()
        # modified here, use the pre-trained model of XLM as a layer of the ClassifierModel
        self.embedder = embedder
        self.xlmModel = embedder.model
        # class torch.nn.Sequential(*args)   take variable arguments
        # predict into 2 output classes
        self.proj = nn.Sequential(*[
            nn.Dropout(params.dropout),
            nn.Linear(4*self.xlmModel.dim, 2)   # after phrase-level feature engineering, the dimension is 4*out_dim, see below
        ])
        self.embedder.cuda()
        self.xlmModel.cuda()
        self.proj.cuda()

    def forward(self, x, lengths, positions, langs, phrase_data_batch, len1):
        # cf src/model/embedder.py
        # this tensor's shape: (max_sequence_length, batch_size, model_dimension)
        tensor = self.xlmModel('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
        # print("parallel sentence tensor size", tensor.size())

        # id: from 0 to batch_size-1
        batch_prediction = []
        max_phrase_pairs = 0
        # for each group of phrase pairs in each sentence pairs
        for id, pairs in enumerate(phrase_data_batch):
            tabs = pairs.split("\t")
            tmp_max = len(tabs)
            if tmp_max > max_phrase_pairs:
                max_phrase_pairs = tmp_max
            feature_tensor = []
            for tab in tabs:
                en_index, fr_index, label = tab.split(':')
                en_embed = []
                # e.g. [1, 2, 3] EN side is already adapted, can use directly
                for x in ast.literal_eval(en_index):
                    # subword representation = tensor[subword index][sentence index in this batch]
                    en_embed.append(tensor[int(x)][id])
                # max-pooling of each column input tensors
                en_repre = torch.max(torch.stack(en_embed), 0).values
                # print(en_repre.size()) 1024  representation for the English phrase

                fr_embed = []
                for x in ast.literal_eval(fr_index):
                    # len1[id]: English sentence length </s>(0) 1 2 3 4 </s>
                    # input </s>(0) 1 2 3 4 </s> </s> 1 2 3 4 </s>
                    # positions (not used here) </s>(0) 1 2 3 4 </s>(5) </s>(restarts from 0) 1 2 3 4 </s>
                    # tensor starts from 0, and no restarts
                    fr_embed.append(tensor[int(x)+len1[id]][id])
                fr_repre = torch.max(torch.stack(fr_embed), 0).values
                # print(fr_repre.size()) 1024
                abs_diff = torch.abs(en_repre - fr_repre)
                element_wise_product = torch.mul(en_repre, fr_repre)
                feature = torch.cat((en_repre, fr_repre, abs_diff, element_wise_product), dim=0)
                # print(abs_diff.size())  1024
                # print(element_wise_product.size()) 1024
                # print(feature.size()) 4096
                #### this output is for a phrase pair
                feature_tensor.append(feature)
            # size: current_nb_phrase_pairs, 4096
            feature_tensor = torch.stack(feature_tensor)
            # size: current_nb_phrase_pairs, 2
            output = self.proj(feature_tensor)
            batch_prediction.append(output)
        # the output tensor has grad_fn=<AddmmBackward>
        # print(batch_prediction)
        return batch_prediction
