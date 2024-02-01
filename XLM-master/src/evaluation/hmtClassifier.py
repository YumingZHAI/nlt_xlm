import torch.nn as nn

class hmtClassifier(nn.Module):
    def __init__(self, embedder, params):
        super(hmtClassifier, self).__init__()
        # in class SentenceEmbedder: self.out_dim = model.dim  (transformer encoder's dimension)
        self.embedder = embedder
        self.xlmModel = embedder.model
        # class torch.nn.Sequential(*args)   take variable arguments
        # predict into 2 output classes
        self.proj = nn.Sequential(*[
            nn.Dropout(params.dropout),
            nn.Linear(self.xlmModel.dim, 2)
        ])
        self.embedder.cuda()
        self.xlmModel.cuda()
        self.proj.cuda()

    def forward(self, x, lengths, positions, langs):
        # cf src/model/embedder.py
        # embedding tensor's shape: (batch_size, model_dimension), because tensor[0] is returned
        embeddings = self.embedder.get_embeddings(x, lengths, positions, langs)
        # size: batch_size, 2
        output = self.proj(embeddings)
        return output

