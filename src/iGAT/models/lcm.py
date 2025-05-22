import torch
import torch.nn as nn
import torch.nn.functional as F


class LCM(nn.Module):
    '''
    生成软标签
    '''

    def __init__(self, n_input=10, n_hidden=64, n_output=10):
        super(LCM, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.emb = nn.Embedding(num_embeddings=n_input, embedding_dim=n_hidden)  # 10 * 64
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_output, n_output)
        self.sm = nn.Softmax(dim=1)

    def forward(self, labels, input_vec):
        label_representation = self.emb(labels)  # 128 *10 * 64

        label_representation = F.tanh(self.fc1(label_representation))  # 128 *10 * 64

        input_vec = torch.unsqueeze(input_vec, -1)  # 128 * 64 * 1

        doc_product = torch.bmm(label_representation, input_vec)  # 标签嵌入与输入嵌入 128 * 10 *1
        doc_product = torch.squeeze(doc_product, -1)  # 128 * 10

        label_confusion_vector = self.sm(self.fc2(doc_product))  # 软标签，用于构建多样性软标签 128 *10
        label_confusion_vector = torch.squeeze(label_confusion_vector, -1)

        return label_confusion_vector
