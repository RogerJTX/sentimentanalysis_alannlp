import numpy as np
import torch
import torch.optim as optim
from allennlp.data import DataLoader, TextFieldTensors
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.training.trainer import GradientDescentTrainer
# from allennlp_models.classification.dataset_readers.stanford_sentiment_tree_bank import \
#     StanfordSentimentTreeBankDatasetReader
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
from typing import Dict

from predictors import SentenceClassifierPredictor

EMBEDDING_DIM = 128
HIDDEN_DIM = 128


# Model in AllenNLP represents a model that is trained.
@Model.register("lstm_classifier")
class LstmClassifier(Model):
    def __init__(self,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary,
                 positive_label: str = '4') -> None:
        super().__init__(vocab)
        # We need the embeddings to convert word IDs to their vector representations
        self.embedder = embedder

        self.encoder = encoder

        # After converting a sequence of vectors to a single vector, we feed it into
        # a fully-connected linear layer to reduce the dimension to the total number of labels.
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))

        # Monitor the metrics - we use accuracy, as well as prec, rec, f1 for 4 (very positive)
        positive_index = vocab.get_token_index(positive_label, namespace='labels')
        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(positive_index)

        # We use the cross entropy loss because this is a classification task.
        # Note that PyTorch's CrossEntropyLoss combines softmax and log likelihood loss,
        # which makes it unnecessary to add a separate softmax layer.
        self.loss_function = torch.nn.CrossEntropyLoss()

    # Instances are fed to forward after batching.
    # Fields are passed through arguments with the same name.
    def forward(self,
                tokens: TextFieldTensors,
                label: torch.Tensor = None) -> torch.Tensor:
        # In deep NLP, when sequences of tensors in different lengths are batched together,
        # shorter sequences get padded with zeros to make them equal length.
        # Masking is the process to ignore extra zeros added by padding
        mask = get_text_field_mask(tokens)

        # Forward pass
        embeddings = self.embedder(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)

        # In AllenNLP, the output of forward() is a dictionary.
        # Your output dictionary must contain a "loss" key for your model to be trained.
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            self.f1_measure(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = self.f1_measure.get_metric(reset)
        return {'accuracy': self.accuracy.get_metric(reset),
                'precision': precision,
                'recall': recall,
                'f1_measure': f1_measure}


def main():
    reader = StanfordSentimentTreeBankDatasetReader()

    s3_prefix = 'https://s3.amazonaws.com/realworldnlpbook/data'
    # train_dataset = reader.read(f'{s3_prefix}/stanfordSentimentTreebank/trees/train.txt')
    # dev_dataset = reader.read(f'{s3_prefix}/stanfordSentimentTreebank/trees/dev.txt')
    train_dataset = reader.read('Treebank_train.txt')
    print(type(train_dataset))
    print(train_dataset)

    dev_dataset = reader.read('Treebank_dev.txt')



    # You can optionally specify the minimum count of tokens/labels.
    # `min_count={'tokens':3}` here means that any tokens that appear less than three times
    # will be ignored and not included in the vocabulary.

    # 您可以选择指定令牌 / 标签的最小计数。
    # 'min_count = {tokens：3}'
    # 这里的意思是任何出现少于三次的标记都将被忽略，并且不会包含在词汇表中。
    vocab = Vocabulary.from_instances(train_dataset + dev_dataset,
                                      min_count={'tokens': 3})

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)

    # BasicTextFieldEmbedder takes a dict - we need an embedding just for tokens,
    # not for labels, which are used as-is as the "answer" of the sentence classification

    # BasicTextFieldEmbedder需要一个dict-我们需要一个仅用于令牌的嵌入，
    # 不适用于标签，它被用作句子分类的“答案”
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    # Seq2VecEncoder is a neural network abstraction that takes a sequence of something
    # (usually a sequence of embedded word vectors), processes it, and returns a single
    # vector. Oftentimes this is an RNN-based architecture (e.g., LSTM or GRU), but
    # AllenNLP also supports CNNs and other simple architectures (for example,
    # just averaging over the input vectors).

    # Seq2VecEncoder是一个神经网络抽象，它需要一系列的东西
    # （通常是一系列嵌入的词向量），处理它，并返回一个
    # 矢量。通常这是基于RNN的体系结构（例如，LSTM或GRU），但是
    # AllenNLP还支持cnn和其他简单的体系结构（例如，
    # 对输入向量求平均值）。
    encoder = PytorchSeq2VecWrapper(
        torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

    model = LstmClassifier(word_embeddings, encoder, vocab)

    train_dataset.index_with(vocab)
    dev_dataset.index_with(vocab)

    train_data_loader = DataLoader(train_dataset,
                                   batch_sampler=BucketBatchSampler(
                                       train_dataset,
                                       batch_size=32,
                                       sorting_keys=["tokens"]))
    dev_data_loader = DataLoader(dev_dataset,
                                 batch_sampler=BucketBatchSampler(
                                     dev_dataset,
                                     batch_size=32,
                                     sorting_keys=["tokens"]))

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    trainer = GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=train_data_loader,
        validation_data_loader=dev_data_loader,
        patience=10,
        num_epochs=20)

    trainer.train()

    predictor = SentenceClassifierPredictor(model, dataset_reader=reader)
    # logits = predictor.predict('This is the best movie ever!')['logits']
    logits = predictor.predict('''On August 28, Mustafa varank, Turkey's minister of industry and technology, said Turkey plans to become a production center for automotive batteries by investing in cells, battery modules and battery packs. The country also hopes to become Europe's largest and the world's top five electric and autopilot auto makers by 2030. In order to achieve this goal, varank said Turkey would support the investment of electronic and electrical companies in the automotive industry. Varank points out that modern Turkish plants will cover half of the world's I20 capacity, 90% of which is expected to be exported abroad. "It took 27 months to build this line, with a total investment of $194 million. The productivity of I20 in Turkey will exceed 60%, which will increase gradually. In the past year, Turkey has developed EMUs, SUVs, tractors and excavators equipped with electric engines, and now plans to develop electric vehicle technology. Varank said Turkey would build an ecosystem to produce key components for electric vehicles, such as electric engines, inverters, charging equipment and compressors. He stressed that the automobile industry is the "locomotive" of Turkey's industrial sector, which also provides advantages for other industries. In May and June this year, Turkey's industrial production increased by double-digit compared with the same period last year. In the first half of 2020, Turkey issued 1200 investment award certificates worth US $108 billion (about US $16.7 billion) and created 163000 new jobs. On August 28, Turkey released its economic confidence index for August, and varank said: "the positive trend continues, and our citizens have more positive expectations for the post epidemic period." Choi Hong GHI, South Korea's ambassador to Ankara, said that Hyundai Motor, one of the world's top five auto manufacturers, established its first overseas factory in Turkey 23 years ago. "Hyundai's zmit factory is a symbol of economic cooperation between the two countries, which directly promotes employment and exports in Turkey." Eckkyun Oh, chief executive of Hyundai assan, said the company has produced more than two million cars in Turkey, most of which are exported to countries in Europe, the Middle East and North Africa. "We will produce 100000 new I20 cars here," he said.''')['logits']
    label_id = np.argmax(logits)

    print(model.vocab.get_token_from_index(label_id, 'labels'))


if __name__ == '__main__':
    main()


