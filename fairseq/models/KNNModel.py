import logging

import faiss.contrib.torch_utils
import torch

from fairseq.models.CombinationMethod import get_combination_class
from fairseq.models.KNNModel_bak import KNNDatastore, DynamicD, whitening_queries

faiss.logger.level = logging.WARNING


def build_knn_datastore(args, task, **kwargs):
    from fairseq.models.KNNModel_bak import FixKNNDatastore, PositiveDatastore, PositiveNegativeDatastore, \
        PosteriorKNNDatastore
    knn_type = args.knn_type
    if knn_type == "normal":
        return KNNDatastore(args, task, )
    elif knn_type == "fix":
        return FixKNNDatastore(args, task, )
    elif knn_type == "positive":
        return PositiveDatastore(args, task)
    elif knn_type == "positive-negative":
        return PositiveNegativeDatastore(args, task)
    elif knn_type == "posterior":
        return PosteriorKNNDatastore(args, task)
    elif knn_type == "label-datastore":
        return LabelTokenDatastore(args, task)
    elif knn_type == 'dynamic-d':
        return DynamicD(args, task)


class LabelDatastore(KNNDatastore):
    def __init__(self, args, task, token_datastore=None, **kwargs):
        # feature list
        self.label_count = getattr(args, "label_count", False)
        self.distance = getattr(args, "distance", False)

        # combination method
        self.combination_method = getattr(args, "combination_method", 'relative-flat')
        self.value_method = getattr(args, "value_method", 'in-all')
        self.combination = get_combination_class(self.combination_method, args.k, self.value_method, token_datastore)

        super(LabelDatastore, self).__init__(args, task, **kwargs)

        self.temperature = args.label_temperature_value
        self.whitening = getattr(args, "whitening", "none")
        if self.whitening != "none":
            self.key = torch.zeros((self.dstore_size, self.dimension), dtype=torch.float).cuda()

    def get_index_dim(self, args):
        feature_num = 0
        if self.label_count:
            feature_num += 1
        if self.distance:
            feature_num += 1

        return self.combination.get_index_dim(self.k, feature_num)

    def get_dstore_size(self, task=None, **kwargs):

        return self.combination.get_datastore_size(task.datasets['train'].tgt.sizes.sum(), self.k)

    def extract_feature(self, knn_result, search=False, keepdim=False):
        """ call: (1). search label datastore    (2). save knn datastore"""
        feature = None

        batch_size, seq_len, _ = knn_result['distance'].shape
        if self.distance:
            distance = self.combination.get_distance_feature(knn_result['distance'], search=search)
            feature = distance

        if self.label_count:
            label_count = self.combination.get_label_count(knn_result['tgt_index'], search=search).float()
            feature = label_count if feature is None else torch.cat((feature, label_count), dim=-1)
        if keepdim:
            knn_dim = feature.size(-1)
            feature = feature.view(batch_size, seq_len, knn_dim)

        return feature  # [-1, knn_dim]

    def extract_value(self, retrieve_tokens, reference, **kwargs):
        return self.combination.extract_value(retrieve_tokens, reference, **kwargs)  # [-1, knn_dim]

    def retrieve_and_score(self, queries, token_knn=None, **kwargs):
        """ token_knn=None  => directly input queries"""
        if token_knn is not None and token_knn['distance'] is not None:
            queries = self.extract_feature(token_knn, search=True, keepdim=True)

            if self.whitening == "datastore":
                q_dim = queries.size(-1)
                queries = whitening_queries(queries.view(-1, q_dim), self.mu, self.s, self.u)

        result = super().retrieve_and_score(queries, **kwargs)
        if not isinstance(result['score'], int):
            result['score'] = result['score'][:, :, 1].unsqueeze(-1)  # 取出使用knn的p。
            # 1.0 --> 0.99, else, log(0)
            score_mask = result['score'] > 0.99
            result['score'].masked_fill_(score_mask, 0.99)
        return result


class LabelTokenDatastore(object):
    def __init__(self, args, task, **kwargs):
        self.token_datastore = KNNDatastore(args, task)
        self.token_datastore.temperature = 10
        self.label_datastore = LabelDatastore(args, task, vocab_size=2, token_datastore=self.token_datastore)

    def retrieve_and_score(self, queries, **kwargs):
        token_result = self.token_datastore.retrieve_and_score(queries, **kwargs)
        token_score, token_lambda = token_result['score'], token_result['lambda']

        label_result = self.label_datastore.retrieve_and_score(queries, token_knn=token_result, **kwargs)
        label_score = label_result['score']

        return {"score": token_score, "lambda": label_score}

    @staticmethod
    def add_args(parser):
        parser.add_argument('--combination-method', type=str, default='relative-flat')
        parser.add_argument('--value-method', type=str, default="equal")

        parser.add_argument('--distance', action="store_true")
        parser.add_argument('--label-count', action="store_true")

        parser.add_argument('--save-label-datastore', action="store_true")
        parser.add_argument('--compute-label-accuracy', action="store_true")

    def get_normalized_probs(
            self,
            logits,
            log_probs,
            **extra
    ):
        return self.token_datastore.get_normalized_probs(logits, log_probs, **extra)

    def add_datastore(self, key, value, **kwargs):
        knn_result = self.token_datastore.retrieve(key)  # [B, S, K]
        retrieve_tokens = knn_result['tgt_index']
        if retrieve_tokens is not None and retrieve_tokens.size(-1) == self.token_datastore.k:
            mask = self.token_datastore.get_add_mask(value, **kwargs)
            assert (~mask).long().sum() == 0  # b=1, no invalid token  ==>  don't mask the label key and  label value,

            retrieve_tokens = retrieve_tokens.squeeze(-1)
            label_value = self.label_datastore.extract_value(retrieve_tokens=retrieve_tokens,
                                                             reference=value, p_nmt=kwargs.get('p_nmt', None),
                                                             knn_result=knn_result)

            label_key = self.label_datastore.extract_feature(knn_result)

            self.label_datastore.add_mask_value(label_key.contiguous(), label_value.contiguous())

        # first add label datastore, then add token datastore
        self.token_datastore.add_datastore(key, value, **kwargs)

    # def compute_accuracy(self, label_key, label_value):
    #     result = self.label_datastore.retrieve_and_score(label_key, token_knn=None)
    #     label_score = result['score']
    #     # self.save_temp(label_score, )
    #
    # def save_temp(self, label_key, label_value, filename=None):
    #     content = []
    #     for d, v in zip(label_key.cpu().tolist(), label_value.cpu().tolist()):
    #         content.append("%.2f\t%s\n" % (d[0], v))
    #
    #     with open(self.filename, 'a') as f:
    #         f.writelines(content)

    # self.save_label_datastore = getattr(args, "save_label_datastore", False)
    # if self.save_label_datastore:
    #     f = os.path.basename(self.label_datastore.args.path)
    #     self.filename = os.path.join('/home/data_ti5_c/wangdq/code/knn-mt/output/', f)
    #     with open(self.filename, 'w'):
    #         pass
    #
    # self.compute_label_accuracy = getattr(args, "compute_label_accuracy", False)
    # if self.compute_label_accuracy:
    #     f = os.path.basename(self.label_datastore.args.path)
    #     self.label_filename = os.path.join('/home/data_ti5_c/wangdq/code/knn-mt/label_accuracy/', f)
    #     with open(self.filename, 'w'):
    #         pass
