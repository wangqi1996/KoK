import logging

import faiss.contrib.torch_utils
import torch

from fairseq.models.KNNModel_bak import KNNDatastore

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


def count_knn_result(tgt_index, mask_for_label_count):
    B, S, K = tgt_index.size()

    expand_tgt_idx = tgt_index.unsqueeze(-2).expand(B, S, K, K)
    expand_tgt_idx = expand_tgt_idx.masked_fill(mask_for_label_count[:K, :K], value=-1)

    labels_sorted, _ = expand_tgt_idx.sort(dim=-1)  # [B, S, K, K]
    labels_sorted[:, :, :, 1:] *= ((labels_sorted[:, :, :, 1:] - labels_sorted[:, :, :, :-1]) != 0).long()
    retrieve_label_counts = labels_sorted.ne(0).sum(-1)  # [B, S, K]
    retrieve_label_counts[:, :, :-1] -= 1

    return retrieve_label_counts


def generate_label_count_mask(max_k):
    # [0, 1, 1]
    # [0, 0, 1]
    # [0, 0, 0]
    mask_for_label_count = torch.empty((max_k, max_k)).fill_(1)
    mask_for_label_count = torch.triu(mask_for_label_count, diagonal=1).bool()

    if torch.cuda.is_available():
        mask_for_label_count = mask_for_label_count.cuda()

        mask_for_label_count.requires_grad = False
    return mask_for_label_count


class LabelDatastore(KNNDatastore):
    def __init__(self, args, task, **kwargs):
        self.label_knn_key = getattr(args, "label_knn_key", '1')
        self.label_knn_value = getattr(args, "label_knn_value", '1')
        super(LabelDatastore, self).__init__(args, task, **kwargs)
        self.mask_for_label_count = generate_label_count_mask(self.k)

    def get_index_dim(self, args):
        if self.label_knn_key == '1':
            return self.k * 2

    def set_dstore_size(self, task=None, **kwargs):
        if self.label_knn_key == '1':
            super(LabelDatastore, self).set_dstore_size(task, **kwargs)

    def extract_feature(self, knn_result):
        if self.label_knn_key == '1':
            # [k1 distance, k2 distance, k1 count, k2 count]
            retrieve_distance = knn_result['distance']
            relative_distance = retrieve_distance / retrieve_distance[:, :, 0].unsqueeze(-1)

            retrieve_tgt_index = knn_result['tgt_index']
            label_count = count_knn_result(retrieve_tgt_index, self.mask_for_label_count)

            key = torch.cat((relative_distance, label_count), dim=-1)

        return key

    def extract_value(self, retrieve_tokens, reference):
        if self.label_knn_value == '1':
            # reference token in knn retrieve result, the value is 1, else 0
            real_tokens = reference.unsqueeze(-1)
            value = (real_tokens == retrieve_tokens).long()
            value = value.sum(-1)
            value.masked_fill_(value > 1, 1)

        return value

    def retrieve_and_score(self, queries, token_knn=None, **kwargs):
        if token_knn['distance'] is not None:
            queries = self.extract_feature(token_knn)
        result = super().retrieve_and_score(queries, **kwargs)
        if not isinstance(result['score'], int):
            result['score'] = result['score'][:, :, 1].unsqueeze(-1)

        return result


class LabelTokenDatastore(object):
    def __init__(self, args, task, **kwargs):
        self.token_datastore = KNNDatastore(args, task)
        self.label_datastore = LabelDatastore(args, task, vocab_size=2)

    def retrieve_and_score(self, queries, **kwargs):
        token_result = self.token_datastore.retrieve_and_score(queries, **kwargs)
        token_score, token_lambda = token_result['score'], token_result['lambda']

        label_result = self.label_datastore.retrieve_and_score(queries, token_knn=token_result, **kwargs)
        label_score = label_result['score']

        return {"score": token_score, "lambda": label_score}

    @staticmethod
    def add_args(parser):
        parser.add_argument('--label-knn-key', type=str, default='1')
        parser.add_argument('--label-knn-value', type=str, default='1')

    def get_normalized_probs(
            self,
            logits,
            log_probs,
            **extra
    ):
        return self.token_datastore.get_normalized_probs(logits, log_probs, **extra)

    def add_datastore(self, key, value, **kwargs):
        # get label value
        knn_result = self.token_datastore.retrieve(key)  # [B, S, K]
        retrieve_tokens = knn_result['tgt_index']
        if retrieve_tokens is not None and retrieve_tokens.size(-1) == self.token_datastore.k:
            mask = self.token_datastore.get_add_mask(value, **kwargs)
            assert (~mask).long().sum() == 0  # b=1, no invalid token  ==>  don't mask the label key and  label value,

            label_value = self.label_datastore.extract_value(retrieve_tokens, value)
            label_key = self.label_datastore.extract_feature(knn_result)
            self.label_datastore._add(mask, label_key, label_value)

        # first add label datastore, then add token datastore
        self.token_datastore.add_datastore(key, value, **kwargs)
