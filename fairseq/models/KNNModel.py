import logging

import faiss
import faiss.contrib.torch_utils
import torch
from torch_scatter import scatter

from fairseq.models.CombinationMethod import DisCountAbs

faiss.logger.level = logging.WARNING

def build_knn_datastore(args, task, **kwargs):
    knn_type = args.knn_type
    if knn_type == "normal":
        return KNNDatastore(args, task, )
    elif knn_type == "label-datastore":
        return LabelTokenDatastore(args, task)


def calculate_knn_prob(tgt_index: torch.Tensor,  # [B, S, K]
                       distance: torch.Tensor,  # [B, S, K]
                       temperature: torch.Tensor,  # [B, S, 1]
                       k=8,
                       vocab_size=1,
                       return_every_k=False,
                       ):
    bsz, seq_len, _ = distance.size()

    re_compute_dists = -1 * distance  # [B, S, K]
    scaled_dists = re_compute_dists / temperature
    knn_weight = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)  # [B, S, K, 1]

    knn_tgt_prob = torch.zeros(bsz, seq_len, k, vocab_size).to(distance.device)  # [B, S, K, Vocab Size]
    tgt_index = tgt_index.unsqueeze(-1)  # [B, S, K, 1]

    scatter(src=knn_weight.float(), out=knn_tgt_prob, index=tgt_index, dim=-1)

    prob = knn_tgt_prob.sum(-2)  # [Batch size, seq len, k, vocab size]

    return prob

class KNNDatastore(object):
    def __init__(self, args, task, **kwargs):
        self.args = args
        self.k = args.k
        self.dimension = self.get_index_dim(args)
        self.index = self.setup_index(task)
        self.temperature = args.temperature_value
        self.lambda_value = args.lambda_value

        self.vocab_size = kwargs.get("vocab_size", len(task.target_dictionary))
        self.padding_idx = task.target_dictionary.pad()
        self.distance_threshold = getattr(args, "distance_threshold", -1)

    def get_index_dim(self, args):
        return args.decoder_embed_dim

    def calculate_knn_prob(self, tgt_index: torch.Tensor,
                           distance: torch.Tensor,
                           temperature: torch.Tensor,
                           return_every_k=False):
        return calculate_knn_prob(tgt_index, distance, temperature, k=self.k, vocab_size=self.vocab_size,
                                  return_every_k=return_every_k)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--knn-type', type=str)

        parser.add_argument('--lambda-value', type=float, default=0.3)
        parser.add_argument('--temperature-value', type=float, default=10)
        parser.add_argument('--k', type=int, default=8)

        parser.add_argument("--distance-threshold", type=float, default=-1)
        LabelTokenDatastore.add_args(parser)

    def get_dstore_size(self, task=None, **kwargs):
        return task.datasets['train'].tgt.sizes.sum() + 2

    def setup_index(self, task):

        self.dstore_size = self.get_dstore_size(task)

        index = faiss.IndexFlatL2(self.dimension)
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        index = faiss.index_cpu_to_gpu(res, 0, index, co)

        self.vals = torch.zeros((self.dstore_size, 1), dtype=torch.int64).cuda()
        self.val_index = 0

        return index

    def get_lambda(self, distance=None, **kwargs):
        if self.distance_threshold != -1:
            distance_mask = distance[:, :, 0] > self.distance_threshold
            lambda_value = distance.new_zeros(size=(distance.size(0), distance.size(1))).fill_(self.lambda_value)
            lambda_value.masked_fill_(distance_mask, 0)  # [B, Seq_len, 1]
            return lambda_value.unsqueeze(-1)

        return self.lambda_value

    def retrieve(self, queries):
        if self.index.ntotal < self.k:
            return {'distance': None, 'knn_index': None, 'tgt_index': None}

        bsz, seq_len, q_dim = queries.size()
        queries = queries.view(-1, q_dim)
        dists, knns = self.index.search(queries.contiguous(), self.k)

        tgt_idx = self.vals[knns].to(queries.device).squeeze(-1)
        tgt_idx = tgt_idx.view(bsz, seq_len, -1)
        dists = dists.view(bsz, seq_len, -1)
        knns = knns.view(bsz, seq_len, -1)

        return {'distance': dists, 'knn_index': knns, 'tgt_index': tgt_idx}

    def get_score(self, knn_result, return_every_k=False):
        knn_distance = knn_result['distance']
        if knn_distance is None:
            return {"score": 0, "lambda": 0, "distance": None, "index": None}

        tgt_index = knn_result['tgt_index']

        knn_lambda = self.get_lambda(distance=knn_distance)
        knn_score = self.calculate_knn_prob(tgt_index, knn_distance, self.temperature, return_every_k=return_every_k)
        return {
            "score": knn_score,
            "lambda": knn_lambda,
            "distance": knn_distance,
            "tgt_index": tgt_index
        }

    def retrieve_and_score(self, queries, **kwargs):
        knn_result = self.retrieve(queries)
        score = self.get_score(knn_result)
        return score

    def _add_datastore(self, key, value):
        assert key.size(0) == value.size(0)
        if len(value) == 0:
            return
        _len, = value.shape
        self.vals[self.val_index: self.val_index + _len] = value.unsqueeze(-1)

        self.val_index += _len
        self.index.add(key)

    def add_datastore(self, key, value, **kwargs):
        mask = value != self.padding_idx
        value = value[mask].int()
        key = key[mask].float()
        self._add_datastore(key, value)

    def get_normalized_probs(
            self,
            knn_result
    ):
        knn_lambda, knn_score = knn_result["lambda"], knn_result['score']
        return knn_score, knn_lambda


class LabelDatastore(KNNDatastore):
    def __init__(self, args, task, token_datastore=None, **kwargs):

        self.combination = DisCountAbs(args.k)
        super(LabelDatastore, self).__init__(args, task, **kwargs)
        self.temperature = args.label_temperature_value
        self.distance_threshold = getattr(args, "distance_threshold", -1)

    def get_lambda_value(self, token_result, **kwargs):
        label_result = self.retrieve_and_score(None, token_knn=token_result, **kwargs)
        label_score = label_result['score']

        return label_score

    def get_index_dim(self, args):
        feature_num = 2
        return self.combination.get_index_dim(self.k, feature_num)

    def get_dstore_size(self, task=None, **kwargs):
        return self.combination.get_datastore_size(task.datasets['train'].tgt.sizes.sum())

    def extract_feature(self, knn_result, keepdim=False):
        """ call: (1). search label datastore    (2). save knn datastore"""
        batch_size, seq_len, _ = knn_result['distance'].shape

        distance = self.combination.get_distance_feature(knn_result['distance'])

        label_count = self.combination.get_label_count(knn_result['tgt_index']).float()
        feature = torch.cat((distance, label_count), dim=-1)

        if keepdim:
            feature = feature.view(batch_size, seq_len, -1)

        return feature  # [-1, dim] or [batch, seq_len, dim]

    def extract_value(self, reference, **kwargs):
        return self.combination.extract_value(reference, **kwargs)  # [-1, knn_dim]

    def retrieve_and_score(self, queries, token_knn=None, **kwargs):
        if token_knn is not None and token_knn['distance'] is not None:
            queries = self.extract_feature(token_knn, keepdim=True)
        result = super().retrieve_and_score(queries, **kwargs)
        if not isinstance(result['score'], int):

            result['score'] = result['score'][:, :, 1].unsqueeze(-1)
            # 1.0 --> 0.99, else, log(0)
            score_mask = result['score'] > 0.99
            result['score'].masked_fill_(score_mask, 0.9999)
            if self.distance_threshold != -1:
                d0 = token_knn['distance'][:, :, 0]
                mask = d0 > self.distance_threshold
                result['score'].masked_fill_(mask.unsqueeze(-1), 0)

        return result


class LabelTokenDatastore(object):
    def __init__(self, args, task, **kwargs):
        self.token_datastore = KNNDatastore(args, task)
        self.label_datastore = LabelDatastore(args, task, vocab_size=2, token_datastore=self.token_datastore)
        self.padding_idx = task.target_dictionary.pad()

    def retrieve_and_score(self, queries, **kwargs):
        token_result = self.token_datastore.retrieve_and_score(queries, **kwargs)
        token_score = token_result['score']

        label_score = self.label_datastore.get_lambda_value(token_result, **kwargs)

        return {"score": token_score, "lambda": label_score}

    @staticmethod
    def add_args(parser):
        parser.add_argument('--combination-method', type=str, default='relative-flat')
        parser.add_argument('--value-method', type=str, default="equal")

        parser.add_argument('--distance', action="store_true")
        parser.add_argument('--label-count', action="store_true")

        parser.add_argument('--label-temperature-value', type=float, default=10)

    def get_normalized_probs(
            self,
            knn_result
    ):
        return self.token_datastore.get_normalized_probs(knn_result)

    def add_datastore(self, key, value, **kwargs):
        knn_result = self.token_datastore.retrieve(key)  # [B, S, K]
        retrieve_tokens = knn_result['tgt_index']
        p_nmt = kwargs.get('p_nmt', None)

        if retrieve_tokens is not None and retrieve_tokens.size(-1) == self.token_datastore.k:
            p_knn = self.token_datastore.get_score(knn_result)['score']
            label_value = self.label_datastore.extract_value(reference=value, p_nmt=p_nmt, p_knn=p_knn)

            label_key = self.label_datastore.extract_feature(knn_result)

            self.label_datastore._add_datastore(label_key.contiguous(), label_value.contiguous())

        self.token_datastore._add_datastore(key.view(-1, key.size(-1)), value.view(-1))
