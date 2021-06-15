import logging
import os

import faiss
import faiss.contrib.torch_utils
import faiss.contrib.torch_utils
import numpy as np
import torch
from torch_scatter import scatter

from fairseq import utils

faiss.logger.level = logging.WARNING


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

        self.linear_lambda = getattr(args, "linear_lambda", False)
        self.min_lambda = getattr(args, "min_lambda", 0.3)
        self.max_lambda = getattr(args, "max_lambda", 0.3)
        self.distance_threshold = getattr(args, "distance_threshold", -1)

    def get_index_dim(self, args):
        return args.decoder_embed_dim

    @staticmethod
    def add_args(parser):
        parser.add_argument("--linear-lambda", action="store_true")
        parser.add_argument("--min-lambda", type=float, default=0.0)
        parser.add_argument("--max-lambda", type=float, default=0.0)
        parser.add_argument('--lambda-value', type=float, default=0.3)
        parser.add_argument('--temperature-value', type=float, default=10)
        parser.add_argument('--k', type=int, default=8)

        parser.add_argument("--distance-threshold", type=float, default=-1)

        from fairseq.models.KNNModel_bak import FixKNNDatastore
        FixKNNDatastore.add_args(parser)

    def set_dstore_size(self, task=None, **kwargs):
        self.dstore_size = task.datasets['train'].tgt.sizes.sum() + 2

    def setup_index(self, task):

        self.set_dstore_size(task)

        index = faiss.IndexFlatL2(self.dimension)
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)

        self.vals = torch.zeros((self.dstore_size, 1), dtype=torch.int64).cuda()
        self.val_index = 0

        return index

    def get_lambda(self, distance=None, **kwargs):

        if self.linear_lambda:
            lambda_value = self.index.ntotal / self.dstore_size * (
                    self.max_lambda - self.min_lambda) + self.min_lambda
            return lambda_value

        if self.distance_threshold != -1:
            distance_mask = distance[:, :, 0] > self.distance_threshold
            lambda_value = distance.new_zeros(size=(distance.size(0), distance.size(1))).fill_(self.lambda_value)
            lambda_value.masked_fill_(distance_mask, 0)  # [B, Seq_len, 1]
            return lambda_value.unsqueeze(-1)

        return self.lambda_value

    def retrieve(self, queries):
        if self.index.ntotal <= 0:
            return {'distance': None, 'knn_index': None, 'tgt_index': None}

        bsz, seq_len, q_dim = queries.size()

        # search
        k = min(self.k, self.index.ntotal)
        dists, knns = self.index.search(queries.view(-1, q_dim), k)

        # get the value
        tgt_idx = self.vals[knns].to(queries.device).squeeze(-1)
        tgt_idx = tgt_idx.view(bsz, seq_len, -1)
        dists = dists.view(bsz, seq_len, -1)
        knns = knns.view(bsz, seq_len, -1)

        return {'distance': dists, 'knn_index': knns, 'tgt_index': tgt_idx}

    def retrieve_and_score(self, queries, **kwargs):
        knn_result = self.retrieve(queries)

        knn_distance = knn_result['distance']
        if knn_distance is None:
            return {"score": 0, "lambda": 0, "distance": None, "index": None}

        knn_index, tgt_index = knn_result['knn_index'], knn_result['tgt_index']

        knn_lambda = self.get_lambda(distance=knn_distance)
        knn_score = self.calculate_knn_prob(knn_index, tgt_index, knn_distance, queries, self.temperature)
        return {
            "score": knn_score,
            "lambda": knn_lambda,
            "distance": knn_distance,
            "tgt_index": tgt_index.squeeze(-1)
        }

    def calculate_knn_prob(self,
                           knn_index: torch.Tensor,  # [B, S, K]
                           tgt_index: torch.Tensor,  # [B, S, K]
                           distance: torch.Tensor,  # [B, S, K]
                           queries: torch.Tensor,  # [B, S, H]
                           temperature: torch.Tensor,  # [B, S, 1]
                           ):

        bsz, seq_len, _ = queries.size()

        re_compute_dists = -1 * distance  # [B, S, K]

        scaled_dists = re_compute_dists / temperature
        knn_weight = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)  # [B, S, K, 1]

        knn_tgt_prob = torch.zeros(bsz, seq_len, self.k, self.vocab_size).to(queries.device)  # [B, S, K, Vocab Size]
        tgt_index = tgt_index.unsqueeze_(-1)  # [B, S, K, 1]

        scatter(src=knn_weight.float(), out=knn_tgt_prob, index=tgt_index, dim=-1)

        prob = knn_tgt_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]

        return prob

    def _add_mask_value(self, key, value):
        if len(value) == 0:
            return
        _len, = value.shape
        self.vals[self.val_index: self.val_index + _len] = value.unsqueeze(-1)
        self.val_index += _len
        self.index.add(key)

    def _add(self, mask, key, value):
        value = value[mask].int()
        key = key[mask].float()
        self._add_mask_value(key, value)

    def add_datastore(self, key, value, **kwargs):
        mask = self.get_add_mask(value, **kwargs)
        self._add(mask, key, value)

    def get_add_mask(self, value, **kwargs):
        return value != self.padding_idx

    def get_normalized_probs(
            self,
            logits,
            log_probs,
            **extra
    ):
        logits = utils.softmax(logits, dim=-1, onnx_trace=False)
        knn_result = extra['knn_result']
        knn_lambda, knn_score = knn_result["lambda"], knn_result['score']
        score = logits * (1 - knn_lambda) + knn_score * knn_lambda

        if log_probs:
            # TODO 这会造成概率和大于1
            score_mask = score < 1e-10
            score.masked_fill_(score_mask, 1e-10)
            score = torch.log(score)
        return score


class PosteriorKNNDatastore(KNNDatastore):
    def __init__(self, args, task):
        super(PosteriorKNNDatastore, self).__init__(args, task)

        self.p_y = torch.zeros((self.vocab_size,)).cuda()
        self.p_h_y = dict()
        self.h_count = dict()

    def add_datastore(self, key, value, **kwargs):
        super(PosteriorKNNDatastore, self).add_datastore(key, value, **kwargs)

        # add
        assert key.size(0) == 1
        value = value.squeeze()
        self.p_y[value] += 1
        for index, v in zip(value.cpu().list()):
            if v in self.p_h_y:
                self.p_h_y[v] = key[0][index]
                self.h_count[v] = 1
            else:
                self.p_h_y[v] += key[0][index]
                self.h_count[v] += 1


class FixKNNDatastore(KNNDatastore):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--datastore-file', type=str, default="/home/wangdq/datastore/it")
        parser.add_argument('--dstore-size', type=int, default=1)

    def setup_index(self, task):
        datastore_file = self.args.datastore_file
        knn_index_file = os.path.join(datastore_file, "knn_index")
        value_file = os.path.join(datastore_file, "vals.npy")

        index = faiss.read_index(knn_index_file, faiss.IO_FLAG_ONDISK_SAME_DIR)
        self.dstore_size = self.args.dstore_size

        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)

        self.vals = np.memmap(value_file, dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.vals = torch.from_numpy(self.vals).cuda().to(torch.int64)

        self.val_index = 0

        return index

    def add_datastore(self, key, value, **kwargs):
        pass


class PositiveDatastore(KNNDatastore):
    def get_add_mask(self, value, **kwargs):
        reference, hypos = value, kwargs['hypo_value']
        positive_mask = reference.new_zeros(reference.shape).fill_(False).bool()
        batch_size, seq_len = reference.shape

        for b in range(batch_size):
            for i in range(seq_len):
                if reference[b][i] != self.padding_idx and reference[b][i] not in hypos[b]:
                    positive_mask[b][i] = True
        return positive_mask


class PositiveNegativeDatastore(object):
    def __init__(self, args, task, **kwargs):
        self.positive_datastore = KNNDatastore(args, task)
        self.negative_datastore = KNNDatastore(args, task)
        self.padding_idx = task.target_dictionary.pad()

    def retrieve_and_score(self, queries, **kwargs):
        positive_result = self.positive_datastore.retrieve_and_score(queries)
        negative_result = self.negative_datastore.retrieve_and_score(queries)
        return {"positive_result": positive_result,
                "negative_result": negative_result}

    def get_normalized_probs(
            self,
            logits,
            log_probs,
            **extra
    ):
        logits = utils.softmax(logits, dim=-1, onnx_trace=False)

        knn_result = extra['knn_result']
        positive_result, negative_result = knn_result['positive_result'], knn_result['negative_result']

        pos_score, pos_lambda = positive_result['score'], positive_result['lambda']
        neg_score, neg_lambda = negative_result['score'], negative_result['lambda']

        score = logits * (1 - pos_lambda + neg_lambda) + pos_score * pos_lambda - neg_score * neg_lambda

        score_mask = score < 0
        score.masked_fill_(score_mask, 0)
        score = score / score.sum(-1).unsqueeze(-1)

        if log_probs:
            score = torch.log(score)
        return score

    def add_datastore(self, key, value, **kwargs):
        positive_mask, negative_mask = self.get_add_mask(value, **kwargs)
        self.positive_datastore._add(positive_mask, key, value)

        hypo_key, hypo_value = kwargs['hypo_key'], kwargs['hypo_value']
        self.negative_datastore._add(negative_mask, hypo_key, hypo_value)

    def get_add_mask(self, value, **kwargs):
        reference, hypos = value, kwargs['hypo_value']

        positive_mask = reference.new_zeros(reference.shape).fill_(False).bool()
        negative_mask = reference.new_zeros(hypos.shape).fill_(False).bool()
        batch_size, seq_len = reference.shape

        for b in range(batch_size):
            for i in range(seq_len):
                if reference[b][i] != self.padding_idx and reference[b][i] not in hypos[b]:
                    positive_mask[b][i] = True

        for b in range(batch_size):
            for i in range(len(hypos[b])):
                if hypos[b][i] != self.padding_idx and hypos[b][i] not in reference[b]:
                    negative_mask[b][i] = True

        return positive_mask, negative_mask
