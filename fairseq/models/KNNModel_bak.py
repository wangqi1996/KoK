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


def whitening_torch_final(embeddings):
    mu = torch.mean(embeddings, dim=0, keepdim=True)
    cov = torch.mm((embeddings - mu).t(), embeddings - mu)  # (feature dim) * (feature dim)
    u, s, vt = torch.svd(cov)
    W = torch.mm(u, torch.diag(1 / torch.sqrt(s)))
    embeddings = torch.mm(embeddings - mu, W)
    return embeddings, mu, W


def whitening_queries(queries, mu, W):
    embeddings = torch.mm(queries - mu, W)
    return embeddings


def calculate_knn_prob(tgt_index: torch.Tensor,  # [B, S, K]
                       distance: torch.Tensor,  # [B, S, K]
                       temperature: torch.Tensor,  # [B, S, 1]
                       k=8,
                       vocab_size=1,
                       return_every_k=False,
                       ):
    bsz, seq_len, _ = distance.size()

    re_compute_dists = -1 * distance  # [B, S, K]
    # print(temperature)
    scaled_dists = re_compute_dists / temperature
    knn_weight = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)  # [B, S, K, 1]

    knn_tgt_prob = torch.zeros(bsz, seq_len, k, vocab_size).to(distance.device)  # [B, S, K, Vocab Size]
    tgt_index = tgt_index.unsqueeze_(-1)  # [B, S, K, 1]

    scatter(src=knn_weight.float(), out=knn_tgt_prob, index=tgt_index, dim=-1)

    if not return_every_k:
        prob = knn_tgt_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]
    else:
        prob = knn_tgt_prob  # [Batch size, seq len, k, vocab size]

    return prob


def compute_distance(knn_content, queries):
    d = (knn_content.float() - queries.float()) ** 2
    return d.sum(-1).to(queries)


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

        self.whitening = getattr(args, "whitening", "none")
        self.whitening_method = getattr(args, "whitening_method", 'svd')
        if self.whitening != "none":
            self.key = torch.zeros((self.dstore_size, self.dimension), dtype=torch.float).cuda()

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

        parser.add_argument("--linear-lambda", action="store_true")
        parser.add_argument("--min-lambda", type=float, default=0.0)
        parser.add_argument("--max-lambda", type=float, default=0.0)
        parser.add_argument('--lambda-value', type=float, default=0.3)
        parser.add_argument('--temperature-value', type=float, default=10)
        parser.add_argument('--k', type=int, default=8)

        parser.add_argument("--distance-threshold", type=float, default=-1)

        parser.add_argument('--whitening', type=str, default="none")  # retrieve, datastore
        parser.add_argument('--whitening-method', type=str, default="svd")

        from fairseq.models.KNNModel_bak import FixKNNDatastore
        FixKNNDatastore.add_args(parser)

        from fairseq.models.KNNModel import LabelTokenDatastore
        LabelTokenDatastore.add_args(parser)

    def get_dstore_size(self, task=None, **kwargs):
        return task.datasets['train'].tgt.sizes.sum() + 2

    def setup_index(self, task):

        self.dstore_size = self.get_dstore_size(task)

        index = faiss.IndexFlatL2(self.dimension)
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        # co.useFloat16 = True
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
        if self.index.ntotal < self.k:
            return {'distance': None, 'knn_index': None, 'tgt_index': None}

        bsz, seq_len, q_dim = queries.size()
        if self.whitening == "datastore":
            batch_size, seq_len, q_dim = queries.size()
            queries = self.whitening_method.whitening_queries(queries.view(-1, q_dim))
            queries = queries.view(batch_size, seq_len, q_dim)

        dists, knns = self.index.search(queries.view(-1, q_dim).contiguous(), self.k)

        tgt_idx = self.vals[knns].to(queries.device).squeeze(-1)
        tgt_idx = tgt_idx.view(bsz, seq_len, -1)
        dists = dists.view(bsz, seq_len, -1)
        knns = knns.view(bsz, seq_len, -1)

        return {'distance': dists, 'knn_index': knns, 'tgt_index': tgt_idx}

    def get_score(self, knn_result, return_every_k=False):
        knn_distance = knn_result['distance']
        if knn_distance is None:
            return {"score": 0, "lambda": 0, "distance": None, "index": None}

        knn_index, tgt_index = knn_result['knn_index'], knn_result['tgt_index']

        knn_lambda = self.get_lambda(distance=knn_distance)
        knn_score = self.calculate_knn_prob(tgt_index, knn_distance, self.temperature, return_every_k=return_every_k)
        return {
            "score": knn_score,
            "lambda": knn_lambda,
            "distance": knn_distance,
            "tgt_index": tgt_index.squeeze(-1)
        }

    def retrieve_and_score(self, queries, **kwargs):
        knn_result = self.retrieve(queries)
        score = self.get_score(knn_result)
        return score

    def add_mask_value(self, key, value):
        assert key.size(0) == value.size(0)
        if len(value) == 0:
            return
        _len, = value.shape
        self.vals[self.val_index: self.val_index + _len] = value.unsqueeze(-1)
        if self.whitening == "datastore":
            self.key[self.val_index: self.val_index + _len] = key  # 存储未变化前的数值
            key = self.whitening_method.whitening(self.key[: self.val_index + _len])
            # clear the datastore and store the new value
            self.index.reset()

        self.val_index += _len
        self.index.add(key)

    def _add(self, mask, key, value):
        value = value[mask].int()
        key = key[mask].float()
        self.add_mask_value(key, value)

    def add_datastore(self, key, value, **kwargs):
        mask = self.get_add_mask(value, **kwargs)
        self._add(mask, key, value)

    def get_add_mask(self, value, **kwargs):
        return value != self.padding_idx

    def get_normalized_probs(
            self,
            logits,
            log_probs,
            reference=None,
            **extra
    ):
        logits = utils.softmax(logits, dim=-1, onnx_trace=False)
        knn_result = extra['knn_result']
        knn_lambda, knn_score = knn_result["lambda"], knn_result['score']
        score = logits * (1 - knn_lambda) + knn_score * knn_lambda

        # # 计算reference的准确率
        # if isinstance(knn_lambda, torch.Tensor):
        #     batch_size, seq_len, vocab_size = logits.shape
        #     p_nmt = logits.view(-1, vocab_size)
        #     p_nmt_max, p_nmt_token = p_nmt.max(-1)
        #     p_nmt = p_nmt.gather(-1, reference.view(-1).unsqueeze(-1)).view(-1)
        #     p_knn = knn_score.view(-1, vocab_size).gather(-1, reference.view(-1).unsqueeze(-1)).view(-1)
        #     value_1 = p_knn > p_nmt
        #     lambda_1 = knn_lambda[value_1].sum().item()
        #     count_1 = value_1.long().sum().item()
        #     lambda_0 = knn_lambda[~value_1].sum().item()
        #     count_0 = (~value_1).long().sum().item()
        #     set_key_value("lambda_1", lambda_1)
        #     set_key_value("lambda_0", lambda_0)
        #     set_key_value("count_1", count_1)
        #     set_key_value("count_0", count_0)

        if log_probs:
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
    # def get_add_mask(self, value, **kwargs):
    #     reference, hypos = value, kwargs['hypo_value']
    #     positive_mask = reference.new_zeros(reference.shape).fill_(False).bool()
    #     batch_size, seq_len = reference.shape
    #
    #     for b in range(batch_size):
    #         for i in range(seq_len):
    #             if reference[b][i] != self.padding_idx and reference[b][i] not in hypos[b]:
    #                 positive_mask[b][i] = True
    #     return positive_mask

    def get_add_mask(self, value, **kwargs):
        p_nmt = kwargs.get('p_nmt', None)
        error = (p_nmt.argmax(-1) != value)
        return error


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


class DynamicD(KNNDatastore):
    def __init__(self, args, task):
        super(DynamicD, self).__init__(args, task)
        self.distance_threshold = 100

    def update_distance_threshold(self, key, value):
        knn_result = self.retrieve(key)
        retrieve_tokens = knn_result['tgt_index']
        if retrieve_tokens is None:
            return
        correct_mask = retrieve_tokens == value.unsqueeze(-1)
        distance = knn_result['distance'][correct_mask]
        distance = distance.sum() / len(distance)
        self.distance_threshold = self.distance_threshold * 0.9 + 0.1 * distance
        # print("threshold: ", self.distance_threshold)

    def add_datastore(self, key, value, **kwargs):
        self.update_distance_threshold(key, value)
        mask = self.get_add_mask(value, **kwargs)
        self._add(mask, key, value)
