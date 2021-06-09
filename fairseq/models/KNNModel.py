import faiss
import torch
from torch_scatter import scatter

from fairseq import utils


def build_knn_datastore(args, task, **kwargs):
    knn_type = args.knn_type
    if knn_type == "normal":
        return KNNDatastore(args, task, **kwargs)


class KNNDatastore(object):

    def __init__(self, args, task):
        self.dstore_size = task.datasets['train'].tgt.sizes.sum() + 2
        self.dimension = args.decoder_embed_dim
        self.index = self.setup_index()
        self.k = args.k

        self.temperature = args.knn_temperature_value
        self.lambda_value = args.knn_lambda_value
        self.vocab_size = len(task.target_dictionary)
        self.padding_idx = task.target_dictionary.pad()

        self.linear_lambda = getattr(args, "linear_lambda", False)
        self.min_lambda = getattr(args, "min_lambda", 0.3)
        self.max_lambda = getattr(args, "max_lambda", 0.3)
        self.distance_threshold = getattr(args, "distance_threshold", -1)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--linear-lambda", action="store_true")
        parser.add_argument("--min-lambda", type=float, default=0.0)
        parser.add_argument("--max-lambda", type=float, default=0.0)

        parser.add_argument("--distance-threshold", type=float, default=-1)

    def setup_index(self):
        index = faiss.IndexFlatL2(self.dimension)
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)

        self.vals = torch.zeros((self.dstore_size, 1)).cuda()
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

    def get_temperature(self):
        return self.temperature

    def dist_func(self, d, **kwargs):
        return -1 * d

    def get_knns(self, queries):
        k = min(self.k, self.index.ntotal)
        dists, knns = self.index.search(queries, k)
        return dists, knns

    def retrieve(self, queries):
        if self.index.ntotal <= 0:
            return {'distance': None, 'knn_index': None, 'tgt_index': None}

        bsz = queries.size(0)
        seq_len = queries.size(1)

        dists, knns = self.get_knns(queries.contiguous().view(-1, queries.size(-1)))

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

        knn_index = knn_result['knn_index']
        tgt_index = knn_result['tgt_index']

        knn_lambda = self.get_lambda(distance=knn_distance)
        knn_temperature = self.get_temperature()
        knn_score = self.calculate_knn_prob(
            knn_index, tgt_index, knn_distance,
            queries, knn_temperature
        )
        return {
            "score": knn_score,
            "lambda": knn_lambda,
            "distance": knn_distance,
        }

    def calculate_knn_prob(self,
                           knn_index: torch.Tensor,  # [B, S, K]
                           tgt_index: torch.Tensor,  # [B, S, K]
                           distance: torch.Tensor,  # [B, S, K]
                           queries: torch.Tensor,  # [B, S, H]
                           temperature: torch.Tensor,  # [B, S, 1]
                           ):

        bsz = queries.size(0)
        seq_len = queries.size(1)

        re_compute_dists = self.dist_func(distance, knn_index, queries, function=self.sim_func)  # [B, S, K]

        scaled_dists = re_compute_dists / temperature
        knn_weight = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)  # [B, S, K, 1]

        knn_tgt_prob = torch.zeros(bsz, seq_len, self.k, self.vocab_size).to(queries.device)  # [B, S, K, Vocab Size]
        tgt_index = tgt_index.unsqueeze_(-1)  # [B, S, K, 1]

        scatter(src=knn_weight.float(), out=knn_tgt_prob, index=tgt_index, dim=-1)

        prob = knn_tgt_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]

        return prob

    def _add(self, mask, key, value):
        value = value[mask].int()
        key = key[mask].float()
        if len(value) == 0:
            return
        _len, = value.shape
        self.vals[self.val_index: self.val_index + _len] = value.unsqueeze(-1)
        self.val_index += _len
        self.index.add(key)

    def add_datastore(self, pad_index, key, value):
        mask = self.get_add_mask(pad_index, value)
        self._add(mask, key, value)

    def get_add_mask(self, pad_index, value, **kwargs):
        return value != pad_index

    def get_normalized_probs(
            self,
            logits,
            **extra
    ):
        logits = utils.softmax(logits, dim=-1, onnx_trace=False)
        knn_result = extra['knn_result']
        knn_lambda, knn_score = knn_result["knn_lambda"], knn_result['knn_score']
        score = logits * (1 - knn_lambda) + knn_score * knn_lambda
        return score


class PositiveDatastore(KNNDatastore):
    def get_add_mask(self, pad_index, value, **kwargs):
        reference, hypos = value, kwargs['hypo_value']
        positive_mask = reference.new_zeros(reference.shape).fill_(False)
        batch_size, seq_len = reference.shape

        for b in range(batch_size):
            for i in range(seq_len):
                if reference[b][i] != pad_index and reference[b][i] not in hypos[b]:
                    positive_mask[b][i] = True
        return positive_mask
