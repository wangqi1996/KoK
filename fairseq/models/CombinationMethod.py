import torch

import torch.nn.functional as F

RELATIVE_FLAT = "relative-flat"

# key=[d1, r2, r2, ...,rk, c2, c3, ..., ck] or key=[r2, r2, ...,rk, c2, c3, ..., ck]
FLAT = "flat"

# key=[d1]
TOP1DABS = "top1-d-abs"
# key=[d1, d2, ..., dk]
DISABS = "dis-abs"
# key=[d1, d2, ..., dk] + W
DISWABS = "dis-w-abs"

# key=[ck]
TOPKCABS = "topk-c-abs"
# key = [c1, c2,...,ck]
COUNTABS = "count-abs"
# key = [c1, c2,...,ck] + W
COUNTWABS = "count-w-abs"

# key=[d1, d2, ...,dk, c1, c2, c3,...,ck]
DISCOUNTABS = "dis-count-abs"
DISCOUNTWABS = "dis-count-w-abs"
DISCOUNTKLWABS = "dis-count-kl-w-abs"

TOP1_ALL = "top1-all"
TOP1_TOP1 = "top1-top1"
RELATIVE_FLAT2 = 'relative-flat2'

FLAT_LIST = [RELATIVE_FLAT, FLAT, RELATIVE_FLAT2]
TOP1_LIST = [TOP1_ALL, TOP1_TOP1]

""" value method"""
INALL = 'in-all'
INTOP1 = 'in-top1'
VSALL = 'vs-all'
VSTOP1 = 'vs-top1'
VSALL1 = 'vs-all-1'


def get_combination_class(method, k, value_method, token_datastore):
    if method == RELATIVE_FLAT:
        return Flat(k, value_method, token_datastore, relative_distance=True, append_distance=False)
    if method == RELATIVE_FLAT2:
        return Flat(k, value_method, token_datastore, relative_distance=True, append_distance=True)
    if method == FLAT:
        return Flat(k, value_method, token_datastore)

    # key = [d1]
    if method == TOP1DABS:
        return Top1DAbsolute(k, value_method, token_datastore)

    # key = [d1, d2, ... , dk]
    if method == DISABS:
        return DisAbsolute(k, value_method, token_datastore, weight=False)
    if method == DISWABS:
        return DisAbsolute(k, value_method, token_datastore, weight=True)

    # key = [c8]
    if method == TOPKCABS:
        return TopKCAbsolute(k, value_method, token_datastore)

    # key = [c1, c2,...,c8]
    if method == COUNTABS:
        return CountAbsolute(k, value_method, token_datastore, weight=False)
    if method == COUNTWABS:
        return CountAbsolute(k, value_method, token_datastore, weight=True)

    # key=[d1, d2, ...,dk, c1, c2, c3,...,ck]
    if method == DISCOUNTABS:
        return DisCountAbs(k, value_method, token_datastore, weight=False)
    if method == DISCOUNTWABS:
        return DisCountAbs(k, value_method, token_datastore, weight=True)

    # key = [d1, d2, ..., dk, c1, c2, c3, ..., ck, kl(p_knn, p_nmt), kl(p_nmt, p_knn)]
    if method == DISCOUNTKLWABS:
        return DisCountKLAbs(k, value_method, token_datastore, weight=True)

    # if method == TOP1_ALL:
    #     return Top1(k, value_method, token_datastore, store_all=True)
    # if method == TOP1_TOP1:
    #     return Top1(k, value_method, token_datastore, store_all=False)


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


class CombinationMethod(object):
    def __init__(self, k, value_method="", token_datastore=None, **kwargs):
        self.k = k
        self.mask_for_label_count = generate_label_count_mask(self.k)
        self.value_method = value_method
        self.token_datastore = token_datastore

    def get_index_dim(self, k, feature_num, **kwargs):
        pass

    def get_datastore_size(self, sen_size, k, **kwargs):
        pass

    def get_distance_feature(self, distance, search=False):
        pass

    def get_label_count(self, retrieve_tgt_index, search=False, **kwargs):
        pass

    def extract_value(self, retrieve_tokens, reference, **kwargs):
        pass


def count_knn_result(tgt_index, mask_for_label_count):
    tgt_index = tgt_index.squeeze(-1)
    B, S, K = tgt_index.size()

    expand_tgt_idx = tgt_index.unsqueeze(-2).expand(B, S, K, K)
    expand_tgt_idx = expand_tgt_idx.masked_fill(mask_for_label_count[:K, :K], value=-1)

    labels_sorted, _ = expand_tgt_idx.sort(dim=-1)  # [B, S, K, K]
    labels_sorted[:, :, :, 1:] *= ((labels_sorted[:, :, :, 1:] - labels_sorted[:, :, :, :-1]) != 0).long()
    retrieve_label_counts = labels_sorted.ne(0).sum(-1)  # [B, S, K]
    retrieve_label_counts[:, :, :-1] -= 1

    return retrieve_label_counts


class Flat(CombinationMethod):
    """
    key: [d1, d2, c1, c2]
    value: p_knn(y_t) vs p_nmt(y_t)   or  y_t in [t_1, t_2]
    """

    def __init__(self, k, value_method, token_datastore, relative_distance=True, append_distance=False):
        super(Flat, self).__init__(k, value_method, token_datastore)
        assert relative_distance
        self.relative_distance = relative_distance
        self.append_distance = append_distance
        self.distance_dim = k - 1
        self.label_count_dim = k - 1

    def get_index_dim(self, k, feature_num, **kwargs):
        dim = k * feature_num
        if self.append_distance:
            dim += 1
            self.distance_dim += 1
        dim -= feature_num  # remove  d0 r0
        return dim

    def get_datastore_size(self, sen_size, k, **kwargs):
        return sen_size + 10

    def get_distance_feature(self, distance, **kwargs):
        """ distance: [batch, seq_len, K] """

        top_distance = None
        if self.append_distance:
            top_distance = distance[:, :, 0].unsqueeze(-1)

        if self.relative_distance:
            distance_mask = distance == 0
            distance.masked_fill_(distance_mask, 1e-6)
            distance = distance / distance[:, :, 0].unsqueeze(-1)
            distance = distance[:, :, 1:]  # remove the r1 == 1

        if self.append_distance:
            distance = torch.cat((top_distance, distance), dim=-1)

        return distance.view(-1, self.distance_dim)

    def get_label_count(self, retrieve_tgt_index, **kwargs):
        label_count = count_knn_result(retrieve_tgt_index, self.mask_for_label_count)
        label_count = label_count[:, :, 1:]
        return label_count.view(-1, self.label_count_dim)

    def extract_value(self, retrieve_tokens, reference, p_nmt=None, p_knn=None, knn_result=None, **kwargs):
        """
        reference: [B, S] retrieve_tokens: [B, S, K]
        1. y_t in [t_1, t_2]
        2. y_t in [t_1]
        3. p_knn(y_t) vs p_nmt(y_t)
        """
        assert self.value_method != VSTOP1
        if self.value_method == INALL:
            """ y_t in [t_1, t_2] """
            value = (reference.unsqueeze(-1) == retrieve_tokens).long().sum(-1)
            value_mask = value > 1
            value.masked_fill_(value_mask, 1)
        elif self.value_method == INTOP1:
            """ y_t in [t_1] """
            value = (reference == retrieve_tokens[:, :, 0]).long()
        elif self.value_method in [VSALL1, VSALL]:
            """ p_knn(y_t) vs p_nmt(y_t) """
            batch_size, seq_len, vocab_size = p_nmt.shape
            p_nmt = p_nmt.view(-1, vocab_size)
            p_nmt = p_nmt.gather(-1, reference.view(-1).unsqueeze(-1)).view(-1)

            p_knn = p_knn.view(-1, vocab_size)
            p_knn = p_knn.gather(-1, reference.view(-1).unsqueeze(-1)).view(-1)
            value = (p_nmt < p_knn).long()

        return value.view(-1)


class Top1DAbsolute(Flat):
    def get_index_dim(self, k, feature_num, **kwargs):
        return 1

    def get_distance_feature(self, distance, **kwargs):
        top_distance = distance[:, :, 0]
        return top_distance.view(-1, 1)


class DisAbsolute(Flat):
    def __init__(self, k, value_method, token_datastore, weight=False):
        super().__init__(k, value_method, token_datastore)

        self.weight = weight
        if self.weight:
            self.W = torch.Tensor([1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 128]).unsqueeze(0)

    def get_index_dim(self, k, feature_num, **kwargs):
        self.distance_dim = k
        return k

    def get_distance_feature(self, distance, **kwargs):
        feature = distance.view(-1, self.distance_dim)
        if self.weight:
            feature = feature * (self.W.to(feature))
        return feature


class TopKCAbsolute(Flat):
    def get_index_dim(self, k, feature_num, **kwargs):
        return 1

    def get_label_count(self, retrieve_tgt_index, **kwargs):
        label_count = count_knn_result(retrieve_tgt_index, self.mask_for_label_count)
        label_count = label_count[:, :, -1]
        return label_count.view(-1, 1)


class CountAbsolute(Flat):
    def __init__(self, k, value_method, token_datastore, weight=False):
        super().__init__(k, value_method, token_datastore)

        self.weight = weight
        if self.weight:
            self.W = torch.Tensor([1 / 128, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2]).unsqueeze(0)

    def get_index_dim(self, k, feature_num, **kwargs):
        self.label_count_dim = k
        return k

    def get_label_count(self, retrieve_tgt_index, **kwargs):
        label_count = count_knn_result(retrieve_tgt_index, self.mask_for_label_count)
        feature = label_count.view(-1, self.label_count_dim).float()
        if self.weight:
            feature = feature * (self.W.to(feature))
        return feature


class DisCountAbs(Flat):
    def __init__(self, k, value_method, token_datastore, weight=False):
        super().__init__(k, value_method, token_datastore)

        self.weight = weight
        if self.weight:
            self.label_W = torch.Tensor([1 / 128, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2]).unsqueeze(
                0) / 2
            self.distance_W = torch.Tensor([1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 128]).unsqueeze(
                0) / 2

    def get_index_dim(self, k, feature_num, **kwargs):
        self.label_count_dim = k
        self.distance_dim = k
        return k * feature_num

    def get_label_count(self, retrieve_tgt_index, **kwargs):
        label_count = count_knn_result(retrieve_tgt_index, self.mask_for_label_count)
        feature = label_count.view(-1, self.label_count_dim).float()
        if self.weight:
            feature = feature * (self.label_W.to(feature))
        return feature

    def get_distance_feature(self, distance, **kwargs):
        feature = distance.view(-1, self.distance_dim)
        if self.weight:
            feature = feature * (self.distance_W.to(feature))
        return feature


class DisCountKLAbs(DisCountAbs):
    def __init__(self, k, value_method, token_datastore, weight=False):
        super().__init__(k, value_method, token_datastore, weight)

        if self.weight:
            self.label_W = torch.Tensor([1 / 128, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2]).unsqueeze(
                0) / 3
            self.distance_W = torch.Tensor([1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 128]).unsqueeze(
                0) / 3
            self.kl_W = torch.Tensor([1 / 2, 1 / 2]).unsqueeze(0) / 3

    def get_index_dim(self, k, feature_num, **kwargs):
        self.label_count_dim = k
        self.distance_dim = k
        self.kl_dim = 2
        return k * (feature_num - 1) + self.kl_dim

    def get_kl(self, p_knn, p_nmt):
        batch_size, seq_len, _ = p_nmt.size()
        forward_kl = -1 * F.kl_div((p_knn + 1e10).log(), p_nmt, reduction='none').sum(-1).view(batch_size, -1)
        backward_kl = -1 * F.kl_div((p_nmt + 1e10).log(), p_knn, reduction='none').sum(-1).view(batch_size, -1)

        feature = torch.cat((forward_kl, backward_kl), dim=-1).view(batch_size * seq_len, self.kl_dim)
        if self.weight:
            feature = feature * (self.kl_W.to(feature))
        return feature

# class Top1(CombinationMethod):
#     def __init__(self, k, value_method, token_datastore, store_all=False):
#         super(Top1, self).__init__(k, value_method, token_datastore)
#         self.store_all = store_all
#         self.distance_dim = 1
#         self.label_count_dim = 1
#
#     def get_index_dim(self, k, feature_num, **kwargs):
#         return feature_num
#
#     def get_datastore_size(self, sen_size, k, **kwargs):
#         if self.store_all:
#             return sen_size * k + 10
#         return sen_size + 10
#
#     def get_distance_feature(self, distance, search=False):
#         """
#         distance: [batch, seq_len, K]
#         search: when retrieve, we only need return the top distance.
#         """
#         store_all = self.store_all and not search
#         if not store_all:
#             distance = distance[:, :, 0].unsqueeze(-1)
#         return distance.view(-1, self.distance_dim)
#
#     def get_label_count(self, retrieve_tgt_index, search=False, **kwargs):
#         label_count = count_knn_result(retrieve_tgt_index, self.mask_for_label_count)
#         store_all = self.store_all and not search
#         if not store_all:
#             label_count = label_count[:, :, 0]
#         return label_count.view(-1, self.label_count_dim)
#
#     def extract_value(self, retrieve_tokens, reference, p_nmt=None, knn_result=None, **kwargs):
#         """ call when save datastore.
#         1. y_t == t_i
#         2.
#         """
#         assert self.value_method != INALL
#         if self.value_method == INTOP1:
#             if self.store_all:
#                 """ y_t == t_i """
#                 value = (retrieve_tokens == reference.unsqueeze(-1)).long()
#             else:
#                 """y_t == t_0"""
#                 value = (retrieve_tokens[:, :, 0] == reference).long()
#             return value.view(-1)
#
#         batch, seq_len, vocab_size = p_nmt.shape
#         p_nmt = p_nmt.view(-1, vocab_size)
#         p_nmt = p_nmt.gather(-1, reference.view(-1).unsqueeze(-1))
#
#         if self.value_method == VSTOP1:
#             p_knn = self.token_datastore.get_score(knn_result, return_every_k=True)['score']
#             if self.store_all:
#                 """ p_knn_i(y_t) vs p_nmt(y_i) """
#                 K = p_knn.size(-2)
#                 reference = reference.unsqueeze(-1).repeat(1, 1, K)
#                 p_nmt = p_nmt.repeat(1, K)
#                 p_knn = p_knn.view(-1, vocab_size).gather(-1, reference.view(-1).unsqueeze(-1))
#             else:
#                 """ p_knn_0(y_t) vs p_nmt(y_i) """
#                 p_knn = p_knn[:, :, 0, :].view(-1, vocab_size)
#                 p_knn = p_knn.gather(-1, reference.view(-1).unsqueeze(-1))
#             value = (p_nmt.view(-1, 1) < p_knn).long()
#
#         # elif self.value_method == VSALL:
#         #     p_knn = self.token_datastore.get_score(knn_result)['score']
#         #     p_knn = p_knn.view(-1, vocab_size)
#         #     p_knn = p_knn.gather(-1, reference.view(-1).unsqueeze(-1))
#         #     value = (p_nmt.view(-1, 1) < p_knn).long()
#         #     if self.store_all:
#         #         K = retrieve_tokens.size(-1)
#         #         value = value.view(batch, seq_len).unsqueeze(-1).repeat(1, 1, K)
#         #         value.masked_fill_(reference.unsqueeze(-1) != retrieve_tokens, 0)
#         #         # only select for top1??
#         #         # value[:, :, 1:] = False
#
#         return value.view(-1)
