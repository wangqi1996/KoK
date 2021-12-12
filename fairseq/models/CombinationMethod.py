import torch


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


def count_knn_result(tgt_index, mask_for_label_count):
    # tgt_index = tgt_index.squeeze(-1)
    B, S, K = tgt_index.size()

    expand_tgt_idx = tgt_index.unsqueeze(-2).expand(B, S, K, K)
    expand_tgt_idx = expand_tgt_idx.masked_fill(mask_for_label_count[:K, :K], value=-1)

    labels_sorted, _ = expand_tgt_idx.sort(dim=-1)  # [B, S, K, K]
    labels_sorted[:, :, :, 1:] *= ((labels_sorted[:, :, :, 1:] - labels_sorted[:, :, :, :-1]) != 0).long()
    retrieve_label_counts = labels_sorted.ne(0).sum(-1)  # [B, S, K]
    retrieve_label_counts[:, :, :-1] -= 1

    return retrieve_label_counts

def build_weight(k):
    temp = [1.0 / (2 ** i) for i in range(1, k + 1)]
    temp[-1] = temp[-2] if k > 1 else 1
    return torch.Tensor(temp) / 2

class DisCountAbs():
    def __init__(self, k):
        self.mask_for_label_count = generate_label_count_mask(k)
        self.label_W = build_weight(k).__reversed__()
        self.distance_W = build_weight(k)
        self.label_count_dim = k
        self.distance_dim = k

    def get_index_dim(self, k, feature_num):
        return k * feature_num

    def get_datastore_size(self, sen_size):
        return sen_size + 10

    def get_label_count(self, retrieve_tgt_index):
        label_count = count_knn_result(retrieve_tgt_index, self.mask_for_label_count)
        feature = label_count.view(-1, self.label_count_dim).float()
        feature = feature * (self.label_W.to(feature))
        return feature

    def get_distance_feature(self, distance, **kwargs):
        feature = distance.view(-1, self.distance_dim)
        feature = feature * (self.distance_W.to(feature))
        return feature

    def extract_value(self, reference, p_nmt=None, p_knn=None):
        batch_size, seq_len, vocab_size = p_nmt.shape
        p_nmt = p_nmt.view(-1, vocab_size)
        p_nmt = p_nmt.gather(-1, reference.view(-1).unsqueeze(-1)).view(-1)

        p_knn = p_knn.view(-1, vocab_size)
        p_knn = p_knn.gather(-1, reference.view(-1).unsqueeze(-1)).view(-1)
        value = (p_nmt < p_knn).long()
        return value.view(-1)

