import torch


def get_whitening_method(whiten_method):
    if whiten_method == "svd":
        return SVDWhitening()
    if whiten_method == "var":
        return VarWhitening()
    if whiten_method == 'minmax':
        return MinMaxWhitening()
    if whiten_method == 'feature_dim':
        return FeatureDimWhitening()


class SVDWhitening(object):
    def __init__(self):
        self.mu = None
        self.W = None

    def whitening(self, embeddings):
        self.mu = torch.mean(embeddings, dim=0, keepdim=True)
        cov = torch.mm((embeddings - self.mu).t(), embeddings - self.mu)  # (feature dim) * (feature dim)
        u, s, vt = torch.svd(cov)
        self.W = torch.mm(u, torch.diag(1 / torch.sqrt(s)))
        return self.whitening_queries(embeddings)

    def whitening_queries(self, queries):
        embeddings = torch.mm(queries - self.mu, self.W)
        return embeddings


class VarWhitening(object):
    def __init__(self):
        self.mean = None
        self.std = None

    def whitening(self, embeddings):
        """ all key. embeddings: [B, D]"""
        self.mean = embeddings.mean(dim=0, keepdim=True)
        self.std = embeddings.std(dim=0, keepdim=True) + 1e-10
        return self.whitening_queries(embeddings)

    def whitening_queries(self, embeddings):
        embeddings = (embeddings - self.mean) / self.std
        return embeddings


class MinMaxWhitening(object):
    def __init__(self):
        self.min = None
        self.diff = None

    def whitening(self, embeddings):
        """ all key."""
        self.min = embeddings.min(0, keepdim=True)[0]
        self.diff = embeddings.max(0, keepdim=True)[0] - self.min + 1e-10
        return self.whitening_queries(embeddings)

    def whitening_queries(self, embeddings):
        embeddings = (embeddings - self.min) / self.diff
        return embeddings


class FeatureDimWhitening(object):
    def __init__(self):
        self.mean = None
        self.std = None

    def whitening(self, embeddings):
        """ all key."""
        return self.whitening_queries(embeddings)

    def whitening_queries(self, embeddings):
        mean = embeddings.mean(dim=1, keepdim=True)
        std = embeddings.std(dim=1, keepdim=True) + 1e-9
        embeddings = (embeddings - mean) / std
        return embeddings
