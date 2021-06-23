import torch


def get_whitening_method(whiten_method):
    if whiten_method == "svd":
        return SVDWhitening()
    if whiten_method == "var":
        return VarWhitening()


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
        """ all key."""
        self.mean = embeddings.mean(dim=0)
        self.std = embeddings.std(dim=0) + 1e-9
        return self.whitening_queries(embeddings)

    def whitening_queries(self, embeddings):
        embeddings = (embeddings - self.mean) / self.std
        return embeddings
