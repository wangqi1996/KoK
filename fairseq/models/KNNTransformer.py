from typing import Tuple, Optional, Dict, List

from torch import Tensor

from fairseq.models import BaseFairseqModel
from fairseq.models.KNNModel import build_knn_datastore, KNNDatastore
from fairseq.models.transformer import TransformerModel


class KNNTransformer(BaseFairseqModel):
    def __init__(self, args, transformer_model, knn_datastore):
        super(KNNTransformer, self).__init__()
        self.transformer_model = transformer_model
        self.knn_datastore = knn_datastore
        self.args = args`

    def load_state_dict(self, state_dict, strict=True, args=None):
        self.transformer_model.load_state_dict(state_dict, strict, args)

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        KNNDatastore.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        transformer_model = TransformerModel.build_model(args, task)
        knn_datastore = build_knn_datastore(args, task)
        return cls(args, transformer_model, knn_datastore)

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            **kwargs
    ):
        hidden_state, extra = self.transformer_model.forward(
            src_tokens, src_lengths, prev_output_tokens, **kwargs
        )
        feature = extra['feature']
        knn_result = self.knn_model.retrieve_and_score(feature)
        return hidden_state, extra, knn_result

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        assert log_probs, "only support log probs!"
        logits = net_output[0]
        knn_result = net_output[1]
        self.knn_datastore.get_normalized_probs(logits, knn_result=knn_result)
