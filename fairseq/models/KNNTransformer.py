from typing import Tuple, Optional, Dict, List, Any

from torch import Tensor

from fairseq.models import register_model, register_model_architecture
from fairseq.models.KNNModel import build_knn_datastore, KNNDatastore
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerModel, transformer_wmt19_de_en, TransformerDecoder


class KNNTransformerDecoder(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, task=None, **kwargs):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.knn_datastore = build_knn_datastore(args, task)
        self.knn_type = args.knn_type

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        knn_result = self.knn_datastore.retrieve_and_score(x)
        if not features_only:
            x = self.output_layer(x)
        return x, extra, knn_result

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        logits = net_output[0]
        knn_result = net_output[2]
        score = self.knn_datastore.get_normalized_probs(logits, log_probs, knn_result=knn_result)
        return score

    def add_datastore(self, sample, encoder_out, **kwargs):
        feature, _ = self.extract_features(
            prev_output_tokens=sample['net_input']['prev_output_tokens'],
            encoder_out=encoder_out
        )
        hypo_key = None
        if self.knn_type == "positive-negative":
            tokens = self.compute_hypos_input(kwargs["hypo_value"])
            hypo_key, _ = self.extract_features(
                prev_output_tokens=tokens,
                encoder_out=encoder_out
            )
        return self.knn_datastore.add_datastore(feature, sample['target'], hypo_key=hypo_key, **kwargs)

    def compute_hypos_input(self, hypo_value):
        assert hypo_value.size(0) == 1
        tokens = hypo_value[0].clone()
        tokens[1:] = hypo_value[0][0:-1]
        tokens[0] = self.dictionary.eos()
        return tokens.unsqueeze(0)


@register_model("knn_transformer")
class KNNTransformer(TransformerModel):

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        KNNDatastore.add_args(parser)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, **kwargs):
        return KNNTransformerDecoder(args, tgt_dict, embed_tokens, **kwargs)

    def post_process(self, sample, encoder_out, **kwargs):
        self.decoder.add_datastore(sample, encoder_out, **kwargs)


@register_model_architecture("knn_transformer", "knn_transformer_wmt19")
def knn_transformer_wmt19(args):
    transformer_wmt19_de_en(args)
