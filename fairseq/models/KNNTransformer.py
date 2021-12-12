from typing import Tuple, Optional, Dict, List, Any

import torch
from torch import Tensor

from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.KNNModel import build_knn_datastore, KNNDatastore
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerModel, transformer_wmt19_de_en, TransformerDecoder, \
    transformer_vaswani_wmt_en_de_big


class KNNTransformerDecoder(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, task=None, **kwargs):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.knn_datastore = build_knn_datastore(args, task)

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
            reference=None,
            **kwargs
    ):
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        feature = x
        if not features_only:
            x = self.output_layer(x)
            knn_result = self.knn_datastore.retrieve_and_score(feature, p_nmt=x.softmax(-1))
            extra['knn_result'] = knn_result
        return x, extra

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            **kwargs
    ):
        nmt_score = utils.softmax(net_output[0], dim=-1, onnx_trace=False)
        knn_score, lambda_value = self.knn_datastore.get_normalized_probs(net_output[1]['knn_result'])
        score = nmt_score * (1 - lambda_value) + knn_score * lambda_value

        if log_probs:
            score = torch.log(score)

        return score

    def add_datastore(self, sample, encoder_out, **kwargs):
        feature, _ = self.extract_features(
            prev_output_tokens=sample['net_input']['prev_output_tokens'],
            encoder_out=encoder_out
        )

        x = self.output_layer(feature)
        p_nmt = x.softmax(dim=-1, dtype=torch.float32)
        return self.knn_datastore.add_datastore(feature, sample['target'], p_nmt=p_nmt, **kwargs)



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


@register_model_architecture("knn_transformer", "knn_transformer_wmt14")
def knn_transformer_wmt14(args):
    transformer_vaswani_wmt_en_de_big(args)
