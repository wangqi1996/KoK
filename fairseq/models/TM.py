from typing import Optional, Dict, Any, Tuple, List

import torch
from torch import nn, Tensor

from fairseq import utils
from fairseq.models import register_model_architecture, register_model
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder, \
    transformer_vaswani_wmt_en_de_big, Embedding
from fairseq.modules import MultiheadAttention, LayerNorm, FairseqDropout
from fairseq.tasks.translation import load_langpair_dataset


class MemoryDataset():
    def __init__(self, args, valid_split, tgtdict):
        filename = "/home/data_ti5_c/wangdq/data/ema/wmt14_mem/data-bin-"
        self.num_memory = 5
        self.args = args
        self.mem_dict = tgtdict
        if valid_split == "test":
            self.train_dataset = None
        else:
            self.train_dataset = [self.load_dataset(filename + str(i), "train") for i in range(self.num_memory)]
        self.valid_dataset = [self.load_dataset(filename + str(i), valid_split) for i in range(self.num_memory)]

    def get_samples(self, split, sample_id):
        if split == "train":
            dataset = self.train_dataset
        else:
            dataset = self.valid_dataset
        sample_id = sample_id.cpu().tolist()
        result = []

        for id in sample_id:
            for i in range(self.num_memory):
                result.append(dataset[i].__getitem__(id))

        result = dataset[0].collater(result, is_sort=False)
        mem_tokens = result['net_input']['src_tokens'].cuda()
        mem_len = result['net_input']['src_lengths'].cuda()
        return mem_tokens, mem_len

    def load_dataset(self, data_path, split):
        return load_langpair_dataset(data_path, split,
                                     self.args.target_lang, self.mem_dict,
                                     self.args.source_lang, None, combine=False,
                                     dataset_impl=self.args.dataset_impl,
                                     upsample_primary=self.args.upsample_primary,
                                     left_pad_source=self.args.left_pad_source,
                                     left_pad_target=self.args.left_pad_target,
                                     max_source_positions=self.args.max_source_positions,
                                     max_target_positions=self.args.max_target_positions,
                                     load_alignments=self.args.load_alignments,
                                     truncate_source=self.args.truncate_source,
                                     num_buckets=self.args.num_batch_buckets,
                                     shuffle=False,
                                     pad_to_multiple=self.args.required_seq_len_multiple,
                                     )


class MemoryDecoder(nn.Module):
    def __init__(self, args, dictionary):
        super(MemoryDecoder, self).__init__()
        self.args = args
        self.dictionary = dictionary

        self.embed_dim = self.args.decoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        mem_embed_tokens = self.get_mem_embedding()
        self.encoder = TransformerEncoder(args, self.dictionary, mem_embed_tokens)

        self.attention = self.build_attention(args)
        self.layer_norm = LayerNorm(self.embed_dim)

        input_dim, output_dim = args.decoder_embed_dim, args.decoder_ffn_embed_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            FairseqDropout(args.dropout, module_name=self.__class__.__name__),
            nn.Linear(output_dim, input_dim)
        )
        self.ff_layer_norm = LayerNorm(self.embed_dim)

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)

        self.gate = nn.Linear(self.embed_dim * 2, 2)

        self.num_mem = 5

        self.mem_dataset = MemoryDataset(args, valid_split=self.args.valid_subset, tgtdict=dictionary)

    def forward(self, hidden_state, sample=None, incremental_state=None, split="test", step=0, **kwargs):
        # encoder memory
        if step == 0:
            mem_tokens, mem_len = self.mem_dataset.get_samples(split, sample['id'])  # [batch_size, num_men, mem_len]
            mem_encode = self.encoder(mem_tokens, src_lengths=mem_len)
            mem_out, mem_padding_mask = self.reshape_mem_output(mem_encode, hidden_state.size(0))
        else:
            mem_out, mem_padding_mask = None, None

        hidden_state = hidden_state.transpose(0, 1)
        # attention attn: [batch_size, tgt_len, mem_len]; mem_len = single mem len * N
        x, attn = self.attention(
            query=hidden_state,
            key=mem_out,
            value=mem_out,
            key_padding_mask=mem_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=True
        )
        x = self.layer_norm(self.dropout_module(x))

        gate = utils.softmax(self.gate(torch.cat((hidden_state, x), dim=-1)), -1)  # [tgt_len, batch_size, 2]
        gen_gate, copy_gate = gate.chunk(2, dim=-1)

        hidden_state = self.layer_norm(hidden_state + x)

        # ffn
        x = self.fc(hidden_state)
        x = self.dropout_module(x) + x
        x = self.ff_layer_norm(x)
        return x.transpose(0, 1), mem_tokens, gen_gate, copy_gate, attn

    def build_attention(self, args):
        return MultiheadAttention(
            embed_dim=getattr(args, "encoder_embed_dim", None),
            num_heads=args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def reshape_mem_output(self, mem_output, batch_size):
        seq_len, _, dim = mem_output.encoder_out.shape

        encoder_out = mem_output.encoder_out.transpose(0, 1).reshape(batch_size, -1, dim).transpose(0, 1)
        encoder_padding_mask = mem_output.encoder_padding_mask.view(batch_size, -1)
        return encoder_out, encoder_padding_mask

    def get_mem_embedding(self):
        share_mem_emb = getattr(self.args, "share_mem_emb", False)
        if share_mem_emb:
            return self.embed_tokens

        num_embeddings = len(self.dictionary)
        padding_idx = self.dictionary.pad()

        emb = Embedding(num_embeddings, self.args.decoder_embed_dim, padding_idx)
        return emb


class TransformerTMDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super(TransformerTMDecoder, self).__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.mem_decoder = MemoryDecoder(args, dictionary)

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
            sample=None,
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
        x, mem_tokens, gen_gate, copy_gate, attn = self.mem_decoder(x, sample, incremental_state, **kwargs)
        # nmt probability
        nmt_probs = utils.softmax(self.output_layer(x), -1)  # [batch_size, tgt_len]
        nmt_probs = gen_gate.transpose(0, 1) * nmt_probs

        # mem probability
        batch_size, tgt_len, _ = nmt_probs.shape
        mem_input = mem_tokens.view(batch_size, -1).unsqueeze(-2).expand(-1, tgt_len, -1)
        attn = attn * (copy_gate.transpose(0, 1))
        probs = nmt_probs.scatter_add(-1, mem_input, attn)
        return probs

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            **kwargs
    ):
        if log_probs:
            return (net_output + 1e-10).log()
        else:
            return net_output

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_args('--share-mem-emb', action="store_true")


@register_model("transformer_tm")
class TransformerTM(TransformerModel):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, **kwargs):
        return TransformerTMDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


@register_model_architecture("transformer_tm", "tm_vaswani_wmt_en_de_big")
def tm_vaswani_wmt_en_de_big(args):
    transformer_vaswani_wmt_en_de_big(args)
