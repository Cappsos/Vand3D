import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from collections.abc import Sequence
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False
from .configuration_m3d_clip import M3DCLIPConfig
from transformers import BertModel, BertConfig


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=True,
        rank=0,
        world_size=1,
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'

    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            pos_embed (str, optional): position embedding layer type. Defaults to "conv".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): faction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()
        
        
        ### !!!!!! HARDCODED SAVE ATTENTION = TRUE FOR DEBUGGING
        save_attn = True
        self.save_attn_flag = save_attn

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        
        self.blocks = nn.ModuleList(
            [
            
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            # if post_activation == "Tanh":
            #     self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            # else:
            #     self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def forward(self, x,features_list_indices=None):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        attentions_out = [] if self.save_attn_flag else None # MODIFIED: Initialize list for attentions
        features_list = []
        for ix,blk in enumerate(self.blocks):
            x = blk(x)
            hidden_states_out.append(x)
            
            if ix in features_list_indices:
                features_list.append(x)
            if self.save_attn_flag: # MODIFIED: Save 
                
                attentions_out.append(blk.attn.att_mat)
                
        x = self.norm(x)
        
        # if hasattr(self, "classification_head"):
        #     x = self.classification_head(x[:, 0])
        
        if self.save_attn_flag:
            
            return x, hidden_states_out, attentions_out, features_list
        else:
            return x, hidden_states_out


class M3DCLIP(PreTrainedModel):
    config_class = M3DCLIPConfig

    def __init__(self, config):
        super().__init__(config)
        self.vision_encoder = ViT(
            in_channels=config.in_channels,
            img_size=config.img_size,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            mlp_dim=config.mlp_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            pos_embed=config.pos_embed,
            dropout_rate=config.dropout_rate,
            spatial_dims=config.spatial_dims,
            classification=True,
        )
        # configuration = BertConfig()
        # self.language_encoder = BertModel(configuration)
        self.language_encoder = BertModel.from_pretrained(config.language_model_name_or_path)

        self.mm_vision_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.mm_language_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss

    def encode_image(self, image, features_list_indices=None):
        ### MODIFIED FOR HANDLINF save ATTENTION OUTPUTS
        vision_outputs = self.vision_encoder(image, features_list_indices=features_list_indices)
        
        if isinstance(vision_outputs, tuple) and len(vision_outputs) == 4: # MODIFIED: Check if attentions are returned
            image_feats_raw, _, attentions,features_list = vision_outputs
        elif isinstance(vision_outputs, tuple) and len(vision_outputs) == 2:
                image_feats_raw, _ = vision_outputs
                attentions = None
        else: # Should not happen if ViT.forward is one of the above
                image_feats_raw = vision_outputs 
                attentions = None


        image_feats_projected = self.mm_vision_proj(image_feats_raw)
        image_feats_normalized = F.normalize(image_feats_projected, dim=-1)

        if attentions is not None: # MODIFIED
            return image_feats_normalized, attentions,features_list
        else:
            return image_feats_normalized   
            

    def encode_text(self, input_id, attention_mask):
        text_feats = self.language_encoder(input_id, attention_mask=attention_mask)["last_hidden_state"]
        text_feats = self.mm_language_proj(text_feats)
        text_feats = F.normalize(text_feats, dim=-1)

        return text_feats


    def forward(self, images, input_ids, attention_mask, labels, **kwargs):
        image_features = self.encode_image(images)[:, 0]
        text_features = self.encode_text(input_ids, attention_mask)[:, 0]

        if self.gather_loss:
            all_image_features, all_text_features = gather_features(image_features, text_features)
            if self.local_loss:
                logits_per_image = self.logit_scale * image_features @ all_text_features.T
                logits_per_text = self.logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = self.logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = self.logit_scale * image_features @ text_features.T
            logits_per_text = self.logit_scale * text_features @ image_features.T

        loss = (
                           F.cross_entropy(logits_per_image, labels) +
                           F.cross_entropy(logits_per_text, labels)
                   ) / 2

        ret = {
            "loss": loss,
            "logits": (logits_per_image + logits_per_text) / 2.0,
        }

        return ret
