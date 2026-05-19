import torch
import torch.nn as nn
import torch.nn.functional as F

from model import objectives
from .clip_model import LayerNorm, QuickGELU, build_CLIP_from_openai_pretrained, convert_weights


class CdaFtaFusion(nn.Module):
    """Lightweight fusion of CDA routing and FTA query matching."""

    def __init__(self, embed_dim, num_query=4, condition_scale=1.0, sigma_floor=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query = num_query
        self.condition_scale = condition_scale
        self.sigma_floor = sigma_floor

        self.query = nn.Parameter(torch.randn(num_query, embed_dim))
        self.query_norm = LayerNorm(embed_dim)
        self.token_norm = LayerNorm(embed_dim)
        self.output_norm = LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, embed_dim // 64, batch_first=True)

        self.visual_condition = nn.Sequential(
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            QuickGELU(),
            nn.Linear(embed_dim, num_query * embed_dim),
        )
        self.text_condition = nn.Sequential(
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            QuickGELU(),
            nn.Linear(embed_dim, num_query * embed_dim),
        )
        self.log_sigma2 = nn.Sequential(
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            QuickGELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        scale = self.embed_dim ** -0.5
        proj_std = scale * 0.5
        fc_std = (2 * self.embed_dim) ** -0.5
        nn.init.normal_(self.query, std=scale)
        nn.init.normal_(self.cross_attn.in_proj_weight, std=scale)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)
        for module in [self.visual_condition, self.text_condition, self.log_sigma2]:
            nn.init.normal_(module[1].weight, std=fc_std)
            nn.init.normal_(module[3].weight, std=proj_std)
            nn.init.constant_(module[1].bias, 0.0)
            nn.init.constant_(module[3].bias, 0.0)

    def _conditioned_query(self, cls_feats, condition_mlp):
        batch_size = cls_feats.shape[0]
        base_query = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        delta = condition_mlp(cls_feats).reshape(batch_size, self.num_query, self.embed_dim)
        return base_query + self.condition_scale * delta

    def _attend_tokens(self, query, tokens):
        attended = self.cross_attn(
            self.query_norm(query),
            self.token_norm(tokens),
            self.token_norm(tokens),
            need_weights=False,
        )[0]
        return self.output_norm(query + attended)

    def _fuzzy_membership(self, queries, anchor_feats):
        queries = F.normalize(queries.float(), dim=-1)
        anchors = F.normalize(anchor_feats.float(), dim=-1)
        sigma2 = torch.exp(self.log_sigma2(anchors).float()).clamp(min=self.sigma_floor)

        query_exp = queries.unsqueeze(2)
        anchor_exp = anchors.unsqueeze(0).unsqueeze(0)
        sigma_exp = sigma2.unsqueeze(0).unsqueeze(0)
        membership = torch.exp(-((1.0 - query_exp * anchor_exp) ** 2) / (2.0 * sigma_exp ** 2))
        return membership.mean(dim=-1).transpose(1, 2).contiguous()

    def forward(self, image_tokens, text_tokens, image_cls_feats, text_cls_feats):
        visual_query = self._conditioned_query(image_cls_feats, self.visual_condition)
        text_query = self._conditioned_query(text_cls_feats, self.text_condition)

        visual_slots = F.normalize(self._attend_tokens(visual_query, image_tokens).float(), dim=-1)
        text_slots = F.normalize(self._attend_tokens(text_query, text_tokens).float(), dim=-1)

        sim_t2i_slots = torch.einsum("bkd,nkd->bnk", text_slots, visual_slots)
        sim_i2t_slots = torch.einsum("bkd,nkd->bnk", visual_slots, text_slots)
        membership_t2i = self._fuzzy_membership(text_slots, text_cls_feats)
        membership_i2t = self._fuzzy_membership(visual_slots, image_cls_feats)
        membership = membership_t2i * membership_i2t

        sim_t2i = (sim_t2i_slots * membership).mean(dim=-1)
        sim_i2t = (sim_i2t_slots * membership).mean(dim=-1)
        return sim_t2i, sim_i2t


class IRRA(nn.Module):
    """Minimal AERI finetune baseline.

    Supported training objectives:
        base    = aerial-text SDM + ground-text SDM
        base+id = base + identity classification on aerial, ground, and text features
        bridge  = detached ground-view teacher alignment for aerial features
        cfa     = lightweight CDA+FTA fusion loss
    """

    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice,
            args.img_size,
            args.stride_size,
        )
        self.embed_dim = base_cfg["embed_dim"]
        self.logit_scale = torch.ones([]) * (1 / args.temperature)
        if "cfa" in self.current_task:
            self.cfa_fusion = CdaFtaFusion(
                self.embed_dim,
                num_query=args.cfa_num_query,
                condition_scale=args.cfa_query_condition_scale,
                sigma_floor=args.cfa_sigma_floor,
            )
        if "id" in self.current_task:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [token.strip() for token in loss_names.split("+") if token.strip()]
        supported = {"base", "id", "bridge", "cfa"}
        unknown = [token for token in self.current_task if token not in supported]
        if unknown or "base" not in self.current_task:
            raise ValueError("This branch supports LOSS_NAMES='base', 'base+id', 'base+id+bridge', or '+cfa' variants.")
        print(f"Training Model with {self.current_task} tasks")

    def encode_image(self, image):
        image_feats = self.base_model.encode_image(image)
        return image_feats[:, 0, :].float()

    def encode_text(self, text):
        text_feats = self.base_model.encode_text(text.long())
        return text_feats[torch.arange(text_feats.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        images = batch["images"]
        ground_images = batch["ground_imgs"]
        caption_ids = batch["caption_ids"]

        device_type = images.device.type
        use_amp = device_type == "cuda"
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            image_tokens, ground_image_feats, text_tokens = self.base_model(
                images,
                ground_images,
                caption_ids,
            )

        aerial_feats = image_tokens[:, 0, :].float()
        ground_feats = ground_image_feats[:, 0, :].float()
        text_feats = text_tokens[
            torch.arange(text_tokens.shape[0], device=text_tokens.device),
            caption_ids.argmax(dim=-1),
        ].float()

        base_terms = objectives.compute_aeri_base_sdm_terms(
            aerial_feats,
            ground_feats,
            text_feats,
            batch["pids"],
            self.logit_scale,
        )
        base_loss = (
            base_terms["base_aerial_text"]
            + base_terms["base_ground_text"]
        )

        ret = {
            **base_terms,
            "base_loss": base_loss,
        }
        if "bridge" in self.current_task:
            bridge_loss = 1.0 - nn.functional.cosine_similarity(
                aerial_feats,
                ground_feats.detach(),
                dim=-1,
            ).mean()
            ret["bridge_loss"] = bridge_loss * self.args.bridge_loss_weight

        if "cfa" in self.current_task:
            selective_loss = objectives.compute_selective_bridge_sdm_loss(
                aerial_feats,
                ground_feats,
                text_feats,
                batch["pids"],
                self.logit_scale,
            )
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
                sim_t2i, sim_i2t = self.cfa_fusion(
                    image_tokens,
                    text_tokens,
                    image_cls_feats=aerial_feats,
                    text_cls_feats=text_feats,
                )
            fta_loss = objectives.compute_similarity_distribution_loss(
                sim_t2i,
                sim_i2t,
                batch["pids"],
                self.logit_scale,
            )
            cfa_loss = (
                self.args.cfa_selective_weight * selective_loss
                + self.args.cfa_fta_weight * fta_loss
            )
            ret["cfa_selective_term"] = selective_loss * self.args.cfa_loss_weight * self.args.cfa_selective_weight
            ret["cfa_fta_term"] = fta_loss * self.args.cfa_loss_weight * self.args.cfa_fta_weight
            ret["cfa_loss"] = cfa_loss * self.args.cfa_loss_weight

        if "id" in self.current_task:
            labels = batch["pids"].long()
            classifier_dtype = self.classifier.weight.dtype
            aerial_logits = self.classifier(aerial_feats.to(classifier_dtype)).float()
            ground_logits = self.classifier(ground_feats.to(classifier_dtype)).float()
            text_logits = self.classifier(text_feats.to(classifier_dtype)).float()
            id_loss = (
                nn.functional.cross_entropy(aerial_logits, labels)
                + nn.functional.cross_entropy(ground_logits, labels)
                + nn.functional.cross_entropy(text_logits, labels)
            ) / 3.0
            ret["id_loss"] = id_loss * self.args.id_loss_weight

        return ret


def build_finetune_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    convert_weights(model)
    return model
