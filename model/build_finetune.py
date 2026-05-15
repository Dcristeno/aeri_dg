import torch
import torch.nn as nn
import torch.nn.functional as F

from model import objectives
from .clip_model import LayerNorm, QuickGELU, Transformer, build_CLIP_from_openai_pretrained, convert_weights


class IRRA(nn.Module):
    """Minimal AERI finetune baseline.

    Supported training objectives:
        base    = aerial-text SDM + ground-text SDM
        base+id = base + identity classification on aerial, ground, and text features
        bridge  = detached ground-view teacher alignment for aerial features
        fta     = fuzzy token alignment between aerial and text tokens
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
        if "id" in self.current_task:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)
        if "fta" in self.current_task:
            self._init_fta()

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [token.strip() for token in loss_names.split("+") if token.strip()]
        supported = {"base", "id", "bridge", "fta"}
        unknown = [token for token in self.current_task if token not in supported]
        if unknown or "base" not in self.current_task:
            raise ValueError("This branch supports LOSS_NAMES like 'base', 'base+id+bridge', or 'base+id+bridge+fta'.")
        print(f"Training Model with {self.current_task} tasks")

    def _init_fta(self):
        self.fta_num_query = int(getattr(self.args, "fta_num_query", 4))
        if self.fta_num_query <= 0:
            raise ValueError("fta_num_query must be positive.")

        self.fta_query = nn.Parameter(torch.randn(self.fta_num_query, self.embed_dim))
        self.fta_cross_attn = nn.MultiheadAttention(
            self.embed_dim,
            self.embed_dim // 64,
            batch_first=True,
        )
        self.fta_transformer = Transformer(
            width=self.embed_dim,
            layers=self.args.cmt_depth,
            heads=self.embed_dim // 64,
        )
        self.fta_ln_query = LayerNorm(self.embed_dim)
        self.fta_ln_token = LayerNorm(self.embed_dim)
        self.fta_ln_post = LayerNorm(self.embed_dim)
        self.fta_logsigma2 = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            QuickGELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
        )

        scale = self.fta_transformer.width ** -0.5
        proj_std = scale * ((2 * self.fta_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.fta_transformer.width) ** -0.5
        nn.init.normal_(self.fta_query, std=scale)
        nn.init.normal_(self.fta_cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.fta_cross_attn.out_proj.weight, std=proj_std)
        nn.init.normal_(self.fta_logsigma2[0].weight, std=fc_std)
        nn.init.normal_(self.fta_logsigma2[2].weight, std=proj_std)
        for block in self.fta_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def _fta_cross_former(self, query, tokens):
        query = query.float()
        tokens = tokens.float()
        x = self.fta_cross_attn(
            self.fta_ln_query(query),
            self.fta_ln_token(tokens),
            self.fta_ln_token(tokens),
            need_weights=False,
        )[0]
        x = x.permute(1, 0, 2)
        x = self.fta_transformer(x, None)
        x = x.permute(1, 0, 2)
        return self.fta_ln_post(x)

    def _compute_fuzzy_membership(self, query_tokens, cls_tokens):
        query_tokens = F.normalize(query_tokens.float(), dim=-1)
        cls_tokens = F.normalize(cls_tokens.float(), dim=-1)
        sigma2 = torch.exp(self.fta_logsigma2(cls_tokens))
        sigma2 = torch.clamp(sigma2, min=1e-6)

        query_exp = query_tokens.unsqueeze(2)
        cls_exp = cls_tokens.unsqueeze(0).unsqueeze(0)
        sigma_exp = sigma2.unsqueeze(0).unsqueeze(0)
        membership_dim = torch.exp(-((1.0 - query_exp * cls_exp) ** 2) / (2 * sigma_exp ** 2))
        return membership_dim.mean(dim=-1).transpose(1, 2).contiguous()

    def _compute_fta_loss(self, image_tokens, text_tokens, image_cls_feats, text_cls_feats, pids):
        batch_size = image_cls_feats.shape[0]
        query = self.fta_query.unsqueeze(0).expand(batch_size, -1, -1)

        query_v = self._fta_cross_former(query, image_tokens)
        query_t = self._fta_cross_former(query, text_tokens)

        mu_t2v = self._compute_fuzzy_membership(query_t, text_cls_feats)
        mu_v2t = self._compute_fuzzy_membership(query_v, image_cls_feats)

        query_t = F.normalize(query_t, dim=-1)
        query_v = F.normalize(query_v, dim=-1)
        sim_t2v = torch.einsum("bkd,Bkd->bBk", query_t, query_v)
        sim_v2t = torch.einsum("bkd,Bkd->bBk", query_v, query_t)
        mu_and = mu_t2v * mu_v2t
        sim_t2v = (sim_t2v * mu_and).mean(dim=-1)
        sim_v2t = (sim_v2t * mu_and).mean(dim=-1)

        return objectives.compute_sdm_from_similarity(
            sim_v2t,
            sim_t2v,
            pids,
            self.logit_scale,
        )

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
            image_feats, ground_image_feats, text_tokens = self.base_model(
                images,
                ground_images,
                caption_ids,
            )

        aerial_feats = image_feats[:, 0, :].float()
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

        if "fta" in self.current_task:
            fta_loss = self._compute_fta_loss(
                image_feats,
                text_tokens,
                aerial_feats,
                text_feats,
                batch["pids"],
            )
            ret["fta_loss"] = fta_loss * self.args.fta_loss_weight

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
