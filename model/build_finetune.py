import torch
import torch.nn as nn

from model import objectives
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights


class RepresentationAdapter(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return features + self.net(features)


class IRRA(nn.Module):
    """Minimal AERI finetune baseline.

    Supported training objectives:
        base    = aerial-text SDM + ground-text SDM
        base+id = base + identity classification on aerial, ground, and text features
        representation adapter = task-specific residual feature branch with CLIP-space cosine regularization
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
        self.rep_alpha = getattr(args, "rep_alpha", 0.7)
        self.rep_reg_weight = getattr(args, "rep_reg_weight", 0.5)
        rep_hidden_dim = getattr(args, "rep_hidden_dim", self.embed_dim * 2)
        self.aerial_rep_adapter = RepresentationAdapter(self.embed_dim, rep_hidden_dim)
        self.ground_rep_adapter = RepresentationAdapter(self.embed_dim, rep_hidden_dim)
        self.text_rep_adapter = RepresentationAdapter(self.embed_dim, rep_hidden_dim)
        if "id" in self.current_task:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [token.strip() for token in loss_names.split("+") if token.strip()]
        supported = {"base", "id"}
        unknown = [token for token in self.current_task if token not in supported]
        if unknown or "base" not in self.current_task:
            raise ValueError("This branch supports LOSS_NAMES='base' or 'base+id'.")
        print(f"Training Model with {self.current_task} tasks")

    def apply_representation_adapter(self, features, adapter):
        adapter_dtype = next(adapter.parameters()).dtype
        raw_features = features.float()
        rep_features = adapter(raw_features.to(adapter_dtype)).float()
        return self.rep_alpha * raw_features + (1.0 - self.rep_alpha) * rep_features

    def encode_image(self, image):
        image_feats = self.base_model.encode_image(image)
        return self.apply_representation_adapter(image_feats[:, 0, :], self.aerial_rep_adapter)

    def encode_text(self, text):
        text_feats = self.base_model.encode_text(text.long())
        raw_text_feats = text_feats[torch.arange(text_feats.shape[0]), text.argmax(dim=-1)]
        return self.apply_representation_adapter(raw_text_feats, self.text_rep_adapter)

    def forward(self, batch):
        images = batch["images"]
        ground_images = batch["ground_imgs"]
        caption_ids = batch["caption_ids"]

        device_type = images.device.type
        use_amp = device_type == "cuda"
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            image_feats, ground_image_feats, text_feats = self.base_model(
                images,
                ground_images,
                caption_ids,
            )

        aerial_raw_feats = image_feats[:, 0, :].float()
        ground_raw_feats = ground_image_feats[:, 0, :].float()
        text_raw_feats = text_feats[
            torch.arange(text_feats.shape[0], device=text_feats.device),
            caption_ids.argmax(dim=-1),
        ].float()
        aerial_feats = self.apply_representation_adapter(aerial_raw_feats, self.aerial_rep_adapter)
        ground_feats = self.apply_representation_adapter(ground_raw_feats, self.ground_rep_adapter)
        text_feats = self.apply_representation_adapter(text_raw_feats, self.text_rep_adapter)

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
        if self.rep_reg_weight > 0:
            rep_reg_loss = (
                1.0 - nn.functional.cosine_similarity(aerial_feats, aerial_raw_feats.detach(), dim=-1).mean()
                + 1.0 - nn.functional.cosine_similarity(ground_feats, ground_raw_feats.detach(), dim=-1).mean()
                + 1.0 - nn.functional.cosine_similarity(text_feats, text_raw_feats.detach(), dim=-1).mean()
            ) / 3.0
            ret["rep_reg_loss"] = rep_reg_loss * self.rep_reg_weight
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
