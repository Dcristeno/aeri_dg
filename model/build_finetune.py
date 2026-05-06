import torch
import torch.nn as nn

from model import objectives
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights


class IRRA(nn.Module):
    """Minimal AERI finetune baseline.

    Supported training objectives:
        base    = aerial-text SDM + ground-text SDM
        base+id = base + identity classification on aerial, ground, and text features
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
        self.retrieval_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        nn.init.eye_(self.retrieval_proj.weight)
        if "id" in self.current_task:
            self.id_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            nn.init.eye_(self.id_proj.weight)
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

    def encode_image(self, image):
        image_feats = self.base_model.encode_image(image)
        cls_feats = image_feats[:, 0, :]
        return self.retrieval_proj(cls_feats.to(self.retrieval_proj.weight.dtype)).float()

    def encode_text(self, text):
        text_feats = self.base_model.encode_text(text.long())
        cls_feats = text_feats[torch.arange(text_feats.shape[0]), text.argmax(dim=-1)]
        return self.retrieval_proj(cls_feats.to(self.retrieval_proj.weight.dtype)).float()

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
        retrieval_dtype = self.retrieval_proj.weight.dtype
        aerial_feats = self.retrieval_proj(aerial_raw_feats.to(retrieval_dtype)).float()
        ground_feats = self.retrieval_proj(ground_raw_feats.to(retrieval_dtype)).float()
        text_feats = self.retrieval_proj(text_raw_feats.to(retrieval_dtype)).float()

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
        if "id" in self.current_task:
            labels = batch["pids"].long()
            id_dtype = self.id_proj.weight.dtype
            classifier_dtype = self.classifier.weight.dtype
            aerial_id_feats = self.id_proj(aerial_raw_feats.to(id_dtype))
            ground_id_feats = self.id_proj(ground_raw_feats.to(id_dtype))
            text_id_feats = self.id_proj(text_raw_feats.to(id_dtype))
            aerial_logits = self.classifier(aerial_id_feats.to(classifier_dtype)).float()
            ground_logits = self.classifier(ground_id_feats.to(classifier_dtype)).float()
            text_logits = self.classifier(text_id_feats.to(classifier_dtype)).float()
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
