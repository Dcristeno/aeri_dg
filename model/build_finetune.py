import torch
import torch.nn as nn

from model import objectives
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights


class IRRA(nn.Module):
    """Minimal AERI finetune baseline.

    Supported training objectives:
        base    = aerial-text SDM + ground-text SDM
        base+id = base + identity classification on aerial, ground, and text features
        bridge  = detached ground-view teacher alignment for aerial features
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

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [token.strip() for token in loss_names.split("+") if token.strip()]
        supported = {"base", "id", "bridge"}
        unknown = [token for token in self.current_task if token not in supported]
        if unknown or "base" not in self.current_task:
            raise ValueError("This branch supports LOSS_NAMES='base', 'base+id', or 'base+id+bridge'.")
        print(f"Training Model with {self.current_task} tasks")

    def _pool_image_tokens(self, image_feats):
        cls_feat = image_feats[:, 0, :].float()
        if getattr(self.args, "image_pooling", "cls").lower() == "cls":
            return cls_feat

        if getattr(self.args, "image_pooling", "cls").lower() != "foreground":
            raise ValueError(f"Unsupported image_pooling: {self.args.image_pooling}")

        patch_feats = image_feats[:, 1:, :].float()
        num_patches = patch_feats.shape[1]
        grid_h = self.args.img_size[0] // self.args.stride_size
        grid_w = self.args.img_size[1] // self.args.stride_size
        if grid_h * grid_w != num_patches:
            grid_h = int(num_patches ** 0.5)
            grid_w = max(num_patches // max(grid_h, 1), 1)
        if grid_h * grid_w != num_patches:
            return cls_feat

        token_score = patch_feats.norm(dim=-1)
        token_score = (token_score - token_score.mean(dim=1, keepdim=True)) / (
            token_score.std(dim=1, keepdim=True) + 1e-6
        )

        y = torch.linspace(0.0, 1.0, grid_h, device=patch_feats.device, dtype=patch_feats.dtype)
        x = torch.linspace(0.0, 1.0, grid_w, device=patch_feats.device, dtype=patch_feats.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        center_prior = torch.exp(-(((xx - 0.5) / 0.28) ** 2 + ((yy - 0.5) / 0.42) ** 2) / 2.0)
        depth_prior = yy
        prior = (
            self.args.foreground_center_weight * center_prior
            + self.args.foreground_depth_weight * depth_prior
        ).reshape(1, num_patches)

        tau = max(float(self.args.foreground_pool_tau), 1e-6)
        weights = torch.softmax((token_score + prior) / tau, dim=1)
        foreground_feat = torch.sum(weights.unsqueeze(-1) * patch_feats, dim=1)
        mix = min(max(float(self.args.foreground_pool_mix), 0.0), 1.0)
        return (1.0 - mix) * cls_feat + mix * foreground_feat

    def encode_image(self, image):
        image_feats = self.base_model.encode_image(image)
        return self._pool_image_tokens(image_feats)

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
            image_feats, ground_image_feats, text_feats = self.base_model(
                images,
                ground_images,
                caption_ids,
            )

        aerial_feats = self._pool_image_tokens(image_feats)
        ground_feats = self._pool_image_tokens(ground_image_feats)
        text_feats = text_feats[
            torch.arange(text_feats.shape[0], device=text_feats.device),
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
            pair_loss = 1.0 - nn.functional.cosine_similarity(
                aerial_feats,
                ground_feats.detach(),
                dim=-1,
            )
            bridge_mode = getattr(self.args, "bridge_mode", "plain").lower()
            if bridge_mode == "plain":
                bridge_loss = pair_loss.mean()
            elif bridge_mode == "gated":
                aerial_text_sim = nn.functional.cosine_similarity(aerial_feats, text_feats, dim=-1)
                ground_text_sim = nn.functional.cosine_similarity(ground_feats.detach(), text_feats, dim=-1)
                gate_tau = max(float(self.args.bridge_gate_tau), 1e-6)
                gate_min = float(self.args.bridge_gate_min)
                gate_min = min(max(gate_min, 0.0), 1.0)
                gate = gate_min + (1.0 - gate_min) * torch.sigmoid((ground_text_sim - aerial_text_sim) / gate_tau)
                gate = gate.detach()
                bridge_loss = (gate * pair_loss).mean()
                ret["bridge_gate"] = gate.mean()
            else:
                raise ValueError(f"Unsupported bridge_mode: {self.args.bridge_mode}")
            ret["bridge_loss"] = bridge_loss * self.args.bridge_loss_weight

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
