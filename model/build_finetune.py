import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def encode_image(self, image):
        image_feats = self.base_model.encode_image(image)
        return image_feats[:, 0, :].float()

    def encode_text(self, text):
        text_feats = self.base_model.encode_text(text.long())
        return text_feats[torch.arange(text_feats.shape[0]), text.argmax(dim=-1)].float()

    def _depth_proxy_confidence(self, image):
        image = image.float()
        mean = image.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = image.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        rgb = (image * std + mean).clamp(0.0, 1.0)
        gray = (
            0.299 * rgb[:, 0:1]
            + 0.587 * rgb[:, 1:2]
            + 0.114 * rgb[:, 2:3]
        )

        grad_x = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        grad_y = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        structure = grad_x + grad_y

        height, width = gray.shape[-2:]
        y = torch.linspace(0.0, 1.0, height, device=image.device, dtype=image.dtype)
        x = torch.linspace(0.0, 1.0, width, device=image.device, dtype=image.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        center_prior = torch.exp(-(((xx - 0.5) / 0.28) ** 2 + ((yy - 0.5) / 0.42) ** 2) / 2.0)
        near_prior = yy
        mask = (center_prior * (0.5 + near_prior)).view(1, 1, height, width)

        eps = 1e-6
        focus_structure = (structure * mask).sum(dim=(1, 2, 3)) / (mask.sum() + eps)
        global_structure = structure.mean(dim=(1, 2, 3))
        focus_ratio = focus_structure / (global_structure + eps)

        masked_gray = (gray * mask).sum(dim=(1, 2, 3)) / (mask.sum() + eps)
        masked_contrast = (((gray - masked_gray.view(-1, 1, 1, 1)) ** 2) * mask).sum(dim=(1, 2, 3))
        masked_contrast = torch.sqrt(masked_contrast / (mask.sum() + eps) + eps)

        vertical_mass = (structure * yy.view(1, 1, height, width)).sum(dim=(1, 2, 3))
        vertical_mass = vertical_mass / (structure.sum(dim=(1, 2, 3)) + eps)

        raw_score = focus_ratio + masked_contrast + vertical_mass
        score = (raw_score - raw_score.mean()) / (raw_score.std(unbiased=False) + eps)
        return torch.sigmoid(score)

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

        aerial_feats = image_feats[:, 0, :].float()
        ground_feats = ground_image_feats[:, 0, :].float()
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
            elif bridge_mode == "depth_gated":
                aerial_depth_conf = self._depth_proxy_confidence(images)
                ground_depth_conf = self._depth_proxy_confidence(ground_images)
                gate_tau = max(float(self.args.depth_gate_tau), 1e-6)
                gate_min = float(self.args.depth_gate_min)
                gate_min = min(max(gate_min, 0.0), 1.0)
                ground_trust = torch.sigmoid((ground_depth_conf - aerial_depth_conf) / gate_tau)
                depth_agreement = (1.0 - (ground_depth_conf - aerial_depth_conf).abs()).clamp(0.0, 1.0)
                gate = gate_min + (1.0 - gate_min) * (0.7 * ground_trust + 0.3 * depth_agreement)
                gate = gate.detach()
                bridge_loss = (gate * pair_loss).mean()
                ret["bridge_gate"] = gate.mean()
                ret["bridge_depth_aerial"] = aerial_depth_conf.mean()
                ret["bridge_depth_ground"] = ground_depth_conf.mean()
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
