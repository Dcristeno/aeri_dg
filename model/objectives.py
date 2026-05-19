import torch
import torch.nn.functional as F


def compute_sdm(image_features, text_features, pid, logit_scale, epsilon=1e-8):
    """Similarity Distribution Matching for a batch of matched identities."""
    batch_size = image_features.shape[0]
    pid = pid.reshape((batch_size, 1))
    labels = (pid - pid.t() == 0).float()
    labels = labels / (labels.sum(dim=1, keepdim=True) + epsilon)

    image_norm = F.normalize(image_features, dim=1)
    text_norm = F.normalize(text_features, dim=1)

    logits_i2t = logit_scale * image_norm @ text_norm.t()
    logits_t2i = logit_scale * text_norm @ image_norm.t()

    i2t_loss = F.softmax(logits_i2t, dim=1) * (F.log_softmax(logits_i2t, dim=1) - torch.log(labels + epsilon))
    t2i_loss = F.softmax(logits_t2i, dim=1) * (F.log_softmax(logits_t2i, dim=1) - torch.log(labels + epsilon))
    return i2t_loss.sum(dim=1).mean() + t2i_loss.sum(dim=1).mean()


def compute_sdm_per_sample(image_features, text_features, pid, logit_scale, epsilon=1e-8):
    """Per-sample SDM, used by selective CDA-style routing."""
    batch_size = image_features.shape[0]
    pid = pid.reshape((batch_size, 1))
    labels = (pid - pid.t() == 0).float()
    labels = labels / (labels.sum(dim=1, keepdim=True) + epsilon)

    image_norm = F.normalize(image_features, dim=1)
    text_norm = F.normalize(text_features, dim=1)

    logits_i2t = logit_scale * image_norm @ text_norm.t()
    logits_t2i = logit_scale * text_norm @ image_norm.t()

    i2t_loss = F.softmax(logits_i2t, dim=1) * (F.log_softmax(logits_i2t, dim=1) - torch.log(labels + epsilon))
    t2i_loss = F.softmax(logits_t2i, dim=1) * (F.log_softmax(logits_t2i, dim=1) - torch.log(labels + epsilon))
    return i2t_loss.sum(dim=1) + t2i_loss.sum(dim=1)


def compute_selective_bridge_sdm_loss(aerial_features, ground_features, text_features, pid, logit_scale):
    """CDA essence: prefer direct aerial-text SDM when it is reliable, otherwise lean on ground bridge."""
    if ground_features is None:
        raise ValueError("selective_bridge_sdm_loss requires ground image features, but got None.")

    aerial_norm = F.normalize(aerial_features, dim=-1)
    ground_norm = F.normalize(ground_features, dim=-1)
    text_norm = F.normalize(text_features, dim=-1)

    direct_score = torch.sum(text_norm * aerial_norm, dim=-1)
    bridge_score = torch.sum(text_norm * ground_norm, dim=-1)
    direct_weight = torch.sigmoid(direct_score - bridge_score)

    direct_loss = compute_sdm_per_sample(aerial_norm, text_norm, pid, logit_scale)
    bridge_loss = (
        compute_sdm_per_sample(ground_norm, text_norm, pid, logit_scale)
        + compute_sdm_per_sample(aerial_norm, ground_norm.detach(), pid, logit_scale)
    )
    return (direct_weight * direct_loss + (1.0 - direct_weight) * bridge_loss).mean()


def compute_similarity_distribution_loss(sim_t2i, sim_i2t, pid, logit_scale, epsilon=1e-8):
    """SDM over a custom text-to-image and image-to-text similarity matrix."""
    batch_size = sim_t2i.shape[0]
    pid = pid.reshape((batch_size, 1))
    labels = (pid - pid.t() == 0).float()
    labels = labels / (labels.sum(dim=1, keepdim=True) + epsilon)

    logits_t2i = logit_scale * sim_t2i
    logits_i2t = logit_scale * sim_i2t

    t2i_loss = F.softmax(logits_t2i, dim=1) * (F.log_softmax(logits_t2i, dim=1) - torch.log(labels + epsilon))
    i2t_loss = F.softmax(logits_i2t, dim=1) * (F.log_softmax(logits_i2t, dim=1) - torch.log(labels + epsilon))
    return t2i_loss.sum(dim=1).mean() + i2t_loss.sum(dim=1).mean()


def compute_aeri_base_sdm_terms(aerial_features, ground_features, text_features, pid, logit_scale):
    """Plain AERI baseline: aerial-text + ground-text SDM."""
    if ground_features is None:
        raise ValueError("AERI base loss requires ground image features, but got None.")

    return {
        "base_aerial_text": compute_sdm(aerial_features, text_features, pid, logit_scale),
        "base_ground_text": compute_sdm(ground_features, text_features, pid, logit_scale),
    }
