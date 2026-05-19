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


def compute_aeri_base_sdm_terms(aerial_features, ground_features, text_features, pid, logit_scale):
    """Plain AERI baseline: aerial-text + ground-text SDM."""
    if ground_features is None:
        raise ValueError("AERI base loss requires ground image features, but got None.")

    return {
        "base_aerial_text": compute_sdm(aerial_features, text_features, pid, logit_scale),
        "base_ground_text": compute_sdm(ground_features, text_features, pid, logit_scale),
    }
