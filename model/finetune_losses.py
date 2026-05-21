import torch
import torch.nn.functional as F

from . import objectives


def compute_cda_loss(aerial_feats, ground_feats, text_feats, pids, logit_scale):
    if ground_feats is None:
        raise ValueError("cda loss requires ground image features, but the current batch does not provide them.")
    return objectives.compute_selective_align_loss(
        aerial_feats,
        ground_feats,
        text_feats,
        pids,
        logit_scale,
    )


def compute_bridge_losses(args, aerial_feats, ground_feats, text_feats, pids, logit_scale):
    if ground_feats is None:
        raise ValueError("bridge loss requires ground image features, but the current batch does not provide them.")

    bridge_terms = objectives.compute_ground_to_aerial_bridge_terms(
        aerial_feats,
        ground_feats,
        text_feats,
        pids,
        logit_scale,
        distill_temp=args.bridge_distill_temp,
    )
    weighted_pair_loss = bridge_terms["pair_loss"] * args.bridge_pair_weight * args.bridge_loss_weight
    weighted_distill_loss = bridge_terms["distill_loss"] * args.bridge_distill_weight * args.bridge_loss_weight
    return {
        "bridge_pair_loss": weighted_pair_loss,
        "bridge_distill_loss": weighted_distill_loss,
        "bridge_loss": weighted_pair_loss + weighted_distill_loss,
    }


def compute_fta_loss(model, image_tokens, text_tokens, image_feats, text_feats, pids, logit_scale):
    with torch.autocast(dtype=torch.float16, device_type="cuda"):
        query_v, query_t = model.build_fta_queries(image_feats, text_feats)
        q_v = model.cross_former(query_v.half(), image_tokens, image_tokens)
        q_t = model.cross_former(query_t.half(), text_tokens, text_tokens)

    with torch.autocast(dtype=torch.float16, device_type="cuda"):
        mu_t2v = model.compute_fuzzy_membership(q_t, text_feats)
        mu_v2t = model.compute_fuzzy_membership(q_v, image_feats)

    q_t = F.normalize(q_t, dim=-1)
    q_v = F.normalize(q_v, dim=-1)

    t2v_simi = torch.einsum("bkd,Bkd->bBk", q_t, q_v)
    v2t_simi = torch.einsum("bkd,Bkd->bBk", q_v, q_t)
    mu_and = mu_t2v * mu_v2t
    s_t2v = (t2v_simi * mu_and).mean(dim=-1)
    s_v2t = (v2t_simi * mu_and).mean(dim=-1)

    return 0.5 * objectives.compute_fa_loss(s_t2v, s_v2t, pids, logit_scale)
