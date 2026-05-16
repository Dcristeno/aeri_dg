from model import objectives
from .clip_model import ResidualAttentionBlock, ResidualCrossAttentionBlock, Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        if 'id' in self.current_task:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)
            self._init_id_pid_lookup()
        if 'proto' in self.current_task:
            self._load_aerial_prototypes()
        if 'track' in self.current_task:
            self._init_track_memory()

        if 'fta' in args.loss_names:  
            self.fta_query_mode = getattr(args, "fta_query_mode", "static").lower()
            if self.fta_query_mode not in {"static", "conditioned"}:
                raise ValueError(f"Unsupported fta_query_mode: {self.fta_query_mode}")

            self.num_query = getattr(args, "fta_num_query", 4)
            self.query = nn.Parameter(torch.randn(self.num_query, self.embed_dim))
            
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

            self.mlp_logsigma2 = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim*2),
                nn.ReLU(),
                nn.Linear(self.embed_dim*2, self.embed_dim)
            )

            if self.fta_query_mode == "conditioned":
                self.visual_query_condition = nn.Sequential(
                    LayerNorm(self.embed_dim),
                    nn.Linear(self.embed_dim, self.embed_dim),
                    QuickGELU(),
                    nn.Linear(self.embed_dim, self.num_query * self.embed_dim),
                )
                self.text_query_condition = nn.Sequential(
                    LayerNorm(self.embed_dim),
                    nn.Linear(self.embed_dim, self.embed_dim),
                    QuickGELU(),
                    nn.Linear(self.embed_dim, self.num_query * self.embed_dim),
                )
                nn.init.normal_(self.visual_query_condition[1].weight, std=fc_std)
                nn.init.normal_(self.visual_query_condition[3].weight, std=proj_std)
                nn.init.normal_(self.text_query_condition[1].weight, std=fc_std)
                nn.init.normal_(self.text_query_condition[3].weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def _register_optional_buffer(self, name, tensor):
        try:
            self.register_buffer(name, tensor, persistent=False)
        except TypeError:
            self.register_buffer(name, tensor)

    def _init_id_pid_lookup(self):
        pid_classes = getattr(self.args, "id_pid_classes", None)
        if not pid_classes:
            return

        pid_classes = [int(pid) for pid in pid_classes]
        lookup = torch.full((max(pid_classes) + 1,), -1, dtype=torch.long)
        lookup[torch.tensor(pid_classes, dtype=torch.long)] = torch.arange(len(pid_classes), dtype=torch.long)
        self._register_optional_buffer("id_pid_lookup", lookup)

    def _map_id_labels(self, labels):
        labels = labels.long()
        if not hasattr(self, "id_pid_lookup"):
            return labels

        if labels.numel() == 0:
            return labels
        if labels.min().item() < 0 or labels.max().item() >= self.id_pid_lookup.shape[0]:
            raise ValueError("Encountered pid outside the id classifier lookup range.")

        mapped_labels = self.id_pid_lookup[labels]
        if (mapped_labels < 0).any().item():
            missing = torch.unique(labels[mapped_labels < 0]).detach().cpu().tolist()
            raise ValueError(f"Encountered pids missing from id classifier lookup: {missing}")
        return mapped_labels

    def _load_aerial_prototypes(self):
        prototype_path = getattr(self.args, "aerial_prototype_path", "")
        if not prototype_path:
            raise ValueError("proto loss requires --aerial_prototype_path to point to a saved prototype file.")

        proto_data = torch.load(prototype_path, map_location="cpu")
        if not isinstance(proto_data, dict) or "prototypes" not in proto_data:
            raise ValueError(f"Invalid aerial prototype file: {prototype_path}")

        prototypes = proto_data["prototypes"].float()
        if prototypes.ndim != 2:
            raise ValueError(f"Aerial prototypes must be a 2D tensor, but got shape {tuple(prototypes.shape)}")
        if prototypes.shape[1] != self.embed_dim:
            raise ValueError(
                f"Aerial prototype dim mismatch: expected {self.embed_dim}, got {prototypes.shape[1]}"
            )

        valid_mask = proto_data.get("valid_mask")
        if valid_mask is None:
            valid_mask = torch.ones(prototypes.shape[0], dtype=torch.bool)
        else:
            valid_mask = valid_mask.bool()

        self._register_optional_buffer("aerial_prototypes", prototypes)
        self._register_optional_buffer("aerial_prototype_valid_mask", valid_mask)

    def _init_track_memory(self):
        num_pids = int(getattr(self.args, "track_memory_num_pids", 0))
        if num_pids <= 0:
            raise ValueError("track loss requires a positive --track_memory_num_pids value.")
        self._register_optional_buffer("track_memory", torch.zeros(num_pids, self.embed_dim, dtype=torch.float32))
        self._register_optional_buffer("track_memory_valid", torch.zeros(num_pids, dtype=torch.bool))

    def compute_track_memory_losses(self, image_cls_feats, text_cls_feats, pid_indices):
        if not hasattr(self, "track_memory") or self.track_memory is None:
            raise ValueError("track loss requires initialized track memory buffers.")

        if pid_indices.max().item() >= self.track_memory.shape[0]:
            raise ValueError("Encountered pid outside the range covered by the track memory.")

        valid_mask = self.track_memory_valid[pid_indices]
        zero = image_cls_feats.sum() * 0.0
        if not valid_mask.any().item():
            return zero, zero

        target_memory = self.track_memory[pid_indices[valid_mask]].float()
        image_loss = objectives.compute_feature_alignment_loss(
            image_cls_feats[valid_mask],
            target_memory,
        )
        text_loss = objectives.compute_feature_alignment_loss(
            text_cls_feats[valid_mask],
            target_memory,
        )
        return image_loss, text_loss

    @torch.no_grad()
    def update_track_memory(self, pid_indices, image_cls_feats):
        if not hasattr(self, "track_memory") or self.track_memory is None:
            return

        pid_indices = pid_indices.detach().long().reshape(-1)
        image_cls_feats = F.normalize(image_cls_feats.detach().float(), dim=-1)
        momentum = float(getattr(self.args, "track_memory_momentum", 0.8))

        unique_pids = torch.unique(pid_indices)
        for pid in unique_pids.tolist():
            batch_mask = pid_indices == pid
            batch_memory = image_cls_feats[batch_mask].mean(dim=0)
            batch_memory = F.normalize(batch_memory.unsqueeze(0), dim=-1).squeeze(0)

            if self.track_memory_valid[pid]:
                updated_memory = momentum * self.track_memory[pid] + (1.0 - momentum) * batch_memory
                self.track_memory[pid] = F.normalize(updated_memory.unsqueeze(0), dim=-1).squeeze(0)
            else:
                self.track_memory[pid] = batch_memory
                self.track_memory_valid[pid] = True
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x, None)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def build_fta_queries(self, image_cls_feats, text_cls_feats):
        batch_size = image_cls_feats.shape[0]
        base_query = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        if self.fta_query_mode != "conditioned":
            return base_query, base_query

        visual_delta = self.visual_query_condition(image_cls_feats).reshape(batch_size, self.num_query, self.embed_dim)
        text_delta = self.text_query_condition(text_cls_feats).reshape(batch_size, self.num_query, self.embed_dim)
        condition_scale = getattr(self.args, "fta_query_condition_scale", 1.0)

        visual_query = base_query + condition_scale * visual_delta
        text_query = base_query + condition_scale * text_delta
        return visual_query, text_query

    def encode_image(self, image):
        image_feats = self.base_model.encode_image(image)
        return image_feats[:, 0, :].float()

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
    
    def compute_fuzzy_membership(self, A, B):  # compute_fuzzy_membership v3

        A = A.half() # [B，K，D]
        B = B.half() # [B, D]

        Qn_norm  = F.normalize(A.half(), dim=-1)            # [B, K, D]
        Tn_norm  = F.normalize(B.half(), dim=-1) # [B, D]

        log_sigma2 = self.mlp_logsigma2(Tn_norm) 
        sigma2 = torch.exp(log_sigma2)
        sigma2 = torch.clamp(sigma2, min=1e-6)

        Qn_exp = Qn_norm.unsqueeze(2)  
        Tn_exp = Tn_norm.unsqueeze(0).unsqueeze(0)  
        r_dim = Qn_exp * Tn_exp 

        sigma2_exp = sigma2.unsqueeze(0).unsqueeze(0)
        mu_dim = torch.exp(- ((1.0 - r_dim) ** 2) / (2 * sigma2_exp ** 2))  # [B, K, B, D]
        mu_mean = mu_dim.mean(dim=-1)  # [Bq, K, Bt]
        membership = mu_mean.transpose(1, 2).contiguous()  # [B, B, K]

        return membership
    
    def entropy_maximize_loss_from_mu(self, member, dim=-1, eps=1e-8):
        # mu: [Bq, Bt, K]
        member = member.clamp(min=0.0)
        p = member / (member.sum(dim=dim, keepdim=True) + eps)  # p is prob over Bt
        entropy = - (p * torch.log(p + eps)).sum(dim=dim)  # shape [Bq, K]
        loss = - entropy.mean()  # minimize loss => maximize mean entropy
        return loss

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        ground_images = batch['ground_imgs']
        # ground_images = None
        caption_ids = batch['caption_ids']
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            image_feats, ground_image_feats, text_feats = self.base_model(images, ground_images, caption_ids)

        i_feats = image_feats[:, 0, :].float()
        g_i_feats = None
        if ground_image_feats is not None:
            g_i_feats = ground_image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale

        if 'cda' in self.current_task:
            if g_i_feats is None:
                raise ValueError("cda loss requires ground image features, but the current batch does not provide them.")
            ret.update({'cda_loss': objectives.compute_selective_align_loss(i_feats, g_i_feats, t_feats, batch['pids'], logit_scale)})
            # ret.update({'cda_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)}) # 仅使用这个就是Base

        if 'bridge' in self.current_task:
            if g_i_feats is None:
                raise ValueError("bridge loss requires ground image features, but the current batch does not provide them.")
            bridge_terms = objectives.compute_ground_to_aerial_bridge_terms(
                i_feats,
                g_i_feats,
                t_feats,
                batch['pids'],
                logit_scale,
                distill_temp=self.args.bridge_distill_temp,
            )
            weighted_pair_loss = bridge_terms["pair_loss"] * self.args.bridge_pair_weight * self.args.bridge_loss_weight
            weighted_distill_loss = bridge_terms["distill_loss"] * self.args.bridge_distill_weight * self.args.bridge_loss_weight
            ret.update({'bridge_pair_loss': weighted_pair_loss})
            ret.update({'bridge_distill_loss': weighted_distill_loss})
            ret.update({'bridge_loss': weighted_pair_loss + weighted_distill_loss})

        if 'proto' in self.current_task:
            if not hasattr(self, "aerial_prototypes") or self.aerial_prototypes is None:
                raise ValueError("proto loss requires preloaded aerial prototypes, but none were found.")
            pid_indices = batch['pids'].long()
            if pid_indices.max().item() >= self.aerial_prototypes.shape[0]:
                raise ValueError("Encountered pid outside the range covered by the aerial prototype file.")
            valid_mask = self.aerial_prototype_valid_mask[pid_indices]
            if not valid_mask.all().item():
                missing_pids = torch.unique(pid_indices[~valid_mask]).tolist()
                raise ValueError(f"Missing aerial prototypes for pids: {missing_pids}")
            prototype_feats = self.aerial_prototypes[pid_indices].float()
            ret.update({
                'prototype_loss': objectives.compute_aerial_prototype_alignment_loss(
                    i_feats,
                    prototype_feats,
                ) * self.args.aerial_prototype_weight
            })

        if 'track' in self.current_task:
            pid_indices = batch['pids'].long()
            track_image_loss, track_text_loss = self.compute_track_memory_losses(i_feats, t_feats, pid_indices)
            weighted_track_image_loss = track_image_loss * self.args.track_memory_image_weight * self.args.track_memory_loss_weight
            weighted_track_text_loss = track_text_loss * self.args.track_memory_text_weight * self.args.track_memory_loss_weight
            ret.update({'track_image_loss': weighted_track_image_loss})
            ret.update({'track_text_loss': weighted_track_text_loss})
            ret.update({'track_loss': weighted_track_image_loss + weighted_track_text_loss})
            ret.update({'_track_memory_pids': pid_indices.detach()})
            ret.update({'_track_memory_image_feats': i_feats.detach()})

        if 'fta' in self.current_task:
            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                query_v, query_t = self.build_fta_queries(i_feats, t_feats)
                Q_v = self.cross_former(query_v.half(), image_feats, image_feats) 
                Q_t = self.cross_former(query_t.half(), text_feats, text_feats)  # 

            # v2 cross-modal membership
            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                mu_t2v = self.compute_fuzzy_membership(Q_t, t_feats)
                mu_v2t = self.compute_fuzzy_membership(Q_v, i_feats)

            Q_t = F.normalize(Q_t, dim=-1) # [b,4,512]
            Q_v = F.normalize(Q_v, dim=-1) # [b,4,512]

            t2v_simi = torch.einsum('bkd,Bkd->bBk', Q_t, Q_v)
            v2t_simi = torch.einsum('bkd,Bkd->bBk', Q_v, Q_t)
            mu_and = mu_t2v * mu_v2t
            S_t2v = (t2v_simi * mu_and).mean(dim=-1)
            S_v2t = (v2t_simi * mu_and).mean(dim=-1) 

            ret.update({'fta_loss':0.5*objectives.compute_fa_loss(S_t2v, S_v2t, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'id' in self.current_task:
            id_labels = self._map_id_labels(batch['pids'])
            image_logits = self.classifier(i_feats.float())
            text_logits = self.classifier(t_feats.float())
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, id_labels)*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == id_labels).float().mean()
            text_precision = (text_pred == id_labels).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if 'att_mlm' in self.current_task:
            for att_type in ['shoes','hairstyle','genders','top','trousers','belongings']:
                mlm_ids = batch[att_type+'_mlm_ids']

                mlm_feats = self.base_model.encode_text(mlm_ids)

                x = self.cross_former(mlm_feats, image_feats, image_feats)

                x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

                scores = x.float().reshape(-1, self.args.vocab_size)
                mlm_labels = batch[att_type+'_mlm_labels'].reshape(-1)
                ret.update({att_type+'_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

                pred = scores.max(1)[1]
                mlm_label_idx = torch.nonzero(mlm_labels)
                acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
                ret.update({att_type+'_acc': acc})

        return ret


def build_finetune_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
