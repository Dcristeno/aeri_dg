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
        
        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)
            self._init_id_pid_lookup()

        if 'mlm' in args.loss_names:
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

        if 'fta' in self.current_task:
            self.num_query = getattr(args, "fta_num_query", 4)
            self.query = nn.Parameter(torch.randn(self.num_query, self.embed_dim))

            if not hasattr(self, "cross_attn"):
                self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                        self.embed_dim // 64,
                                                        batch_first=True)
                self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                           layers=args.cmt_depth,
                                                           heads=self.embed_dim // 64)
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

                nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
                nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlp_logsigma2 = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 2),
                nn.ReLU(),
                nn.Linear(self.embed_dim * 2, self.embed_dim)
            )

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def _init_id_pid_lookup(self):
        pid_classes = getattr(self.args, "id_pid_classes", None)
        if not pid_classes:
            return

        pid_classes = [int(pid) for pid in pid_classes]
        lookup = torch.full((max(pid_classes) + 1,), -1, dtype=torch.long)
        lookup[torch.tensor(pid_classes, dtype=torch.long)] = torch.arange(len(pid_classes), dtype=torch.long)
        try:
            self.register_buffer("id_pid_lookup", lookup, persistent=False)
        except TypeError:
            self.register_buffer("id_pid_lookup", lookup)

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

    def build_fta_queries(self, batch_size):
        return self.query.unsqueeze(0).expand(batch_size, -1, -1)

    def compute_fuzzy_membership(self, query_features, target_features):
        query_features = query_features.half()
        target_features = target_features.half()

        query_norm = F.normalize(query_features, dim=-1)
        target_norm = F.normalize(target_features, dim=-1)

        log_sigma2 = self.mlp_logsigma2(target_norm)
        sigma2 = torch.exp(log_sigma2).clamp(min=1e-6)

        query_exp = query_norm.unsqueeze(2)
        target_exp = target_norm.unsqueeze(0).unsqueeze(0)
        relation = query_exp * target_exp

        sigma2_exp = sigma2.unsqueeze(0).unsqueeze(0)
        mu_dim = torch.exp(-((1.0 - relation) ** 2) / (2 * sigma2_exp ** 2))
        mu_mean = mu_dim.mean(dim=-1)
        return mu_mean.transpose(1, 2).contiguous()

    def compute_fta_loss(self, image_feats, text_feats, image_cls_feats, text_cls_feats, pids, logit_scale):
        batch_size = image_cls_feats.shape[0]
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            query = self.build_fta_queries(batch_size)
            query_visual = self.cross_former(query.half(), image_feats, image_feats)
            query_text = self.cross_former(query.half(), text_feats, text_feats)

            mu_t2v = self.compute_fuzzy_membership(query_text, text_cls_feats)
            mu_v2t = self.compute_fuzzy_membership(query_visual, image_cls_feats)

        query_text = F.normalize(query_text, dim=-1)
        query_visual = F.normalize(query_visual, dim=-1)

        t2v_simi = torch.einsum('bkd,Bkd->bBk', query_text, query_visual)
        v2t_simi = torch.einsum('bkd,Bkd->bBk', query_visual, query_text)
        mu_and = mu_t2v * mu_v2t
        s_t2v = (t2v_simi * mu_and).mean(dim=-1)
        s_v2t = (v2t_simi * mu_and).mean(dim=-1)

        fta_loss = 0.5 * objectives.compute_fa_loss(s_t2v, s_v2t, pids, logit_scale)
        return fta_loss * self.args.fta_loss_weight

    def encode_image(self, image):
        image_feats = self.base_model.encode_image(image)
        return image_feats[:, 0, :].float()
        # return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, image, text, ori_text, ground_image=None):
        images = image
        caption_ids = text
        ori_caption_ids = ori_text
        mix_ids = torch.cat([caption_ids,ori_caption_ids],dim=0)
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            image_feats, ground_image_feats, text_feats = self.base_model(images, ground_image, mix_ids)
        image_feats, fu_img_feats = image_feats.chunk(2,dim=0)
        ground_feats = None
        if ground_image_feats is not None:
            ground_feats, _ = ground_image_feats.chunk(2, dim=0)
        text_feats, fu_txt_feats = text_feats.chunk(2,dim=0)
        if ground_feats is not None:
            return image_feats.float(), text_feats.float(), fu_img_feats.float(), fu_txt_feats.float(), ground_feats.float()
        return image_feats.float(), text_feats.float(), fu_img_feats.float(),fu_txt_feats.float()
        
        ret = {}
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        
        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})


        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            id_labels = batch['pids'].long() + int(getattr(self.args, "id_label_offset", 0))
            image_logits = self.classifier(i_feats.float())
            text_logits = self.classifier(t_feats.float())
            if id_labels.numel() > 0 and (id_labels.min().item() < 0 or id_labels.max().item() >= image_logits.shape[1]):
                raise ValueError(
                    f"id label out of classifier range: min_label={id_labels.min().item()}, max_label={id_labels.max().item()}, "
                    f"num_classes={image_logits.shape[1]}"
                )
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


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
