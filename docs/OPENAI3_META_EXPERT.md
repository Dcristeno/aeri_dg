# OpenAI3 Meta Expert

## Goal

Cache the OpenAI backbone expert group as one score-level meta expert:

```text
OpenAI3 = OpenAI ViT-B/16 + OpenAI ViT-B/32 + OpenAI ViT-L/14
```

This avoids reloading the three OpenAI models every time RemoteCLIP, GeoRSCLIP, or another different-pretraining expert is tested.

## Build The Cache

Create the three-expert config:

```bash
cat > docs/merge_pool_openai3.json <<'JSON'
{
  "experts": [
    {
      "name": "openai_vit_b16_full",
      "config_file": "logs/AERI-PEDES/20260521_081034_aeri_cda_fta_bridge_pair_k2_w2p0/configs.yaml",
      "checkpoint": "logs/AERI-PEDES/20260521_081034_aeri_cda_fta_bridge_pair_k2_w2p0/best0.pth",
      "pretrain_choice": "ViT-B/16",
      "loss_names": "cda+fta+bridge"
    },
    {
      "name": "openai_vit_b32_full",
      "config_file": "logs/AERI-PEDES/20260522_164259_aeri_full_vit_b32_garem/configs.yaml",
      "checkpoint": "logs/AERI-PEDES/20260522_164259_aeri_full_vit_b32_garem/best0.pth",
      "pretrain_choice": "ViT-B/32",
      "loss_names": "cda+fta+bridge"
    },
    {
      "name": "openai_vit_l14_full",
      "config_file": "logs/AERI-PEDES/20260522_222323_aeri_full_vit_l14_garem_b16/configs.yaml",
      "checkpoint": "logs/AERI-PEDES/20260522_222323_aeri_full_vit_l14_garem_b16/best0.pth",
      "pretrain_choice": "ViT-L/14",
      "loss_names": "cda+fta+bridge"
    }
  ]
}
JSON
```

Save the prior-adaptive OpenAI3 cache:

```bash
python tools/gar_em_score_fusion.py \
  --expert_config docs/merge_pool_openai3.json \
  --root_dir /home/wuyong/datasets \
  --output_dir logs/merge/openai3_meta_prior_082_008_010 \
  --topk 10 \
  --fixed_weights 0.82,0.08,0.10 \
  --prior_weights 0.82,0.08,0.10 \
  --prior_strength 1.0 \
  --adaptive_temperature 0.5 \
  --save_fusion_cache gar_em_prior_adaptive,fixed \
  --device cuda
```

Expected main score cache:

```text
logs/merge/openai3_meta_prior_082_008_010/gar_em_prior_adaptive_score_cache.pth
```

The fixed cache is also saved for ablation:

```text
logs/merge/openai3_meta_prior_082_008_010/fixed_score_cache.pth
```

## Use The Cache

Use `score_cache` instead of `checkpoint` and `config_file`:

```json
{
  "name": "openai3_meta_prior",
  "score_cache": "logs/merge/openai3_meta_prior_082_008_010/gar_em_prior_adaptive_score_cache.pth",
  "pretrain_choice": "score_cache",
  "loss_names": "score_cache"
}
```

Example: OpenAI3 meta + RemoteCLIP:

```bash
cat > docs/merge_pool_openai3_meta_remoteclip.json <<'JSON'
{
  "experts": [
    {
      "name": "openai3_meta_prior",
      "score_cache": "logs/merge/openai3_meta_prior_082_008_010/gar_em_prior_adaptive_score_cache.pth",
      "pretrain_choice": "score_cache",
      "loss_names": "score_cache"
    },
    {
      "name": "remoteclip_vit_b32_full",
      "config_file": "logs/AERI-PEDES/20260525_141417_aeri_remoteclip_vit_b32_full_r1/configs.yaml",
      "checkpoint": "logs/AERI-PEDES/20260525_141417_aeri_remoteclip_vit_b32_full_r1/best0.pth",
      "pretrain_choice": "/home/wuyong/pretrained/RemoteCLIP-ViT-B-32.pt",
      "loss_names": "cda+fta+bridge",
      "stride_size": 32
    }
  ]
}
JSON
```

Run the two-expert fusion:

```bash
python tools/gar_em_score_fusion.py \
  --expert_config docs/merge_pool_openai3_meta_remoteclip.json \
  --root_dir /home/wuyong/datasets \
  --output_dir logs/merge/openai3_meta_plus_remoteclip_w008 \
  --topk 10 \
  --fixed_weights 0.92,0.08 \
  --prior_weights 0.92,0.08 \
  --prior_strength 1.0 \
  --adaptive_temperature 0.5 \
  --device cuda
```

This is equivalent to the expanded four-expert fixed weights:

```text
0.7544,0.0736,0.092,0.08
```
