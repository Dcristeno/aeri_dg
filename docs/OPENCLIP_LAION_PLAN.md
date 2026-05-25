# OpenCLIP LAION ViT-B/32 Internal Trial

## Decision

OpenCLIP LAION ViT-B/32 is an internal control, not a formal direction-two expert.

```text
model_name=ViT-B-32
pretrained=laion2b_s34b_b79k
```

It is not remote-sensing-specific. It is a general large-scale web image-text pretraining source and was considered as a control:

```text
Does generic OpenCLIP pretraining provide complementary evidence beyond OpenAI CLIP and RemoteCLIP?
```

The zero-shot score-cache result is weak under the current protocol:

```text
R1=5.064
R5=11.724
R10=16.365
mAP=4.247
mINP=1.278
```

A fair full-recipe comparison would require exporting/adapting OpenCLIP weights into the current training stack and then running 60 epochs. That is outside the current direction-two scope. The formal result remains RemoteCLIP.

The model can be loaded directly through OpenCLIP:

```python
open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
```

## Build Score Cache

Install OpenCLIP if needed:

```bash
pip install -U open_clip_torch
```

Build the score cache:

```bash
python tools/build_openclip_score_cache.py \
  --model_name 'ViT-B-32' \
  --pretrained 'laion2b_s34b_b79k' \
  --root_dir /home/wuyong/datasets \
  --dataset_name AERI-PEDES \
  --split test \
  --batch_size 256 \
  --num_workers 8 \
  --device cuda \
  --output logs/merge/openclip_laion_vit_b32/openclip_laion_vit_b32_score_cache.pth
```

The script also writes:

```text
logs/merge/openclip_laion_vit_b32/openclip_laion_vit_b32_score_cache.json
```

## First Fusion: OpenAI3 Meta + OpenCLIP LAION

Use the R1-oriented OpenAI3 fixed meta cache:

```text
logs/merge/openai3_meta_prior_082_008_010/fixed_score_cache.pth
```

Create the pool:

```bash
cat > docs/merge_pool_openai3_fixed_meta_openclip_laion.json <<'JSON'
{
  "experts": [
    {
      "name": "openai3_meta_fixed",
      "score_cache": "logs/merge/openai3_meta_prior_082_008_010/fixed_score_cache.pth",
      "pretrain_choice": "score_cache",
      "loss_names": "score_cache"
    },
    {
      "name": "openclip_laion_vit_b32",
      "score_cache": "logs/merge/openclip_laion_vit_b32/openclip_laion_vit_b32_score_cache.pth",
      "pretrain_choice": "ViT-B-32/laion2b_s34b_b79k",
      "loss_names": "score_cache"
    }
  ]
}
JSON
```

Recommended scan:

```bash
python tools/gar_em_score_fusion.py \
  --expert_config docs/merge_pool_openai3_fixed_meta_openclip_laion.json \
  --root_dir /home/wuyong/datasets \
  --output_dir logs/merge/openai3_fixed_meta_plus_openclip_laion_w002 \
  --topk 10 \
  --fixed_weights 0.98,0.02 \
  --prior_weights 0.98,0.02 \
  --prior_strength 1.0 \
  --adaptive_temperature 0.5 \
  --device cuda
```

Then try:

```text
0.96,0.04
0.94,0.06
0.92,0.08
0.90,0.10
```

## Second Fusion: Add To RemoteCLIP Result

If OpenCLIP LAION has useful signal, combine it with the current best direction-two result:

```text
OpenAI3 fixed meta + RemoteCLIP + OpenCLIP LAION
```

Start with:

```text
0.90,0.08,0.02
0.88,0.08,0.04
0.86,0.08,0.06
```

RemoteCLIP remains the strongest auxiliary expert so far:

```text
OpenAI3 fixed meta + RemoteCLIP r=0.08
R1=49.959
```

OpenCLIP LAION should be retained only if it improves R1 or provides a meaningful mAP/mINP gain without hurting R1 too much.
