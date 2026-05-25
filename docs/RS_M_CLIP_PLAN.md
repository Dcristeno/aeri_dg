# RS-M-CLIP Internal Trial

## Decision

RS-M-CLIP is an internal trial and is not included in the formal direction-two fusion pool.

Unlike RemoteCLIP/GeoRSCLIP, RS-M-CLIP uses an OpenCLIP HuggingFace model with a multilingual text encoder. It is not directly compatible with the current `cda+fta+bridge` finetuning path, which assumes the OpenAI CLIP BPE tokenizer and OpenAI-style text transformer. The zero-shot score-cache result is extremely weak under the current AERI-PEDES protocol:

```text
R1=0.195
R5=0.765
R10=1.319
mAP=0.379
mINP=0.179
```

This does not prove that RS-M-CLIP cannot work after adapter finetuning, but it does mean it should not enter the current score-level fusion pool. A fair RS-M-CLIP experiment would require a separate adapter-finetuning path, not the current direction-two pipeline.

## Build Score Cache

Install OpenCLIP if needed:

```bash
pip install -U open_clip_torch
```

Use the HuggingFace mirror if the server cannot reach HuggingFace:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Build the score cache:

```bash
python tools/build_openclip_score_cache.py \
  --model_name 'hf-hub:joaodaniel/RS-M-CLIP' \
  --root_dir /home/wuyong/datasets \
  --dataset_name AERI-PEDES \
  --split test \
  --batch_size 256 \
  --num_workers 8 \
  --device cuda \
  --output logs/merge/rs_m_clip/rs_m_clip_score_cache.pth
```

The script also writes:

```text
logs/merge/rs_m_clip/rs_m_clip_score_cache.json
```

## Fuse With Meta Experts

Use the OpenAI3 fixed meta cache for the R1-oriented main line:

```text
logs/merge/openai3_meta_prior_082_008_010/fixed_score_cache.pth
```

Create a two-expert pool:

```bash
cat > docs/merge_pool_openai3_fixed_meta_rs_m_clip.json <<'JSON'
{
  "experts": [
    {
      "name": "openai3_meta_fixed",
      "score_cache": "logs/merge/openai3_meta_prior_082_008_010/fixed_score_cache.pth",
      "pretrain_choice": "score_cache",
      "loss_names": "score_cache"
    },
    {
      "name": "rs_m_clip",
      "score_cache": "logs/merge/rs_m_clip/rs_m_clip_score_cache.pth",
      "pretrain_choice": "hf-hub:joaodaniel/RS-M-CLIP",
      "loss_names": "score_cache"
    }
  ]
}
JSON
```

Recommended weight scan:

```bash
python tools/gar_em_score_fusion.py \
  --expert_config docs/merge_pool_openai3_fixed_meta_rs_m_clip.json \
  --root_dir /home/wuyong/datasets \
  --output_dir logs/merge/openai3_fixed_meta_plus_rs_m_clip_w002 \
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
```

## Fuse With RemoteCLIP

If RS-M-CLIP has useful signal, combine it with the current best RemoteCLIP result:

```text
OpenAI3 fixed meta + RemoteCLIP + RS-M-CLIP
```

Start with:

```text
0.90,0.08,0.02
0.88,0.08,0.04
0.86,0.08,0.06
```

RemoteCLIP remains the strongest R1 auxiliary expert so far, while RS-M-CLIP should be tested as a small text-semantics or multilingual pretraining complement.
