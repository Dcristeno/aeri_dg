# Merge Direction Two Summary

## Final Decision

Use **RemoteCLIP** as the only formal different-pretraining expert.

Direction two should not become a broad sweep over many pretrained models. The paper needs a clear and reproducible demonstration that a task-relevant pretraining source can complement the OpenAI CLIP backbone expert pool. RemoteCLIP gives that result cleanly.

## Formal Result

Stage 1 builds the OpenAI backbone meta expert:

```text
OpenAI3 = ViT-B/16 + ViT-B/32 + ViT-L/14
fixed weights = 0.82,0.08,0.10
```

Stage 2 adds the different-pretraining expert:

```text
OpenAI3 fixed meta + RemoteCLIP ViT-B/32
fixed weights = 0.92,0.08
```

Best direction-two result:

| method | R1 | R5 | R10 | RSum | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI3 GAR-EM prior | 49.813 | 66.650 | 74.955 | 191.418 | 48.307 | 35.797 |
| OpenAI3 fixed meta + RemoteCLIP r=0.08 | **49.959** | 67.204 | 75.444 | 192.607 | 48.383 | 35.844 |

Note: this remains the best direction-two result. It is no longer the global best after adding the direction-three HardNeg expert. See `docs/MERGE_DIRECTION_THREE_HARDNEG_RESULTS.md` for the current global best:

```text
OpenAI3 + RemoteCLIP + HardNeg
GAR-EM prior-adaptive, HardNeg h=0.05
R1 = 50.513
```

Gain:

```text
+0.146 R1
+0.076 mAP
+0.047 mINP
```

## Why RemoteCLIP

RemoteCLIP is trained for remote-sensing vision-language alignment. It provides aerial layout, land-cover, building, road, and overhead-view priors that are naturally aligned with the AERI-PEDES ground-aerial retrieval setting.

Its single-expert R1 is only `34.294`, so it should not be a main expert. However, with a small prior (`r=0.08`) it provides complementary ranking evidence and improves the OpenAI3 meta expert.

## Internal Trials Not Used In Main Direction Two

| model | status | reason |
| --- | --- | --- |
| GeoRSCLIP | internal trial only | Trained successfully, but adding it to RemoteCLIP lowers R1; it mainly improves mAP/mINP. |
| RS-M-CLIP | internal trial only | Zero-shot score cache is extremely weak; proper use would require adapter finetuning. |
| OpenCLIP LAION ViT-B/32 | internal control only | Zero-shot score cache is weak; a fair full-recipe test would require weight export/adaptation and 60-epoch finetuning. |
| GeoCLIP | not used | Geolocation/GPS objective is not a direct text-image retrieval expert. |

## Reporting Guidance

Use RemoteCLIP in the main paper as the direction-two instantiation:

```text
Different pretraining: OpenAI3 + RemoteCLIP
```

If space allows, mention the internal trials briefly:

```text
Additional pretrained sources were explored internally. GeoRSCLIP helped tail-ranking metrics but reduced R1, while RS-M-CLIP and generic OpenCLIP LAION were not directly suitable under the current training/evaluation protocol. Therefore, the formal different-pretraining expert pool uses RemoteCLIP only.
```

This avoids overcomplicating the method section and keeps the contribution focused on reliable complementarity rather than model zoo exploration.
