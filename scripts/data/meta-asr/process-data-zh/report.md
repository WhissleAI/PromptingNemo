# Voices-in-the-Wild Chinese Subset — Data Report

**Dataset:** zhifeixie/Voices-in-the-Wild-2M (Chinese-only filter)
**Date:** 2026-05-24
**Pipeline:** Download → GPU annotation (AGE, GENDER, EMOTION) → Gemini annotation (INTENT, ENTITY)

---

## Overview

| Metric | Train | Valid | Total |
|---|---|---|---|
| Samples | 143,902 | 12,226 | 156,128 |
| Duration (hours) | 139.5h | 11.9h | 151.4h |
| Lang | MANDARIN | MANDARIN | — |
| Source | voices_in_the_wild | voices_in_the_wild | — |

Validation split: 7.8% (target 5%, applied before annotation)

---

## Duration Statistics

| Metric | Train | Valid |
|---|---|---|
| Mean | 3.49s | 3.49s |
| Median | 3.28s | 3.25s |
| Std Dev | 1.86s | 1.84s |
| Min | 0.46s | 0.50s |
| Max | 20.0s | 19.88s |

### Duration Distribution

| Bucket | Train | Train % | Valid | Valid % |
|---|---|---|---|---|
| 0–1s | 4,763 | 3.3% | 392 | 3.2% |
| 1–2s | 30,349 | 21.1% | 2,504 | 20.5% |
| 2–3s | 27,319 | 19.0% | 2,434 | 19.9% |
| 3–5s | 53,681 | 37.3% | 4,560 | 37.3% |
| 5–10s | 27,348 | 19.0% | 2,301 | 18.8% |
| 10–20s | 441 | 0.3% | 35 | 0.3% |
| 20–30s | 1 | <0.1% | 0 | 0% |

Bulk of samples (76%) fall in the 1–5s range. Very few long utterances (>10s).

---

## Text Statistics (Chinese characters, excluding tags)

| Metric | Train | Valid |
|---|---|---|
| Mean length | 14.7 chars | 14.7 chars |
| Median length | 13 chars | 13 chars |
| Min length | 2 chars | 3 chars |
| Max length | 125 chars | 106 chars |

---

## Annotation Tags

### Tag Coverage

| Tag Type | Train | Valid |
|---|---|---|
| AGE | 143,902 (100%) | 12,226 (100%) |
| GENDER | 143,902 (100%) | 12,226 (100%) |
| EMOTION | 143,902 (100%) | 12,226 (100%) |
| INTENT | 143,902 (100%) | 12,226 (100%) |
| ENTITY | 73,907 (51.4%) | 5,574 (45.6%) |

100% coverage for AGE, GENDER, EMOTION, and INTENT. ~51% of train and ~46% of valid samples contain at least one named entity. Sentences without entities are typically short utterances or fragments lacking identifiable named entities.

### AGE Distribution

| Tag | Train | Train % | Valid | Valid % |
|---|---|---|---|---|
| AGE_0_18 | 2,704 | 1.9% | 216 | 1.8% |
| AGE_18_30 | 35,201 | 24.5% | 2,907 | 23.8% |
| AGE_30_45 | 76,025 | 52.8% | 6,517 | 53.3% |
| AGE_45_60 | 25,295 | 17.6% | 2,201 | 18.0% |
| AGE_60PLUS | 4,677 | 3.3% | 385 | 3.1% |

The 30–45 age group dominates (53%), consistent with a dataset sourced from news broadcasts, podcasts, and web media.

### GENDER Distribution

| Tag | Train | Train % | Valid | Valid % |
|---|---|---|---|---|
| GENDER_MALE | 73,188 | 50.9% | 6,225 | 50.9% |
| GENDER_FEMALE | 66,627 | 46.3% | 5,636 | 46.1% |
| GENDER_OTHER | 4,087 | 2.8% | 365 | 3.0% |

Well-balanced male/female split (~51/46). GENDER_OTHER (2.8%) likely represents ambiguous or heavily distorted audio.

### EMOTION Distribution

| Tag | Train | Train % | Valid | Valid % |
|---|---|---|---|---|
| EMOTION_SAD | 70,946 | 49.3% | 6,072 | 49.7% |
| EMOTION_NEUTRAL | 46,791 | 32.5% | 3,920 | 32.1% |
| EMOTION_HAPPY | 20,825 | 14.5% | 1,782 | 14.6% |
| EMOTION_ANGRY | 5,340 | 3.7% | 452 | 3.7% |

**Note:** EMOTION_SAD at 49% is disproportionately high. This is a known artifact of the HuBERT emotion classifier (`superb/hubert-large-superb-er`) on degraded audio. The Voices-in-the-Wild dataset contains acoustic conditions like echo, far-field, noise, distortion, and obstructed recordings, which the model tends to misclassify as "sad." The AGE and GENDER predictions are more robust to these conditions.

Only 4 of 7 canonical emotions are present (DISGUST, FEAR, SURPRISE absent), likely because the model's confidence thresholds map most Chinese speech to these four categories.

### INTENT Distribution

| Tag | Train | Train % | Valid | Valid % |
|---|---|---|---|---|
| INTENT_STATEMENT | 52,068 | 36.2% | 2,641 | 21.6% |
| INTENT_INFORM | 36,936 | 25.7% | 3,253 | 26.6% |
| INTENT_OPINION | 15,170 | 10.5% | 1,277 | 10.4% |
| INTENT_DESCRIBE | 10,577 | 7.4% | 1,830 | 15.0% |
| INTENT_QUESTION | 9,646 | 6.7% | 750 | 6.1% |
| INTENT_EXPLAIN | 7,920 | 5.5% | 1,149 | 9.4% |
| INTENT_COMMAND | 5,534 | 3.8% | 617 | 5.0% |
| INTENT_EXCLAIM | 2,832 | 2.0% | 485 | 4.0% |
| INTENT_REQUEST | 2,134 | 1.5% | 224 | 1.8% |

**Note:** Train and valid intent distributions differ because they were annotated in separate Gemini runs. The train set was re-annotated with the combined ENTITY+INTENT prompt (which shifted some classifications), while the valid set retained its original INTENT labels from an earlier run and only had entities added. The 9 primary intent categories cover >99.5% of samples; a small number of non-canonical intents (COMPLAINT, ANNOUNCEMENT, RESPONSE, etc.) appear in train due to Gemini occasionally using entity types as intents.

### ENTITY Distribution (Top 15)

| Entity Type | Train | Valid |
|---|---|---|
| ENTITY_ORGANIZATION | 20,920 | 1,587 |
| ENTITY_PERSON_NAME | 18,837 | 1,665 |
| ENTITY_DATE | 9,362 | 798 |
| ENTITY_NUMBER | 7,210 | 594 |
| ENTITY_LOCATION | 6,442 | 479 |
| ENTITY_COUNTRY | 5,948 | 485 |
| ENTITY_EVENT | 5,769 | 420 |
| ENTITY_PRODUCT | 4,652 | 283 |
| ENTITY_CITY | 4,361 | 387 |
| ENTITY_DURATION | 4,073 | 332 |
| ENTITY_CATEGORY | 2,419 | — |
| ENTITY_PRICE | 2,121 | 164 |
| ENTITY_MEASUREMENT | 1,611 | 112 |
| ENTITY_PROJECT_NAME | 1,593 | 93 |
| ENTITY_OCCUPATION | 1,591 | — |

ORGANIZATION and PERSON_NAME are the most common entity types, consistent with news/media content. The 130+ entity type vocabulary from the annotation prompt provides fine-grained NER coverage.

---

## Annotation Pipeline

| Stage | Status | Duration | Tool |
|---|---|---|---|
| 1. Download + base manifest | Complete | ~2h | HuggingFace datasets (streaming) |
| 2. AGE, GENDER, EMOTION | Complete | ~30m | wav2vec2-large-robust-6-ft-age-gender, hubert-large-superb-er (T4 GPU) |
| 3. INTENT + ENTITY | Complete | ~9h (train) + ~35m (valid) | Gemini 2.5 Flash (5 workers, batch size 20) |

---

## Sample Entries (fully annotated)

```json
{"audio_filepath": "/mnt/nfs/data/vitw_zh/audio/vitw_zh_recording_noise_0000045.wav", "text": "ENTITY_ORGANIZATION 海外网 END ENTITY_DATE 六月三十日 END 报道，据 ENTITY_ORGANIZATION 美国有线电视新闻网 END 报道。 AGE_30_45 GENDER_FEMALE EMOTION_SAD INTENT_INFORM", "duration": 4.16, "lang": "MANDARIN", "source": "voices_in_the_wild"}
{"audio_filepath": "/mnt/nfs/data/vitw_zh/audio/vitw_zh_far_field_noise_0067252.wav", "text": "她也经常在网络上分享生活趣事。 AGE_18_30 GENDER_FEMALE EMOTION_NEUTRAL INTENT_STATEMENT", "duration": 4.116, "lang": "MANDARIN", "source": "voices_in_the_wild"}
{"audio_filepath": "/mnt/nfs/data/vitw_zh/audio/vitw_zh_distortion_0036251.wav", "text": "我问你。 AGE_45_60 GENDER_MALE EMOTION_NEUTRAL INTENT_COMMAND", "duration": 1.36, "lang": "MANDARIN", "source": "voices_in_the_wild"}
```

---

## Files

| File | Entries | Size |
|---|---|---|
| train.json | 143,902 | 39 MB |
| valid.json | 12,226 | 3.3 MB |
| dataset_info.json | — | <1 KB |
| Audio directory | 244,526 WAVs | ~170 GB (on NFS) |

### Storage Locations

- **NFS:** `/mnt/nfs/data/vitw_zh/` (audio + manifests)
- **Local:** `/home/sridhar/Documents/whissle/data/vitw_zh/` (manifests only)
- **K8s YAMLs:** `/home/sridhar/Documents/whissle/k8s/vitw-zh-*.yaml`
