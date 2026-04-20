# Tag Schema

PromptingNemo models output tagged transcriptions where meta-information is embedded directly in the token stream. This document defines the canonical tag format.

## Tag Format

Tags appear as uppercase tokens in the transcription output. There are two categories:

### Inline Tags (wrap spans of text)
```
ENTITY_<TYPE> ... END
```
Example: `ENTITY_PERSON_NAME John Smith END said hello`

### Sentence-Level Tags (appended after the transcription)
```
<transcription> AGE_<range> GENDER_<value> EMOTION_<value> INTENT_<value> DIALECT_<value>
```
Example: `hello how are you AGE_30_45 GENDER_FEMALE EMOTION_HAPPY INTENT_GREETING DIALECT_NORTH`

## Canonical Tag Names

### AGE
| Tag | Description |
|-----|-------------|
| `AGE_0_18` | Child/teenager |
| `AGE_18_30` | Young adult |
| `AGE_30_45` | Adult |
| `AGE_45_60` | Middle-aged |
| `AGE_60+` | Senior |

**Note:** Legacy data may use `AGE_60PLUS` — normalize to `AGE_60+`. The alternate scheme `AGE_14_25`, `AGE_26_40`, `AGE_>41` appears in some Chinese datasets.

### GENDER
| Tag | Description |
|-----|-------------|
| `GENDER_MALE` | Male speaker |
| `GENDER_FEMALE` | Female speaker |
| `GENDER_OTHER` | Other/unknown |

**Note:** Legacy data may use `GER_MALE`/`GER_FEMALE` — always normalize to `GENDER_*`.

### EMOTION
| Tag | Description |
|-----|-------------|
| `EMOTION_HAPPY` | Happy/positive |
| `EMOTION_NEUTRAL` | Neutral |
| `EMOTION_SAD` | Sad |
| `EMOTION_ANGRY` | Angry |
| `EMOTION_FEAR` | Fearful |
| `EMOTION_SURPRISE` | Surprised |
| `EMOTION_DISGUST` | Disgusted |

**Note:** Legacy data may use abbreviated forms `EMOTION_HAP`, `EMOTION_NEU`, `EMOTION_ANG` — always normalize to the full form.

### INTENT
Intent tags vary by domain. Common ones include:
- `INTENT_INFORM`, `INTENT_QUESTION`, `INTENT_COMMAND`, `INTENT_GREETING`
- `INTENT_CONVERSATION`, `INTENT_KEYWORDS_SPOTTING`, `INTENT_LANGUAGE_SPECIFIC`

### ENTITY (inline)
Common entity types:
- `ENTITY_PERSON_NAME`, `ENTITY_ORGANIZATION`, `ENTITY_CITY`, `ENTITY_LOCATION`
- `ENTITY_DATE`, `ENTITY_TIME`, `ENTITY_PERCENTAGE`, `ENTITY_AMOUNT`
- Closed with `END` token

### DIALECT
Regional dialect markers (language-specific):
- Hindi: `DIALECT_BIHAR`, `DIALECT_RAJASTHAN`, `DIALECT_MADHYA_PRADESH`, etc.
- Chinese: `DIALECT_NORTH`, `DIALECT_SOUTH`

## Normalization

Use `promptingnemo.data.normalize.normalize_text()` to canonicalize tags:
```python
from promptingnemo.data.normalize import normalize_text

text = "hello GER_FEMALE EMOTION_HAP AGE_60PLUS"
clean = normalize_text(text)
# → "hello GENDER_FEMALE EMOTION_HAPPY AGE_60+"
```
