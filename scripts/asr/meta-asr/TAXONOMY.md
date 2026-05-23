# Whissle Meta-ASR Behavioral Taxonomy

**Version:** 2.0  
**Date:** 2026-05-22  
**Based on:** MISC 2.0 (Motivational Interviewing Skill Code, Miller et al. 2003)  
**Domain:** Tech Interview Conversations  
**Annotation Model:** Gemini 2.5 Flash  

## Overview

This taxonomy defines the tag system used in Whissle's Meta-ASR models. Tags are appended inline to transcript text and predicted by a multi-head CTC + tag classifier architecture. The system is grounded in the MISC 2.0 behavioral coding framework, adapted for technical interview conversations.

### Example

```
so the approach I would take is to use a binary search tree SPEAKER_CHANGE okay SPEAKER_CHANGE because it gives us log n lookup time and we can maintain sorted order ENTITY_CONCEPT ENTITY_CONCEPT AGE_20_30 GENDER_MALE EMOTION_NEUTRAL ROLE_INTERVIEWEE BEHAVIOR_REASON EVAL_NONE
```

---

## 1. ROLE — Speaker Role (2 classes)

Identifies the speaker's role in the conversation.

| Tag | Description |
|-----|-------------|
| `ROLE_INTERVIEWER` | The person asking questions, setting problems, evaluating, managing flow |
| `ROLE_INTERVIEWEE` | The candidate answering questions, explaining solutions, demonstrating knowledge |

---

## 2. BEHAVIOR — Behavioral Code (25 classes)

Adapted from MISC 2.0 counselor and client codes. Categorizes the communicative function of each utterance.

### Interviewer Behaviors (12)

Adapted from MISC 2.0 counselor behavioral codes.

| Tag | MISC Origin | Description | Example |
|-----|-------------|-------------|---------|
| `BEHAVIOR_QUESTION_OPEN` | QUO | Open question inviting wide elaboration | "Tell me about your experience with distributed systems" |
| `BEHAVIOR_QUESTION_CLOSED` | QUC | Yes/no or specific-answer question | "Did you use Redis for caching?" |
| `BEHAVIOR_INFORM` | GI | Giving context, explaining the problem setup | "This system needs to handle 10K requests per second" |
| `BEHAVIOR_REFLECT` | RE | Paraphrasing or summarizing candidate's answer | "So you're saying you'd use event sourcing here" |
| `BEHAVIOR_AFFIRM` | AF | Positive feedback, appreciation, encouragement | "That's a great point about eventual consistency" |
| `BEHAVIOR_DIRECT` | DI | Instructions, commands | "Now implement the LRU cache" |
| `BEHAVIOR_ADVISE` | AD | Hints, suggestions, nudging toward solution | "What if you considered using a hash map instead?" |
| `BEHAVIOR_CONFRONT` | CO | Challenging an answer, probing weakness | "But wouldn't that approach be O(n²)?" |
| `BEHAVIOR_STRUCTURE` | ST | Managing flow — greetings, transitions, time | "Let's move on to system design" |
| `BEHAVIOR_FACILITATE` | FA | Backchannels | "Mm-hmm", "Okay", "Go on" |
| `BEHAVIOR_EVALUATE` | — | Explicit assessment of answer quality | "That's correct", "Not quite what I was looking for" |
| `BEHAVIOR_WARN` | WA | Constraints, consequences to consider | "Remember this needs to handle edge cases" |

### Interviewee Behaviors (8)

Adapted from MISC 2.0 client change talk codes (DARN-CAT framework).

| Tag | MISC Origin | Description | Example |
|-----|-------------|-------------|---------|
| `BEHAVIOR_EXPLAIN` | GI adapted | Detailed technical explanation — primary answer mode | "A B-tree stores keys in sorted order in nodes with multiple children..." |
| `BEHAVIOR_REASON` | R (DARN) | Justifying an approach, giving rationale | "Because Redis gives us sub-millisecond latency" |
| `BEHAVIOR_COMMIT` | C (CAT) | Stating a design decision | "I would use a binary search tree here" |
| `BEHAVIOR_ABILITY` | A (DARN) | Expressing capability or uncertainty | "I've worked extensively with Kubernetes" / "I'm not sure about that" |
| `BEHAVIOR_QUESTION` | QU adapted | Asking for clarification | "Can I assume the input is sorted?" |
| `BEHAVIOR_ACKNOWLEDGE` | FN partial | Confirming understanding | "I see", "Got it", "Makes sense" |
| `BEHAVIOR_THINK_ALOUD` | — | Verbalizing thought process | "Let me think... if we use a trie here..." |
| `BEHAVIOR_EXPRESS` | — | Emotional expression, realization | "Oh, I see what you mean!", "That's interesting" |

### Shared Behaviors (5)

Can be used by either role.

| Tag | MISC Origin | Description | Example |
|-----|-------------|-------------|---------|
| `BEHAVIOR_FOLLOW_NEUTRAL` | FN | Non-committal, following along | "Okay", "Sure" |
| `BEHAVIOR_SUPPORT` | SU | Sympathetic, encouraging | "That's a tough problem", "Good point" |
| `BEHAVIOR_REFRAME` | RF | Offering a new perspective | "Another way to think about this..." |
| `BEHAVIOR_RAISE_CONCERN` | RC | Pointing out a potential problem | "But what about thread safety?" |
| `BEHAVIOR_FILLER` | — | Pleasantries, non-substantive | "Good morning", "Nice to meet you" |

---

## 3. EVAL — Interviewer Evaluation (6 classes)

Captures the interviewer's real-time assessment of the candidate's response. Critical for behavioral intelligence platforms (e.g., VILS.ai) and live coaching.

| Tag | Description | Signal |
|-----|-------------|--------|
| `EVAL_CORRECT` | Interviewer confirms the answer is correct | Positive hiring signal |
| `EVAL_INCORRECT` | Interviewer indicates the answer is wrong | Negative signal |
| `EVAL_PARTIAL` | Answer is partially correct, needs more | Mixed signal — candidate on right track |
| `EVAL_PROBE` | Interviewer digs deeper, tests depth | Rigor metric — measures interview quality |
| `EVAL_HINT` | Interviewer provides a clue or nudge | Independence signal — candidate needed help |
| `EVAL_SKIP` | Interviewer moves on without resolution | Implicit negative — topic abandoned |
| `EVAL_NONE` | No evaluation happening in this utterance | Default for non-evaluative segments |

### Streaming Use Cases

During real-time ASR streaming, EVAL tags enable:
- **Live coaching:** "They confirmed your answer was correct" or "They seem unsatisfied, try elaborating"
- **Post-interview scoring:** Ratio of EVAL_CORRECT to EVAL_INCORRECT across the session
- **Interview quality metrics:** How often the interviewer probes (EVAL_PROBE count)
- **Candidate independence:** How many hints were needed (EVAL_HINT count)

---

## 4. SPEAKER_CHANGE — Inline Speaker Transition

An inline tag inserted at the exact position in the text where the speaker changes within a single audio segment. Unlike end-of-line categorical tags, this appears within the spoken text.

```
so binary search would work here SPEAKER_CHANGE right and what about the worst case SPEAKER_CHANGE the worst case would be o of log n
```

### Streaming Use Cases

- Immediate speaker context switch without requiring separate diarization
- Enables role-aware behavior attribution within mixed-speaker segments
- Triggers UI updates (speaker label, avatar) in real-time transcription

---

## 5. EMOTION — Emotional State (7 classes)

| Tag | Description |
|-----|-------------|
| `EMOTION_NEUTRAL` | Calm, baseline emotional state |
| `EMOTION_HAPPY` | Positive, enthusiastic, excited |
| `EMOTION_SAD` | Disappointed, dejected |
| `EMOTION_ANGRY` | Frustrated, annoyed |
| `EMOTION_FEAR` | Nervous, anxious, stressed |
| `EMOTION_SURPRISE` | Surprised, unexpected |
| `EMOTION_DISGUST` | Disapproval, displeasure |

---

## 6. AGE — Speaker Age Group (5 classes)

| Tag | Description |
|-----|-------------|
| `AGE_20_30` | 20–30 years |
| `AGE_30_45` | 30–45 years |
| `AGE_45_60` | 45–60 years |
| `AGE_60PLUS` | 60+ years |
| `AGE_CHILD` | Under 20 |

---

## 7. GENDER — Speaker Gender (3 classes)

| Tag | Description |
|-----|-------------|
| `GENDER_MALE` | Male speaker |
| `GENDER_FEMALE` | Female speaker |
| `GENDER_OTHER` | Non-binary or indeterminate |

---

## 8. ENTITY — Inline Named Entities (8 types)

Entity tags appear inline after the spoken text, before categorical tags. They mark the presence of domain-specific entities extracted from the utterance.

| Tag | Description | Examples |
|-----|-------------|----------|
| `ENTITY_TECHNOLOGY` | Languages, frameworks, tools, platforms | Python, React, AWS, Kubernetes, Docker |
| `ENTITY_CONCEPT` | Algorithms, data structures, design patterns | binary search, linked list, microservices, CAP theorem |
| `ENTITY_SYSTEM` | Infrastructure components, architecture elements | load balancer, message queue, CDN, database replica |
| `ENTITY_METRIC` | Performance numbers, complexity, benchmarks | O(n log n), 99.9% uptime, 10ms p99 latency |
| `ENTITY_COMPANY` | Companies, organizations, products | Google, MongoDB, Redis Labs, Stripe |
| `ENTITY_ROLE` | Job titles, team roles | backend engineer, SRE, tech lead, CTO |
| `ENTITY_PROJECT` | Specific projects or features mentioned | "our payments service", "the migration project" |
| `ENTITY_ACRONYM` | Technical acronyms | API, SDK, CI/CD, DNS, SQL, gRPC, REST |

---

## 9. KEYWORD — Behavioral Markers (inline)

Keywords marking behavioral patterns in speech, useful for communication analysis.

| Category | Examples |
|----------|----------|
| Confidence markers | "definitely", "absolutely", "I'm certain", "without doubt" |
| Hedging markers | "maybe", "perhaps", "I think", "sort of", "I guess" |
| Structure markers | "first", "second", "in summary", "step by step" |
| Filler words | "um", "uh", "like", "you know", "basically" |

---

## Behavioral Metadata (non-inline, manifest-only)

These fields are stored in the JSONL manifest as metadata. They are annotated by Gemini but NOT predicted by the ASR model — they are used for downstream analytics, VILS.ai reports, and training data quality assessment.

| Field | Type | Scale | Description |
|-------|------|-------|-------------|
| `confidence_level` | int | 1–5 | How confident the speaker sounds |
| `fluency_score` | int | 1–5 | Verbal fluency — minimal fillers/hesitations |
| `technical_depth` | int | 1–5 | Depth of technical content |
| `communication_clarity` | int | 1–5 | How clearly the point is communicated |
| `interview_stage` | str | enum | `introduction` · `technical` · `behavioral` · `closing` |
| `speaker_role` | str | enum | `interviewer` · `interviewee` |
| `behavioral_keywords` | list | — | Extracted behavioral marker phrases |
| `entities` | list | — | Extracted entities with type and text |

---

## Tag Order in Text

Tags appear at the end of each transcript line in this order:

```
<spoken text> [SPEAKER_CHANGE ...] [ENTITY_* ...] [KEYWORD_* ...] AGE_* GENDER_* EMOTION_* ROLE_* BEHAVIOR_* EVAL_*
```

1. **Spoken text** — the actual transcription
2. **SPEAKER_CHANGE** — inline within text at transition points
3. **ENTITY_*** — inline entity type markers
4. **KEYWORD_*** — inline behavioral markers
5. **AGE_*** — speaker age classification
6. **GENDER_*** — speaker gender classification
7. **EMOTION_*** — speaker emotional state
8. **ROLE_*** — speaker role (interviewer/interviewee)
9. **BEHAVIOR_*** — MISC behavioral code
10. **EVAL_*** — interviewer evaluation code

---

## Tag Count Summary

| Category | Count | Predicted by ASR | Inline |
|----------|-------|-----------------|--------|
| ROLE | 2 | Yes (classifier) | No |
| BEHAVIOR | 25 | Yes (classifier) | No |
| EVAL | 7 | Yes (classifier) | No |
| EMOTION | 7 | Yes (classifier) | No |
| AGE | 5 | Yes (classifier) | No |
| GENDER | 3 | Yes (classifier) | No |
| SPEAKER_CHANGE | 1 | Yes (CTC) | Yes |
| ENTITY | 8 types | Yes (CTC) | Yes |
| KEYWORD | 4 categories | Yes (CTC) | Yes |
| **Total unique tags** | **~62** | | |

---

## Theoretical Foundation

### MISC 2.0 (Miller, Moyers, Ernst & Amrhein, 2003)

The Motivational Interviewing Skill Code is a behavioral coding system designed to capture the dynamics of counselor-client conversations. It provides:

- **Counselor codes (19):** Categorize counselor utterances by function (advise, affirm, confront, direct, facilitate, etc.)
- **Client change talk (DARN-CAT):** Desire, Ability, Reasons, Need, Commitment, Activation, Taking Steps — each with positive/negative valence
- **Global ratings:** Empathy, MI Spirit, client self-exploration scales

### Adaptation for Tech Interviews

We adapted MISC for the interviewer-interviewee dynamic:
- Counselor codes → Interviewer behaviors (question types, feedback, facilitation)
- Client change talk → Interviewee behaviors (explain, reason, commit, ability)
- Added domain-specific codes: THINK_ALOUD, EVALUATE, WARN
- Added EVAL category for real-time assessment tracking
- Added SPEAKER_CHANGE for diarization-free speaker tracking
- Added tech-domain entity types

### Reference

Miller, W. R., Moyers, T. B., Ernst, D., & Amrhein, P. (2003). *Manual for the Motivational Interviewing Skill Code (MISC) Version 2.0*. University of New Mexico Center on Alcoholism, Substance Abuse, and Addictions.

---

## License

This taxonomy is released under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). The underlying MISC 2.0 framework is publicly available academic research.

**Citation:**
```bibtex
@misc{whissle_meta_asr_taxonomy_2026,
  title={Whissle Meta-ASR Behavioral Taxonomy v2.0},
  author={Whissle AI},
  year={2026},
  note={Based on MISC 2.0 by Miller et al. (2003), adapted for tech interview domain}
}
```
