Assess a complaint made to Riverbend Dental Care, a dental practice.

Your assessment decides whether a human being is alerted, so judge the
severity on what could go wrong, not on how angry the wording is. A calmly
worded report of a clinical injury outranks a furious message about a
waiting room magazine.

## The complaint

{{message}}

## Output

Reply with a JSON object using exactly these keys:

{
  "summary": string,
  "category": "clinical" | "billing" | "waiting_time" | "staff_conduct" | "facilities" | "communication" | "other",
  "severity": "low" | "medium" | "high" | "critical",
  "requires_escalation": true | false,
  "patient_appears_distressed": true | false,
  "mentions_legal_action": true | false,
  "mentions_harm": true | false,
  "is_actually_a_complaint": true | false
}

### Severity

- **critical** — alleged clinical harm, a safety incident, a
  safeguarding concern, discrimination, or an explicit threat of legal
  or regulatory action.
- **high** — treatment that failed or caused avoidable pain, a
  significant billing error, repeated unanswered contact, or a patient
  saying they intend to leave the practice.
- **medium** — a single service failure with real inconvenience: a long
  wait, a cancelled appointment, a rude interaction, a billing query.
- **low** — minor dissatisfaction, a comment on comfort or facilities,
  mild irritation.

### The other fields

- `summary` is one factual sentence in neutral language. No adjectives
  the patient did not use, and no interpretation of motive.
- `requires_escalation` is true for every critical case, for any
  clinical category at high severity, for any mention of legal action,
  and for any alleged harm. When in doubt, escalate.
- `mentions_harm` covers physical injury, pain caused by treatment, and
  psychological distress attributed to the practice.
- `is_actually_a_complaint` is false when the message is praise, a
  neutral question, or a factual statement with no grievance. Do not
  manufacture a complaint out of a neutral message.
