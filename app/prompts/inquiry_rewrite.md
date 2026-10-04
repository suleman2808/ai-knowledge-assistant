Rewrite a patient's latest message as a standalone question, so it can be
searched without the conversation around it.

## Conversation so far

{{history}}

## Latest message

{{message}}

## Rules

- Resolve every reference to earlier turns. "Do I need to fast for it?"
  after a question about the lipid profile becomes "Do I need to fast
  for a lipid profile?". "How long does it take?" after a question
  about a urine culture becomes "How long does a urine culture take to
  report?".
- Use the laboratory's own term where the patient used a colloquial one:
  "sugar test" becomes "fasting blood glucose", "thyroid test" becomes
  "thyroid profile", "blood count" becomes "full blood count". Keep the
  patient's own words as well, since either may be what the documents
  use.
- If the latest message already stands alone, return it unchanged.
- Do not answer the question. Do not add facts, prices or names that are
  not in the conversation.
- One question, one line, no quotation marks, no preamble.
