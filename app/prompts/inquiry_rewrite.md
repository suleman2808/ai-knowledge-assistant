Rewrite a patient's latest message as a standalone question, so it can be
searched without the conversation around it.

## Conversation so far

{{history}}

## Latest message

{{message}}

## Rules

- Resolve every reference to earlier turns. "Is that for one surface?"
  after a question about filling prices becomes "Is the price of a
  filling for one surface?". "How long can I spread it over?" after a
  question about payment plans becomes "How long can a payment plan be
  spread over?".
- Use the clinical term where the patient used a colloquial one:
  "having a tooth out" becomes "tooth extraction", "a clean" becomes
  "a scale and polish cleaning". Keep the patient's own words as well,
  since either may be what the documents use.
- If the latest message already stands alone, return it unchanged.
- Do not answer the question. Do not add facts, prices or names that are
  not in the conversation.
- One question, one line, no quotation marks, no preamble.
