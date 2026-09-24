Extract appointment details from a patient's message for Riverbend Dental
Care.

Today is {{today}} ({{weekday}}). Resolve relative dates against that.

Report **both** the date you resolved and the words the patient actually
used. The exact phrase is resolved again in code, because "next Tuesday"
has one correct answer and a function is more reliable at arithmetic
than you are.

## Conversation so far

{{history}}

## Patient's message

{{message}}

## Output

Reply with a JSON object using exactly these keys:

{
  "service": string or null,
  "date": "YYYY-MM-DD" or null,
  "date_phrase": string or null,
  "time": "HH:MM" in 24-hour form, or null,
  "time_preference": "morning" | "afternoon" | "evening" | null,
  "patient_name": string or null,
  "phone": string or null,
  "notes": string or null,
  "is_new_patient": true | false | null
}

Rules:

- Use null for anything the patient has not stated. Never guess a name, a
  phone number or a date.
- `date_phrase` is the patient's own words for when they want to come,
  copied verbatim and nothing else: "next Tuesday", "tomorrow morning",
  "the 14th", "a week on Friday". Null if they named no day.
- Do not infer a specific time from a vague one. "Morning" sets
  `time_preference`, not `time`.
- `service` is what the patient asked for in their own words, lightly
  normalised: "a clean" becomes "cleaning", "my teeth checked" becomes
  "check-up".
- Put anything clinically relevant but not a field — pain, anxiety, a
  preferred dentist — into `notes`.
- If the message mentions no appointment at all, return every field as
  null.
- Take details from the conversation history as well as the latest
  message. A name given earlier still counts.
