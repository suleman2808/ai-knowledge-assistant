Extract appointment details from a patient's message for Riverbend Dental
Care.

Today is {{today}} ({{weekday}}). Resolve all relative dates against that.
"Tomorrow" is the next calendar day. "Next Tuesday" means the Tuesday of
the following week, not the Tuesday of this week. "This Friday" means the
Friday of the current week.

## Conversation so far

{{history}}

## Patient's message

{{message}}

## Output

Reply with a JSON object using exactly these keys:

{
  "service": string or null,
  "date": "YYYY-MM-DD" or null,
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
