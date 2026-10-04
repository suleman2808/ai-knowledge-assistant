/*
 * Chat UI.
 *
 * Plain JavaScript, no framework and no CDN. The page is served by the
 * same FastAPI process as the API, so there is one thing to run and one
 * origin to trust — and nothing to rebuild when the backend changes.
 */

const transcript = document.getElementById('transcript');
const form = document.getElementById('composer');
const input = document.getElementById('input');
const sendButton = document.getElementById('send');
const suggestions = document.getElementById('suggestions');
const rail = document.getElementById('rail');
const railScrim = document.getElementById('railScrim');

let sessionId = '';
let busy = false;

/* ------------------------------------------------------------------ */
/* Rendering                                                           */
/* ------------------------------------------------------------------ */

/**
 * Escape HTML. Called on every piece of text before it reaches innerHTML.
 *
 * Model output is untrusted: it is derived from documents and from
 * whatever a patient typed. Escaping first, then adding our own markup,
 * means no input can inject an element.
 */
function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

/**
 * Render the small subset of markdown the model actually emits: bold,
 * italics, inline code, bullet lists and paragraphs.
 *
 * Written rather than imported. A markdown library is ~40 KB for four
 * features, and every one of them here runs on already-escaped text, so
 * the usual sanitisation question does not arise.
 */
function renderMarkdown(text) {
  const inline = (s) => escapeHtml(s)
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');

  const blocks = [];
  let list = null;

  for (const rawLine of String(text).split('\n')) {
    const line = rawLine.trimEnd();
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);

    if (bullet) {
      list = list || [];
      list.push(`<li>${inline(bullet[1])}</li>`);
      continue;
    }
    if (list) {
      blocks.push(`<ul>${list.join('')}</ul>`);
      list = null;
    }
    if (line.trim()) blocks.push(`<p>${inline(line)}</p>`);
  }
  if (list) blocks.push(`<ul>${list.join('')}</ul>`);

  return blocks.join('') || `<p>${inline(text)}</p>`;
}

function addMessage(who, html) {
  const wrapper = document.createElement('div');
  wrapper.className = `msg from-${who}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = html;
  wrapper.appendChild(bubble);
  transcript.appendChild(wrapper);
  transcript.scrollTop = transcript.scrollHeight;
  return bubble;
}

/**
 * Reveal streamed text at a readable pace.
 *
 * The tokens are real — the server streams them as the model writes —
 * but Groq finishes a short answer in about a fifth of a second, which
 * arrives as a wall of text indistinguishable from no streaming at all.
 * So the characters are buffered and released on animation frames.
 *
 * The rate is proportional to what is waiting, which keeps two
 * properties that a fixed delay would not: a long answer never crawls,
 * and the reveal always finishes promptly after the last token rather
 * than running on after the model has stopped.
 */
function createTypewriter(element, onSettled) {
  let full = '';
  let shown = 0;
  let frame = null;
  let finishing = false;

  function paint() {
    frame = null;
    const remaining = full.length - shown;
    if (remaining > 0) {
      // Pacing is for someone watching. A hidden tab gets the text
      // immediately: nobody is reading it, and animation frames do not
      // run there, so pacing it would mean never finishing at all.
      shown = document.hidden
        ? full.length
        // A proportion of the backlog rather than a fixed rate, so the
        // reveal takes about a second whether the answer is two lines or
        // twenty, and never crawls behind a long one. The minimum keeps
        // the last few characters from taking a frame each.
        : shown + Math.max(2, Math.ceil(remaining / 25));
      element.innerHTML = renderMarkdown(full.slice(0, shown));
      transcript.scrollTop = transcript.scrollHeight;
    }
    if (shown < full.length) {
      schedule();
    } else if (finishing) {
      onSettled();
    }
  }

  /* Animation frames stop in a hidden tab, so a timer takes over there.
     Without this a reader who switches away mid-answer comes back to a
     half-written reply and a disabled composer. */
  function schedule() {
    if (frame !== null) return;
    frame = document.hidden
      ? setTimeout(paint, 0)
      : requestAnimationFrame(paint);
  }

  return {
    push(text) { full += text; schedule(); },
    reset() { full = ''; shown = 0; element.innerHTML = PROGRESS; },
    /** Stop pacing and hand over, once what has arrived has been shown. */
    settle() {
      finishing = true;
      if (shown >= full.length) onSettled(); else schedule();
    },
    /** Abandon the reveal — the caller is taking the element over. */
    stop() {
      if (frame !== null) {
        cancelAnimationFrame(frame);
        clearTimeout(frame);
      }
      frame = null;
    },
  };
}

const PROGRESS = `
  <div class="stages"><span class="stage">
    <span class="spin"></span><span id="stageText">Working out what you need</span>
  </span></div>`;

const STAGE_LABELS = {
  router: 'Working out what you need',
  inquiry: 'Searching the laboratory’s documents',
  booking: 'Checking the appointment diary',
  complaint: 'Recording your complaint',
  other: 'Thinking',
  finalise: 'Writing a reply',
};

const INTENT_LABELS = {
  booking: 'Appointments',
  inquiry: 'Information',
  complaint: 'Complaint',
  other: 'General',
};

/** Footer showing how the answer was produced: routing, grounding, sources. */
function renderMeta(bubble, data) {
  const meta = document.createElement('div');
  meta.className = 'meta';

  if (INTENT_LABELS[data.intent]) {
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = INTENT_LABELS[data.intent];
    meta.appendChild(badge);
  }

  // Grounding is the promise this project makes, so it is stated on
  // every answer rather than hidden in a developer console.
  if (data.grounded === true) {
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = 'From laboratory documents';
    meta.appendChild(badge);
  } else if (data.grounded === false) {
    const badge = document.createElement('span');
    badge.className = 'badge warn';
    badge.textContent = 'Not in our documents';
    meta.appendChild(badge);
  }

  if (data.latency_ms) {
    const timing = document.createElement('span');
    timing.textContent = `${(data.latency_ms / 1000).toFixed(1)}s`;
    meta.appendChild(timing);
  }

  if (data.sources && data.sources.length) {
    const details = document.createElement('details');
    details.className = 'sources';
    const summary = document.createElement('summary');
    summary.textContent = `${data.sources.length} source${data.sources.length > 1 ? 's' : ''}`;
    details.appendChild(summary);

    const list = document.createElement('ol');
    for (const source of data.sources) {
      const item = document.createElement('li');
      const crumb = document.createElement('span');
      crumb.className = 'crumb';
      crumb.textContent = source.breadcrumb;
      item.appendChild(crumb);
      item.append(` - ${source.source} (${source.score})`);
      list.appendChild(item);
    }
    details.appendChild(list);
    meta.appendChild(details);
  }

  bubble.appendChild(meta);
}

/* ------------------------------------------------------------------ */
/* Sending                                                             */
/* ------------------------------------------------------------------ */

function setBusy(state) {
  busy = state;
  sendButton.disabled = state;
  input.disabled = state;
  if (!state) input.focus();
}

async function send(message) {
  if (busy || !message.trim()) return;

  addMessage('patient', renderMarkdown(message));
  input.value = '';
  input.style.height = 'auto';
  setBusy(true);

  const pending = addMessage('assistant', PROGRESS);
  let finished = null;
  const typing = createTypewriter(pending, () => {
    if (finished) finished();
  });

  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId }),
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let completed = false;

    while (!completed) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line; the last fragment may
      // be incomplete, so it stays in the buffer.
      const frames = buffer.split('\n\n');
      buffer = frames.pop();

      for (const frame of frames) {
        const line = frame.split('\n').find((l) => l.startsWith('data: '));
        if (!line) continue;
        const event = JSON.parse(line.slice(6));

        if (event.event === 'node') {
          const label = STAGE_LABELS[event.node];
          const stageText = document.getElementById('stageText');
          if (label && stageText) stageText.textContent = label;
        } else if (event.event === 'token') {
          // The first token replaces the progress spinner: once there are
          // words to read, a label saying what we are doing is noise.
          typing.push(event.text);
        } else if (event.event === 'reset') {
          // The attempt was abandoned and is being retried. Drop what was
          // shown rather than letting a second attempt append to a first.
          typing.reset();
        } else if (event.event === 'error') {
          throw new Error(event.detail);
        } else if (event.event === 'done') {
          sessionId = event.session_id || sessionId;
          // Wait for the reveal to catch up, then replace it with the
          // authoritative answer: a turn can gain a handoff note after
          // the model stopped, and an agent can discard the model's text
          // entirely.
          finished = () => {
            typing.stop();
            pending.innerHTML = renderMarkdown(event.answer);
            renderMeta(pending, event);
            setBusy(false);
            transcript.scrollTop = transcript.scrollHeight;
          };
          typing.settle();
          completed = true;
        }
      }
    }

    if (!completed) throw new Error('The connection ended before an answer arrived.');
  } catch (error) {
    typing.stop();
    pending.innerHTML = '';
    const box = document.createElement('div');
    box.className = 'error';
    box.textContent = error.message ||
      'Something went wrong. Please try again, or call (503) 555-0142.';
    pending.appendChild(box);
    setBusy(false);
    transcript.scrollTop = transcript.scrollHeight;
  }
}

/* ------------------------------------------------------------------ */
/* Wiring                                                              */
/* ------------------------------------------------------------------ */

form.addEventListener('submit', (event) => {
  event.preventDefault();
  send(input.value);
});

// Enter sends, Shift+Enter makes a new line.
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    send(input.value);
  }
});

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
});

suggestions.addEventListener('click', (event) => {
  if (event.target.tagName !== 'BUTTON') return;
  closeRail();
  send(event.target.textContent);
});

/**
 * Fill the "under the hood" panel from /api/health.
 *
 * Everything shown is already public on that endpoint, and a visitor
 * reading it learns what the thing is made of without opening the repo:
 * which models, how many chunks, where embeddings run, and — the honest
 * part — which calendar a booking would land in.
 */
const CALENDAR_LABELS = {
  google: 'Google Calendar',
  in_memory: 'In-memory (demo)',
};

function fact(term, value, extra) {
  const row = document.createElement('div');
  const dt = document.createElement('dt');
  dt.textContent = term;
  const dd = document.createElement('dd');
  if (extra) dd.appendChild(extra);
  if (value !== null && value !== undefined) {
    const span = document.createElement('span');
    span.textContent = value;
    dd.appendChild(span);
  }
  row.append(dt, dd);
  return row;
}

function monospace(text) {
  const code = document.createElement('code');
  code.textContent = text;
  return code;
}

async function checkHealth() {
  const facts = document.getElementById('facts');
  const dot = document.createElement('span');
  dot.className = 'dot';

  let health;
  try {
    health = await (await fetch('/api/health')).json();
  } catch {
    dot.classList.add('down');
    facts.replaceChildren(fact('Status', 'Offline', dot));
    return;
  }

  dot.classList.add(health.status === 'ok' ? 'ok' : 'degraded');
  const rows = [
    fact('Status', health.status === 'ok' ? 'Online' : 'Limited service', dot),
  ];

  const kb = health.checks?.knowledge_base;
  if (kb?.chunks) {
    rows.push(fact('Knowledge base', `${kb.chunks} chunks · ${kb.documents} docs`));
  }

  const llm = health.checks?.llm;
  if (llm?.model) {
    rows.push(fact('Agents', null, monospace(llm.model.split('/').pop())));
    rows.push(fact('Router', null, monospace(llm.router_model.split('/').pop())));
  }

  const embeddings = health.checks?.embeddings;
  if (embeddings?.backend) {
    rows.push(fact('Embeddings', `Local · ${embeddings.backend}`));
  }

  const calendar = health.checks?.calendar;
  if (calendar?.backend) {
    rows.push(fact('Calendar', CALENDAR_LABELS[calendar.backend] || calendar.backend));
  }

  facts.replaceChildren(...rows);
}

/* The rail is a drawer on a narrow screen and a fixture on a wide one. */
function closeRail() {
  rail.classList.remove('open');
  railScrim.hidden = true;
}

document.getElementById('railToggle').addEventListener('click', () => {
  const open = rail.classList.toggle('open');
  railScrim.hidden = !open;
});
railScrim.addEventListener('click', closeRail);

document.getElementById('reset').addEventListener('click', () => {
  if (busy) return;
  sessionId = '';
  transcript.replaceChildren();
  greet();
  input.focus();
});

function greet() {
  addMessage('assistant', renderMarkdown(GREETING));
}

const GREETING =
  'Hello, I’m the assistant for Riverbend Diagnostics. I can answer ' +
  'questions about our services, prices, hours and policies, book you an ' +
  'appointment, or pass on a complaint. What can I help with?';

greet();
checkHealth();
input.focus();
