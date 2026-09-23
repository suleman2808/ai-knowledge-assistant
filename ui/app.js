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

const STAGE_LABELS = {
  router: 'Working out what you need',
  inquiry: 'Searching the clinic’s documents',
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
    badge.textContent = 'From clinic documents';
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
      item.append(` — ${source.source} (${source.score})`);
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
  suggestions.hidden = true;
  setBusy(true);

  const pending = addMessage('assistant', `
    <div class="stages"><span class="stage">
      <span class="spin"></span><span id="stageText">Working out what you need</span>
    </span></div>`);

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
    let finished = false;

    while (!finished) {
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
        } else if (event.event === 'error') {
          throw new Error(event.detail);
        } else if (event.event === 'done') {
          sessionId = event.session_id || sessionId;
          pending.innerHTML = renderMarkdown(event.answer);
          renderMeta(pending, event);
          finished = true;
        }
      }
    }

    if (!finished) throw new Error('The connection ended before an answer arrived.');
  } catch (error) {
    pending.innerHTML = '';
    const box = document.createElement('div');
    box.className = 'error';
    box.textContent = error.message ||
      'Something went wrong. Please try again, or call (503) 555-0142.';
    pending.appendChild(box);
  } finally {
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
  if (event.target.tagName === 'BUTTON') send(event.target.textContent);
});

async function checkHealth() {
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  try {
    const health = await (await fetch('/api/health')).json();
    dot.className = `dot ${health.status === 'ok' ? 'ok' : 'degraded'}`;
    text.textContent = health.status === 'ok' ? 'Online' : 'Limited service';
  } catch {
    dot.className = 'dot down';
    text.textContent = 'Offline';
  }
}

addMessage('assistant', renderMarkdown(
  'Hello — I’m the assistant for Riverbend Dental Care. I can answer ' +
  'questions about our services, prices, hours and policies, book you an ' +
  'appointment, or pass on a complaint. What can I help with?'
));

checkHealth();
input.focus();
