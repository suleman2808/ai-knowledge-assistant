/*
 * Admin dashboard.
 *
 * Same approach as the chat page: plain JavaScript, no framework, no
 * build step, served by the same process as the API.
 *
 * The view worth the most here is a conversation's retrieval trace. A
 * grounded answer is only a claim until you can see which sections of
 * which documents produced it, and that is what the transcript shows
 * underneath every answer.
 */

const gate = document.getElementById('gate');
const app = document.getElementById('app');
const view = document.getElementById('view');
const tabs = document.getElementById('tabs');

const escapeHtml = (s) => String(s ?? '')
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;');

const when = (iso) => {
  if (!iso) return '-';
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? escapeHtml(iso)
    : d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
};

async function api(path, options) {
  const response = await fetch(path, options);
  if (response.status === 401) { showGate(); throw new Error('Signed out.'); }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

/* ------------------------------------------------------------------ */
/* Sign in                                                             */
/* ------------------------------------------------------------------ */

function showGate() { gate.hidden = false; app.hidden = true; }
function showApp() { gate.hidden = true; app.hidden = false; }

/* Signing in does NOT go through `api()`. There, a 401 means the session
   expired and "Signed out." is the right thing to say; here it means the
   password was wrong, and reporting that as "Signed out." sent someone
   hunting for a configuration problem that did not exist. */
const LOGIN_ERRORS = {
  401: 'That password was not recognised.',
  429: 'Too many attempts. Wait a minute and try again.',
  503: 'No admin password is configured on the server. Set ADMIN_PASSWORD and restart.',
};

document.getElementById('loginForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const error = document.getElementById('loginError');
  const field = document.getElementById('password');
  error.hidden = true;

  let response;
  try {
    response = await fetch('/api/admin/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password: field.value }),
    });
  } catch {
    error.textContent = 'Could not reach the server.';
    error.hidden = false;
    return;
  }

  if (response.ok) {
    field.value = '';
    showApp();
    render('overview');
    return;
  }

  error.textContent = LOGIN_ERRORS[response.status] || `Sign in failed (${response.status}).`;
  error.hidden = false;
  field.select();
});

document.getElementById('signOut').addEventListener('click', async () => {
  await fetch('/api/admin/logout', { method: 'POST' });
  showGate();
});

/* ------------------------------------------------------------------ */
/* Views                                                               */
/* ------------------------------------------------------------------ */

const table = (headers, rows) => `
  <table>
    <thead><tr>${headers.map((h) => `<th>${h}</th>`).join('')}</tr></thead>
    <tbody>${rows.join('')}</tbody>
  </table>`;

const empty = (text) => `<p class="empty">${text}</p>`;

const views = {
  async overview() {
    const d = await api('/api/analytics?days=30');
    if (!d.turns) return empty('No conversations recorded yet.');
    const inq = d.inquiries;
    const card = (n, k) => `<div class="card"><div class="n">${n}</div><div class="k">${k}</div></div>`;
    const rate = inq.answer_rate_pct === null ? '-' : `${inq.answer_rate_pct}%`;
    const median = d.latency_ms.p50 === null ? '-' : `${(d.latency_ms.p50 / 1000).toFixed(1)}s`;
    const gaps = d.knowledge_gaps.length
      ? d.knowledge_gaps.map((g) => `<div class="gap"><span>${escapeHtml(g.question)}</span>
          <span class="times">${g.times > 1 ? `asked ${g.times} times` : 'asked once'}</span></div>`).join('')
      : '<p class="empty">None - everything asked was answerable.</p>';

    return `
      <div class="cards">
        ${card(d.turns, 'messages')}
        ${card(d.conversations, 'conversations')}
        ${card(rate, 'questions answered')}
        ${card(d.bookings_made, 'bookings')}
        ${card(d.escalations, 'escalated')}
        ${card(median, 'median reply')}
      </div>
      <div class="panel">
        <h2>Knowledge gaps</h2>
        <p class="note">Questions the assistant declined, things customers want to
        know that the documents do not cover.</p>
        ${gaps}
      </div>`;
  },

  async conversations() {
    const rows = await api('/api/admin/conversations?limit=100');
    if (!rows.length) return empty('No conversations yet.');
    return table(
      ['Started', 'Session', 'Turns', 'Handled by', 'Outcome'],
      rows.map((r) => `
        <tr class="clickable" data-session="${escapeHtml(r.session_id)}">
          <td>${when(r.started_at)}</td>
          <td><code>${escapeHtml(r.session_id)}</code></td>
          <td>${r.turns}</td>
          <td>${r.intents.map((i) => `<span class="pill grey">${escapeHtml(i)}</span>`).join(' ')}</td>
          <td>
            ${r.bookings ? '<span class="pill">booked</span> ' : ''}
            ${r.escalations ? '<span class="pill warn">escalated</span> ' : ''}
            ${r.declined ? `<span class="pill grey">${r.declined} declined</span> ` : ''}
            ${r.failures ? '<span class="pill bad">failure</span>' : ''}
          </td>
        </tr>`),
    );
  },

  async bookings() {
    const rows = await api('/api/admin/bookings');
    if (!rows.length) return empty('No bookings yet.');
    return table(
      ['Appointment', 'Customer', 'Phone', 'Service', 'Calendar'],
      rows.map((r) => `
        <tr>
          <td>${when(r.starts_at)}</td>
          <td>${escapeHtml(r.patient_name) || '-'}</td>
          <td>${escapeHtml(r.phone) || '-'}</td>
          <td>${escapeHtml(r.service) || '-'}</td>
          <td><span class="pill ${r.calendar_backend === 'google' ? '' : 'grey'}">${escapeHtml(r.calendar_backend)}</span></td>
        </tr>`),
    );
  },

  async complaints() {
    const rows = await api('/api/admin/complaints');
    if (!rows.length) return empty('No complaints logged.');
    const tone = { critical: 'bad', high: 'bad', medium: 'warn', low: 'grey' };
    return table(
      ['Received', 'Reference', 'Category', 'Severity', 'Summary', 'Escalated'],
      rows.map((r) => `
        <tr>
          <td>${when(r.received_at)}</td>
          <td><code>${escapeHtml(r.reference)}</code></td>
          <td>${escapeHtml(r.category)}</td>
          <td><span class="pill ${tone[r.severity] || 'grey'}">${escapeHtml(r.severity)}</span></td>
          <td>${escapeHtml(r.summary)}</td>
          <td>${r.escalated
            ? `<span class="pill warn" title="${escapeHtml(r.escalation_reasons.join('; '))}">yes</span>`
            : '<span class="pill grey">no</span>'}</td>
        </tr>`),
    );
  },

  async customers() {
    const rows = await api('/api/admin/customers');
    if (!rows.length) return empty('No returning customers yet.');
    return table(
      ['Name', 'Phone', 'Bookings', 'First seen', 'Last seen'],
      rows.map((r) => `
        <tr>
          <td>${escapeHtml(r.name) || '-'}</td>
          <td><code>${escapeHtml(r.phone)}</code></td>
          <td>${r.visit_count}</td>
          <td>${when(r.first_seen)}</td>
          <td>${when(r.last_seen)}</td>
        </tr>`),
    );
  },
};

/** One conversation, every turn, with the sections behind each answer. */
async function showConversation(sessionId) {
  const turns = await api(`/api/admin/conversations/${encodeURIComponent(sessionId)}`);

  const blocks = turns.map((t) => {
    const badges = [
      `<span class="pill grey">${escapeHtml(t.intent)}</span>`,
      t.grounded === true ? '<span class="pill">from documents</span>' : '',
      t.grounded === false ? '<span class="pill warn">declined</span>' : '',
      t.success ? '' : '<span class="pill bad">failed</span>',
      `<span class="pill grey">${t.latency_ms} ms</span>`,
    ].join(' ');

    const rewritten = t.search_query && t.search_query !== t.message
      ? `<p class="note">Searched as: <em>${escapeHtml(t.search_query)}</em></p>`
      : '';

    // Three distinct cases, and conflating them produces a transcript
    // that contradicts its own badges: an answer marked "from documents"
    // sitting above "the documents did not cover this".
    let trace = '';
    if ((t.sources || []).length) {
      const items = t.sources.map((s) => `<li><span class="crumb">${escapeHtml(s.breadcrumb)}</span>
        - ${escapeHtml(s.source)} (${s.score})</li>`).join('');
      trace = `<div class="trace"><h4>Answered from</h4>${rewritten}<ol>${items}</ol></div>`;
    } else if (t.grounded === false) {
      trace = `<div class="trace"><h4>Answered from</h4>${rewritten}
        <p class="note">Nothing - the documents did not cover this, so it declined.</p></div>`;
    } else if (t.grounded === true) {
      // Answered from documents, but recorded before the trace was kept.
      trace = `<div class="trace"><h4>Answered from</h4>
        <p class="note">Not recorded - this conversation pre-dates the
        retrieval trace.</p></div>`;
    }

    return `<div class="turn">
        <div class="q"><strong>Customer:</strong> ${escapeHtml(t.message)}</div>
        <div class="a">${t.answer
            ? escapeHtml(t.answer)
            : '<span class="muted">(answer not recorded)</span>'}<div class="meta">${badges}</div></div>
        ${trace}
      </div>`;
  });

  view.innerHTML = `
    <button class="back" id="back">← All conversations</button>
    <p class="muted">Session <code>${escapeHtml(sessionId)}</code> · ${turns.length} turn(s)</p>
    ${blocks.join('')}`;
  document.getElementById('back').addEventListener('click', () => render('conversations'));
}

/* ------------------------------------------------------------------ */
/* Wiring                                                              */
/* ------------------------------------------------------------------ */

async function render(name) {
  for (const button of tabs.querySelectorAll('button')) {
    button.classList.toggle('active', button.dataset.view === name);
  }
  view.innerHTML = '<p class="empty">Loading…</p>';
  try {
    view.innerHTML = await views[name]();
  } catch (e) {
    view.innerHTML = `<p class="empty">${escapeHtml(e.message)}</p>`;
    return;
  }
  for (const row of view.querySelectorAll('tr[data-session]')) {
    row.addEventListener('click', () => showConversation(row.dataset.session));
  }
}

tabs.addEventListener('click', (event) => {
  if (event.target.dataset.view) render(event.target.dataset.view);
});

(async function start() {
  try {
    const session = await (await fetch('/api/admin/session')).json();
    if (!session.configured) {
      document.body.innerHTML =
        '<p class="empty">The dashboard is not configured on this server. '
        + 'Set ADMIN_PASSWORD and restart.</p>';
      return;
    }
    if (session.signed_in) { showApp(); render('overview'); } else { showGate(); }
  } catch {
    showGate();
  }
})();
