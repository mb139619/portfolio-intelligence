/*
   Portfolio Intelligence — dashboard renderer.

   The page is deliberately generic: it knows the payload *schema* (sections
   holding stats, figures, tables and notes) but nothing about portfolio
   analytics. Adding a new analysis means adding a section in build.py; no
   JavaScript changes. That is the whole reason the schema is uniform.

   Two behaviours worth knowing about:

     * Figures render lazily, on first visit to their section. The payload
       carries ~19 Plotly figures and drawing them all up front costs seconds
       of blocking layout for charts the user may never scroll to.

     * The payload is read from an inline <script id="payload"> when present
       (the --standalone build) and fetched otherwise. That fallback is what
       lets one file work both as a served SPA and as a double-clickable
       artefact, since fetch() is blocked under file://.
*/

(function () {
  'use strict';

  // ── Formatting ─────────────────────────────────────────────────────────
  // The payload carries raw numbers and a format token; rendering decisions
  // live here so the JSON stays useful to any other consumer.

  const NBSP = ' '; // figure space — keeps "n/a" aligned in numeric columns

  function formatValue(value, token) {
    if (value === null || value === undefined) return 'n/a';
    switch (token) {
      case 'percent':
        return (value * 100).toFixed(value !== 0 && Math.abs(value) < 0.01 ? 3 : 2) + '%';
      case 'ratio':
        return Number(value).toFixed(3);
      case 'integer':
        return Number(value).toLocaleString('en-US');
      case 'text':
        return String(value);
      case 'number':
      default:
        if (typeof value !== 'number') return String(value);
        if (value === 0) return '0';
        if (Math.abs(value) >= 1000) return value.toLocaleString('en-US', { maximumFractionDigits: 0 });
        if (Math.abs(value) < 0.001) return value.toExponential(2);
        return value.toFixed(3);
    }
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  // ── Payload loading ────────────────────────────────────────────────────

  function loadPayload() {
    const inline = document.getElementById('payload');
    if (inline) {
      return Promise.resolve(JSON.parse(inline.textContent));
    }
    return fetch('data/analysis.json', { cache: 'no-cache' }).then(function (res) {
      if (!res.ok) throw new Error('HTTP ' + res.status + ' fetching data/analysis.json');
      return res.json();
    });
  }

  function showError(err) {
    const app = document.getElementById('app');
    app.className = 'booting';
    app.innerHTML = '';

    const box = el('div', 'boot-error');
    box.appendChild(el('h1', null, 'Could not load the analysis'));

    const isFileProtocol = window.location.protocol === 'file:';
    if (isFileProtocol) {
      // By far the most likely failure, and the least obvious from the console.
      box.appendChild(el('p', null,
        'The page was opened directly from disk, so the browser blocked the ' +
        'request for the data file. Serve the folder over HTTP instead:'));
      box.appendChild(el('pre', null, 'python -m src.export --serve'));
      box.appendChild(el('p', null,
        'Alternatively, build a self-contained file that needs no server:'));
      box.appendChild(el('pre', null, 'python -m src.export --standalone'));
    } else {
      box.appendChild(el('p', null,
        'The payload is missing or unreadable. Build it with:'));
      box.appendChild(el('pre', null, 'python -m src.export'));
    }

    const detail = el('p', null, '');
    detail.appendChild(el('code', null, String(err && err.message ? err.message : err)));
    box.appendChild(detail);
    app.appendChild(box);
  }

  // ── Rendering ──────────────────────────────────────────────────────────

  function renderStats(stats) {
    const grid = el('div', 'stats');
    stats.forEach(function (s) {
      const tile = el('div', 'stat' + (s.tone && s.tone !== 'neutral' ? ' tone-' + s.tone : ''));
      tile.appendChild(el('p', 'stat-label', s.label));
      const isText = s.format === 'text';
      tile.appendChild(el('p', 'stat-value' + (isText ? ' is-text' : ''),
        formatValue(s.value, s.format)));
      if (s.hint) tile.appendChild(el('p', 'stat-hint', s.hint));
      grid.appendChild(tile);
    });
    return grid;
  }

  function renderTable(spec) {
    const card = el('div', 'card');
    const head = el('div', 'card-head');
    head.appendChild(el('h3', 'card-title', spec.title));
    if (spec.note) head.appendChild(el('p', 'card-note', spec.note));
    card.appendChild(head);

    const scroll = el('div', 'table-scroll');
    const table = el('table');

    const thead = el('thead');
    const hrow = el('tr');
    spec.columns.forEach(function (c) {
      hrow.appendChild(el('th', null, c.replace(/_/g, ' ')));
    });
    thead.appendChild(hrow);
    table.appendChild(thead);

    const tbody = el('tbody');
    spec.rows.forEach(function (row) {
      const tr = el('tr');
      spec.columns.forEach(function (c) {
        const token = spec.formats[c] || (typeof row[c] === 'number' ? 'number' : 'text');
        const td = el('td', null, formatValue(row[c], token));
        if (token === 'text') td.className = 'is-text';
        if (typeof row[c] === 'number' && row[c] < 0) td.classList.add('is-negative');
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    scroll.appendChild(table);
    card.appendChild(scroll);
    return card;
  }

  function renderFigureShell(spec) {
    const card = el('div', 'card');
    if (spec.title || spec.note) {
      const head = el('div', 'card-head');
      if (spec.title) head.appendChild(el('h3', 'card-title', spec.title));
      if (spec.note) head.appendChild(el('p', 'card-note', spec.note));
      card.appendChild(head);
    }
    const wrap = el('div', 'figure');
    const target = el('div', 'figure-shell');
    target.id = 'fig-' + spec.id;
    wrap.appendChild(target);
    card.appendChild(wrap);
    return card;
  }

  const PLOT_CONFIG = {
    displaylogo: false,
    responsive: true,
    // Chart-editing controls are noise on a read-only report; download,
    // zoom and reset are the ones people actually reach for.
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines'],
    toImageButtonOptions: { format: 'png', scale: 2 }
  };

  function renderSection(section, host, drawnFigures) {
    host.innerHTML = '';

    const head = el('div', 'page-head');
    head.appendChild(el('h2', 'page-title', section.title));
    if (section.subtitle) head.appendChild(el('p', 'page-subtitle', section.subtitle));
    host.appendChild(head);

    if (section.error) {
      const box = el('div', 'section-error');
      box.appendChild(el('h3', null, 'This analysis could not be computed'));
      const p = el('p', null, '');
      p.appendChild(el('code', null, section.error));
      box.appendChild(p);
      box.appendChild(el('p', 'card-note',
        'The remaining sections were built normally. Re-run the export with ' +
        'more history, or check the build log for the full traceback.'));
      host.appendChild(box);
      return;
    }

    if (section.stats && section.stats.length) host.appendChild(renderStats(section.stats));

    (section.figures || []).forEach(function (fig) {
      host.appendChild(renderFigureShell(fig));
    });

    (section.tables || []).forEach(function (tbl) {
      host.appendChild(renderTable(tbl));
    });

    if (section.notes && section.notes.length) {
      const notes = el('div', 'notes');
      section.notes.forEach(function (n) { notes.appendChild(el('p', 'note', n)); });
      host.appendChild(notes);
    }

    // Draw after the shells are in the document: Plotly sizes to its container,
    // and a container that is not laid out yet gets the wrong width.
    (section.figures || []).forEach(function (fig) {
      const target = document.getElementById('fig-' + fig.id);
      if (!target) return;
      Plotly.newPlot(target, fig.data, fig.layout, PLOT_CONFIG);
      drawnFigures.add(fig.id);
    });
  }

  function renderWarnings(meta) {
    const messages = (meta.warnings || []).slice();
    if (meta.quality && meta.quality.critical > 0) {
      messages.unshift(
        meta.quality.critical + ' critical data quality finding(s) across ' +
        meta.quality.tickers_checked + ' tickers. Inspect them before trusting ' +
        'anything downstream.');
    }
    if (!messages.length) return null;

    const banner = el('div', 'banner');
    banner.appendChild(el('span', 'banner-icon', '!'));
    const list = el('ul');
    messages.forEach(function (m) { list.appendChild(el('li', null, m)); });
    banner.appendChild(list);
    return banner;
  }

  // ── Application ────────────────────────────────────────────────────────

  function render(payload) {
    const app = document.getElementById('app');
    const meta = payload.meta;
    const sections = payload.sections;
    const drawnFigures = new Set();

    app.className = 'shell';
    app.innerHTML = '';

    // Sidebar
    const sidebar = el('aside', 'sidebar');
    const brand = el('div', 'brand');
    brand.appendChild(el('p', 'brand-name', meta.portfolio.name));
    brand.appendChild(el('p', 'brand-sub',
      meta.coverage.start + ' → ' + meta.coverage.end +
      ' · ' + meta.coverage.observations.toLocaleString('en-US') + ' obs'));
    sidebar.appendChild(brand);

    const nav = el('nav', 'nav');
    const buttons = {};
    sections.forEach(function (s, i) {
      const btn = el('button', 'nav-item' + (s.error ? ' has-error' : ''));
      btn.type = 'button';
      btn.appendChild(el('span', 'nav-index', String(i + 1).padStart(2, '0')));
      btn.appendChild(el('span', null, s.title));
      btn.addEventListener('click', function () { activate(s.id); });
      nav.appendChild(btn);
      buttons[s.id] = btn;
    });
    sidebar.appendChild(nav);

    const foot = el('dl', 'sidebar-foot');
    foot.appendChild(el('dt', null, 'Generated'));
    foot.appendChild(el('dd', null, meta.generated_at.replace('T', ' ')));
    foot.appendChild(el('dt', null, 'Covariance'));
    foot.appendChild(el('dd', null, meta.portfolio.cov_method));
    if (meta.quality) {
      foot.appendChild(el('dt', null, 'Data quality'));
      foot.appendChild(el('dd', null,
        meta.quality.findings + ' findings, ' + meta.quality.critical + ' critical'));
    }
    sidebar.appendChild(foot);
    app.appendChild(sidebar);

    // Main column
    const main = el('main', 'main');
    const warnings = renderWarnings(meta);
    if (warnings) main.appendChild(warnings);
    const host = el('div', 'section-host');
    main.appendChild(host);
    app.appendChild(main);

    function activate(id) {
      const section = sections.find(function (s) { return s.id === id; }) || sections[0];
      Object.keys(buttons).forEach(function (k) {
        buttons[k].classList.toggle('is-active', k === section.id);
      });
      renderSection(section, host, drawnFigures);
      if (window.location.hash !== '#' + section.id) {
        // replaceState, not a hash assignment: this must not push a history
        // entry per click, or Back walks the user through every tab visited.
        history.replaceState(null, '', '#' + section.id);
      }
      window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
    }

    window.addEventListener('hashchange', function () {
      const id = window.location.hash.slice(1);
      if (id && buttons[id]) activate(id);
    });

    window.addEventListener('resize', function () {
      host.querySelectorAll('.figure-shell').forEach(function (node) {
        if (node.data) Plotly.Plots.resize(node);
      });
    });

    const initial = window.location.hash.slice(1);
    activate(buttons[initial] ? initial : sections[0].id);
  }

  loadPayload().then(render).catch(showError);
})();
