/**
 * VerifAI Universal Hallucination Detector — content.js
 * Implements: 
 * - Shadow DOM Sidebar for results
 * - Selection Tooltip for manual verification
 * - Auto-detection for AI platforms (ChatGPT, Gemini, etc.)
 * - Smart Context Extraction
 */
// VERIFAI_VERSION: 1.1.2_FIXED_RESILIENCE

(function() {
  'use strict';
  console.log("VerifAI: === UNIVERSAL DETECTOR LOADED ===");

  // --- Constants & Config ---
  const BACKEND_URL = 'http://localhost:8000/verify';
  const CHECK_ATTR = 'data-verifai-checked';
  const PLATFORMS = {
    'chat.openai.com': { selector: 'div[data-message-author-role="assistant"]' },
    'gemini.google.com': { selector: '.message-content, .model-response-text' },
    'perplexity.ai': { selector: '.prose' },
    'claude.ai': { selector: '.font-claude-message' },
    'copilot.microsoft.com': { selector: '.ac-textBlock' }
  };

  const CONVERSATIONAL_CLEANUP_REGEX = /(^|\n|\. )(Would you like me to|Let me know if|Hope this helps|Do you want to|Shall I|Can I help with|Feel free to ask|Is there anything else|I can also help|Looking forward to|Let's dive into|Stay tuned for).*(\?|\!|\.)$|(\? )$/gi;

  let abortController = null;

  // --- UI Elements (Shadow DOM) ---
  let sidebarContainer = null;
  let shadowRoot = null;
  let tooltip = null;
  let lastSelection = "";
  let lastQuery = "";

  // Initialize UI
  function initUI() {
    if (sidebarContainer) return;

    // 1. Create Sidebar
    sidebarContainer = document.createElement('div');
    sidebarContainer.id = 'verifai-sidebar-container';
    document.body.appendChild(sidebarContainer);
    shadowRoot = sidebarContainer.attachShadow({ mode: 'open' }); // Changed to open for easier debugging

    // 2. Add Styles to Shadow DOM
    const style = document.createElement('style');
    style.textContent = `
      :host {
        --bg-color: #0d1117;
        --card-bg: #161b22;
        --border-color: #30363d;
        --text-primary: #c9d1d9;
        --text-secondary: #8b949e;
        --link-color: #58a6ff;
        --error-color: #f85149;
        --success-color: #3fb950;
        --warning-color: #d29922;
      }

      #sidebar {
        position: fixed !important;
        right: -420px !important;
        top: 0 !important;
        width: 380px !important;
        height: 100vh !important;
        background: var(--bg-color) !important;
        border-left: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        z-index: 2147483647 !important;
        transition: right 0.3s ease !important;
        display: flex !important;
        flex-direction: column !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        box-shadow: -5px 0 25px rgba(0,0,0,0.5) !important;
        visibility: visible !important;
      }

      #sidebar.open { right: 0 !important; }

      header {
        padding: 16px;
        border-bottom: 1px solid var(--border-color);
        display: flex;
        justify-content: space-between;
        align-items: center;
      }

      .logo { font-size: 20px; font-weight: bold; color: var(--link-color); }
      .close-btn { cursor: pointer; font-size: 24px; color: var(--text-secondary); }

      .content { flex: 1; overflow-y: auto; padding: 20px; }

      .claim-box { font-style: italic; color: var(--text-secondary); margin-bottom: 15px; border-left: 3px solid var(--border-color); padding-left: 10px; }
      
      .verdict-badge {
        padding: 10px; border-radius: 8px; font-weight: bold; text-align: center; margin-bottom: 15px; font-size: 18px;
      }
      .verdict-correct { background: rgba(63, 185, 80, 0.1); color: var(--success-color); border: 1px solid var(--success-color); }
      .verdict-hallucinated { background: rgba(248, 81, 73, 0.1); color: var(--error-color); border: 1px solid var(--error-color); }

      .progress-bar { height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; margin-bottom: 20px; }
      .progress-fill { height: 100%; background: var(--link-color); width: 0; transition: width 1s ease; }

      .error-pill { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; margin-bottom: 10px; }

      .correction-box { border: 1px solid var(--success-color); border-radius: 8px; padding: 12px; margin-bottom: 15px; background: rgba(63, 185, 80, 0.05); }
      .counterfactual-box { border: 1px solid var(--warning-color); border-radius: 8px; padding: 12px; margin-bottom: 15px; background: rgba(210, 153, 34, 0.05); }

      .highlight { background: rgba(248, 81, 73, 0.3); color: white; border-bottom: 2px solid var(--error-color); }

      .evidence-item { margin-top: 10px; font-size: 13px; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; }

      /* Manual Input Section */
      .input-section { padding: 15px; border-bottom: 1px solid var(--border-color); background: var(--card-bg); }
      .input-group { margin-bottom: 10px; }
      .input-group label { display: block; font-size: 11px; color: var(--text-secondary); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
      .input-group textarea {
        width: 100%; background: var(--bg-color); border: 1px solid var(--border-color);
        color: var(--text-primary); border-radius: 6px; padding: 10px; font-size: 13px;
        resize: vertical; box-sizing: border-box; font-family: inherit; transition: border-color 0.2s;
      }
      .input-group textarea:focus { border-color: var(--link-color); outline: none; }
      
      .verify-btn {
        width: 100%; background: #238636; color: white; border: none; padding: 10px;
        border-radius: 6px; font-weight: bold; cursor: pointer; transition: all 0.2s;
        display: flex; align-items: center; justify-content: center; gap: 8px;
        font-size: 13px;
      }
      .verify-btn:hover { background: #2ea043; transform: translateY(-1px); }
      .verify-btn:disabled { background: #21262d; color: var(--text-secondary); cursor: not-allowed; transform: none; }
      
      .spinner {
        width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3);
        border-radius: 50%; border-top-color: #fff; animation: spin 0.8s linear infinite;
        display: none;
      }
      @keyframes spin { to { transform: rotate(360deg); } }
      .verify-btn.loading .spinner { display: block; }
      .verify-btn.loading .btn-text { display: none; }

      .skeleton { height: 20px; background: linear-gradient(90deg, #161b22 25%, #21262d 50%, #161b22 75%); background-size: 200% 100%; animation: pulse 1.5s infinite; border-radius: 4px; margin-bottom: 10px; }
      @keyframes pulse { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }

      footer { padding: 16px; font-size: 11px; color: var(--text-secondary); text-align: center; }
    `;
    shadowRoot.appendChild(style);

    const sidebar = document.createElement('div');
    sidebar.id = 'sidebar';
    sidebar.innerHTML = `
      <header>
        <div class="logo">VerifAI</div>
        <div class="close-btn" id="close-sidebar">×</div>
      </header>
      <div class="input-section">
        <div class="input-group">
          <label>CLAIM TO VERIFY</label>
          <textarea id="manual-claim" rows="3" placeholder="Paste claim here..."></textarea>
        </div>
        <div class="input-group">
          <label>CONTEXT / QUERY</label>
          <textarea id="manual-query" rows="2" placeholder="What is the topic?"></textarea>
        </div>
        <button class="verify-btn" id="manual-verify">
          <div class="spinner"></div>
          <span class="btn-text">Verify Now</span>
        </button>
      </div>
      <div class="content" id="sidebar-content"></div>
      <footer>Powered by VerifED | <a href="#" id="open-dashboard-link" style="color:var(--link-color)">Open Dashboard</a></footer>
    `;
    shadowRoot.appendChild(sidebar);

    shadowRoot.getElementById('close-sidebar').onclick = () => sidebar.classList.remove('open');
    
    // Open Dashboard Event
    const dashLink = shadowRoot.getElementById('open-dashboard-link');
    if (dashLink) {
      dashLink.onclick = (e) => {
        e.preventDefault();
        safeSendMessage({ action: "openDashboard" });
      };
    }
    
    // Manual Verification Event
    shadowRoot.getElementById('manual-verify').onclick = () => {
      const claim = shadowRoot.getElementById('manual-claim').value;
      const query = shadowRoot.getElementById('manual-query').value;
      if (claim && query) {
        verifyClaim(claim, query);
      }
    };

    // 3. Create Tooltip
    tooltip = document.createElement('div');
    tooltip.id = 'verifai-tooltip';
    tooltip.style.cssText = `
      position: absolute; display: none; background: #0d1117; color: white; padding: 6px 12px;
      border-radius: 20px; font-size: 13px; cursor: pointer; z-index: 2147483647;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5); border: 1px solid #30363d; font-weight: bold;
    `;
    tooltip.innerHTML = '🔍 Verify with VerifAI';
    document.body.appendChild(tooltip);

    tooltip.onclick = (e) => {
      e.stopPropagation();
      console.log("VerifAI: Tooltip clicked. Claim:", lastSelection);
      if (lastSelection) {
        verifyClaim(lastSelection, lastQuery || extractPageContext());
        tooltip.style.display = 'none';
      } else {
        console.warn("VerifAI: No selection captured.");
      }
    };

    // 4. Create Floating Action Button (FAB)
    const fab = document.createElement('div');
    fab.id = 'verifai-fab';
    fab.title = "VerifAI - Toggle Sidebar";
    fab.style.cssText = `
      position: fixed; bottom: 20px; right: 20px; width: 60px; height: 60px;
      background: #238636; color: white; border-radius: 50%; display: flex;
      align-items: center; justify-content: center; cursor: pointer;
      z-index: 2147483645; box-shadow: 0 4px 15px rgba(0,0,0,0.5);
      font-weight: bold; font-size: 11px; text-align: center;
      transition: transform 0.2s; border: 2px solid rgba(255,255,255,0.1);
    `;
    fab.innerHTML = 'VERIFAI';
    fab.onmouseover = () => fab.style.transform = 'scale(1.1)';
    fab.onmouseout = () => fab.style.transform = 'scale(1.0)';
    fab.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log("VerifAI: FAB Clicked");
      const sidebar = shadowRoot.getElementById('sidebar');
      if (sidebar) {
        const isOpen = sidebar.classList.toggle('open');
        console.log("VerifAI: Sidebar toggle. New state:", isOpen ? "OPEN" : "CLOSED");
        if (isOpen && !shadowRoot.getElementById('sidebar-content').innerHTML) {
          shadowRoot.getElementById('sidebar-content').innerHTML = `
            <div style="text-align:center; padding:40px 20px; color:var(--text-secondary);">
              <div style="font-size:40px; margin-bottom:10px;">🔍</div>
              <p>Highlight text on the page and click the floating <strong>Verify</strong> tooltip to see results here.</p>
            </div>
          `;
        }
      } else {
        console.error("VerifAI: Sidebar element not found in Shadow DOM!");
      }
    };
    document.body.appendChild(fab);
    console.log("VerifAI: FAB Injected.");
  }

  // --- Logic Functions ---

  function cleanLLMResponse(text) {
    if (!text) return "";
    // Remove common conversational endings and trailing questions
    let cleaned = text.replace(CONVERSATIONAL_CLEANUP_REGEX, '$1');
    // Remove trailing whitespace and repeated newlines
    return cleaned.trim().replace(/\n{3,}/g, '\n\n');
  }

  // Ultimate safety wrapper for chrome.runtime
  function getSafeRuntime() {
    try {
      if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.id) {
        return chrome.runtime;
      }
    } catch (e) {
      console.warn("VerifAI: Extension context is fully invalidated.");
    }
    return null;
  }

  function isContextValid() {
    return !!getSafeRuntime();
  }

  function showRefreshNotice() {
    if (!shadowRoot) return;
    const sidebarContent = shadowRoot.getElementById('sidebar-content');
    if (sidebarContent) {
      sidebarContent.innerHTML = `
        <div style="color:var(--error-color); padding: 20px; text-align:center;">
          <div style="font-size:30px; margin-bottom:10px;">🔄</div>
          <strong>Extension Updated</strong><br>
          Please <b>refresh this page</b> to continue using VerifAI.<br><br>
          <button id="refresh-page-btn" style="background:var(--link-color); color:white; border:none; padding:8px 16px; border-radius:4px; cursor:pointer; font-weight:bold;">Refresh Now</button>
        </div>
      `;
      const btn = shadowRoot.getElementById('refresh-page-btn');
      if (btn) btn.onclick = () => window.location.reload();
    }
  }

  function safeSendMessage(message, callback) {
    const runtime = getSafeRuntime();
    if (runtime) {
      try {
        runtime.sendMessage(message, (response) => {
          const lastErr = runtime.lastError;
          if (lastErr) {
            console.warn("VerifAI: Runtime error during sendMessage:", lastErr.message);
            if (lastErr.message.includes("context invalidated")) {
              showRefreshNotice();
            }
            if (callback) callback({ status: 'error', message: lastErr.message });
          } else {
            if (callback) callback(response);
          }
        });
      } catch (e) {
        console.error("VerifAI: sendMessage exception:", e);
        if (e.message.includes("context invalidated")) {
          showRefreshNotice();
        }
      }
    } else {
      showRefreshNotice();
    }
  }

  function extractPageContext() {
    const title = document.title;
    const metaDesc = document.querySelector('meta[name="description"]')?.content || '';
    const h1 = document.querySelector('h1')?.innerText || '';
    return `${title} | ${h1} | ${metaDesc}`;
  }

  function getPlatformQuery(node) {
    const host = window.location.hostname;
    try {
      if (host.includes('chat.openai.com')) {
        // Updated ChatGPT logic: match the user message in the previous turn
        const turn = node.closest('[data-testid^="conversation-turn-"]');
        if (turn) {
          const prevTurn = turn.previousElementSibling;
          const userMsg = prevTurn?.querySelector('[data-message-author-role="user"]');
          if (userMsg) return userMsg.innerText.trim();
        }
      } else if (host.includes('gemini.google.com')) {
        // Gemini: Find the query text associated with this model response
        const responseElement = node.closest('.model-response-container, .message-content-container');
        if (responseElement) {
          const userQueryNode = responseElement.previousElementSibling?.querySelector('.query-text');
          if (userQueryNode) return userQueryNode.innerText.trim();
        }
      } else if (host.includes('claude.ai')) {
        // Claude: Specific container-based lookup
        const container = node.closest('[data-testid="message-container"]');
        if (container) {
          const userMsg = container.previousElementSibling?.querySelector('[data-testid="user-message"]');
          if (userMsg) return userMsg.innerText.trim();
        }
      }
    } catch (e) {
      console.warn("VerifAI: Smart Query Extraction Failed", e);
    }
    return extractPageContext();
  }

  async function verifyClaim(claim, context) {
    console.log("VerifAI: Starting verification for:", claim.substring(0, 50));
    initUI();
    const sidebar = shadowRoot.getElementById('sidebar');
    if (!sidebar) {
      console.error("VerifAI: Cannot open sidebar - element missing.");
      return;
    }

    // Fill inputs for visibility
    const clInput = shadowRoot.getElementById('manual-claim');
    const quInput = shadowRoot.getElementById('manual-query');
    if (clInput) clInput.value = claim;
    if (quInput) quInput.value = context;

    sidebar.classList.add('open');

    const sidebarContent = shadowRoot.getElementById('sidebar-content');
    const verifyBtn = shadowRoot.getElementById('manual-verify');
    if (!sidebarContent) {
      console.error("VerifAI: Sidebar content area not found.");
      return;
    }

    // Loading State
    if (verifyBtn) {
      verifyBtn.classList.add('loading');
      verifyBtn.disabled = true;
    }

    sidebarContent.innerHTML = `
      <div style="padding: 20px; text-align:center;">
        <div class="error-label" style="font-size:11px; margin-bottom:10px;">ANALYZING...</div>
        <div class="skeleton"></div><div class="skeleton"></div><div class="skeleton" style="width:70%"></div>
      </div>
    `;

    // Send to background for fetch
    safeSendMessage({
      action: "verifyClaim",
      claim: claim,
      query: context
    }, (response) => {
      if (verifyBtn) {
        verifyBtn.classList.remove('loading');
        verifyBtn.disabled = false;
      }
      if (response && response.status === 'success') {
        renderResults(response.data, claim);
      } else {
        const error = response ? response.message : 'Unknown error';
        // Only show error if it's not a context invalidation (which is handled in safeSendMessage)
        if (!error.includes("context invalidated")) {
          sidebarContent.innerHTML = `<div style="color:var(--error-color); padding: 20px; text-align:center;">
            <div style="font-size:30px; margin-bottom:10px;">⚠️</div>
            <strong>Backend Error</strong><br>
            ${error}<br><br>
            <small>Make sure FastAPI is running on port 8000</small>
          </div>`;
        }
      }
    });
  }

  function renderResults(data, claim) {
    const sidebarContent = shadowRoot.getElementById('sidebar-content');
    if (!sidebarContent) return;
    const badgeClass = data.verdict === 'Factually Correct' ? 'verdict-correct' : 'verdict-hallucinated';
    
    let highlightedClaim = claim;
    if (data.highlighted_span) {
      highlightedClaim = claim.replace(data.highlighted_span, `<mark class="highlight">${data.highlighted_span}</mark>`);
    }

    const errorColors = {
      'Fabrication': { bg: 'rgba(248,81,73,0.2)', text: '#f85149' },
      'Overclaiming': { bg: 'rgba(210,153,34,0.2)', text: '#d29922' },
      'Omission': { bg: 'rgba(210,153,34,0.1)', text: '#e3b341' },
      'Entity Confusion': { bg: 'rgba(88,166,255,0.2)', text: '#58a6ff' },
      'Factual Substitution': { bg: 'rgba(163,113,247,0.2)', text: '#a371f7' }
    };
    const eStyle = errorColors[data.error_type] || { bg: 'rgba(139,148,158,0.2)', text: '#8b949e' };

    sidebarContent.innerHTML = `
      <div class="claim-box">${highlightedClaim}</div>
      <div class="verdict-badge ${badgeClass}">${data.verdict.toUpperCase()}</div>
      
      <div class="progress-bar"><div class="progress-fill" id="fill-bar"></div></div>
      
      ${data.error_type ? `<div class="error-pill" style="background:${eStyle.bg}; color:${eStyle.text}; border: 1px solid ${eStyle.text}">${data.error_type}</div>` : ''}
      
      <p style="font-size:13px; color:var(--text-secondary); margin-bottom:20px;">${data.reason}</p>

      ${data.correction ? `<div class="error-label" style="font-size:11px; margin-bottom:5px;">CORRECTION</div><div class="correction-box">${data.correction}</div>` : ''}
      ${data.counterfactual ? `<div class="error-label" style="font-size:11px; margin-bottom:5px;">COUNTERFACTUAL</div><div class="counterfactual-box">${data.counterfactual}</div>` : ''}

      <div class="error-label" style="font-size:11px; margin-top:20px;">TOP EVIDENCE</div>
      <div id="evidence-list"></div>
    `;

    // Animate progress bar
    setTimeout(() => {
      shadowRoot.getElementById('fill-bar').style.width = `${data.confidence * 100}%`;
    }, 50);

    if (data.evidence && data.evidence.length > 0) {
      const eList = shadowRoot.getElementById('evidence-list');
      data.evidence.slice(0, 3).forEach(e => {
        const item = document.createElement('div');
        item.className = 'evidence-item';
        item.innerHTML = `<strong>[${e.source}]</strong> ${e.text.substring(0, 200)}...`;
        eList.appendChild(item);
      });
    }
  }

  // --- Observation & Interaction ---

  function injectVerifyButtons() {
    const host = window.location.hostname;
    let selector = null;
    for (const [key, platform] of Object.entries(PLATFORMS)) {
      if (host.includes(key)) {
        selector = platform.selector;
        break;
      }
    }

    if (!selector) return;

    const elements = document.querySelectorAll(`${selector}:not([${CHECK_ATTR}])`);
    elements.forEach(el => {
      el.setAttribute(CHECK_ATTR, 'true');
      const btn = document.createElement('button');
      btn.innerHTML = '🔍 Verify with VerifAI';
      btn.style.cssText = `
        display: block; margin-top: 10px; background: #238636; color: white; border: none;
        padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: bold;
      `;
      btn.onclick = (e) => {
        e.preventDefault();
        const rawClaim = el.innerText;
        const cleanedClaim = cleanLLMResponse(rawClaim);
        console.log("VerifAI: Auto-detect verification triggered.", { raw: rawClaim.length, cleaned: cleanedClaim.length });
        verifyClaim(cleanedClaim, getPlatformQuery(el));
      };
      
      // Inject after the element (or at the end of it)
      el.appendChild(btn);
    });
  }

  // Handle Selection Tooltip
  document.addEventListener('mouseup', (e) => {
    if (!isContextValid()) return;
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();
    
    if (selectedText.length < 5) { // Ignore tiny selections or empty
      tooltip.style.display = 'none';
      return;
    }

    // Capture state immediately
    lastSelection = selectedText;
    
    // Find surrounding paragraph/div as context
    const container = selection.anchorNode?.parentElement?.closest('p, div, li, blockquote');
    const paraContext = container ? container.innerText.trim().substring(0, 500) : "";
    lastQuery = paraContext || extractPageContext();
    
    console.log("VerifAI: Captured selection:", { claim: lastSelection.substring(0, 50), context: lastQuery.substring(0, 50) });

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    
    // Position tooltip 10px above selection
    tooltip.style.left = `${rect.left + window.scrollX + (rect.width/2) - 60}px`;
    tooltip.style.top = `${rect.top + window.scrollY - 40}px`;
    tooltip.style.display = 'block';
  });

  // Hide tooltip on click anywhere else
  document.addEventListener('mousedown', (e) => {
    if (!isContextValid()) return;
    if (tooltip && e.target !== tooltip) {
      tooltip.style.display = 'none';
    }
  });

  // Robust Initialization
  function startVerifAI() {
    if (document.body) {
      console.log("VerifAI: Initializing UI...");
      initUI();
      injectVerifyButtons();
      
      const observer = new MutationObserver(() => {
        if (!isContextValid()) {
          observer.disconnect();
          return;
        }
        clearTimeout(timeout);
        timeout = setTimeout(injectVerifyButtons, 50);
      });
      observer.observe(document.body, { childList: true, subtree: true });
      console.log("VerifAI: Fully active.");
    } else {
      console.warn("VerifAI: document.body not ready. Retrying...");
      setTimeout(startVerifAI, 100);
    }
  }

  let timeout = null;
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    startVerifAI();
  } else {
    window.addEventListener('load', startVerifAI);
  }

})();
