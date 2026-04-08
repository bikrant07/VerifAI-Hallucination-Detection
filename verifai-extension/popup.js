document.addEventListener('DOMContentLoaded', () => {
  const queryInput = document.getElementById('query');
  const claimInput = document.getElementById('claim');
  const grabBtn = document.getElementById('grab-btn');
  const verifyBtn = document.getElementById('verify-btn');
  const loading = document.querySelector('.loading');
  const resultsSection = document.getElementById('results');
  const errorMsg = document.getElementById('error-message');

  // UI Elements for results
  const verdictBadge = document.getElementById('verdict-badge');
  const confidenceVal = document.getElementById('confidence-val');
  const claimDisplay = document.getElementById('claim-display');
  const reasonText = document.getElementById('reason-text');
  const errorTypeContainer = document.getElementById('error-type-container');
  const errorTypeBadge = document.getElementById('error-type-badge');
  const correctionSection = document.getElementById('correction-section');
  const correctionText = document.getElementById('correction-text');
  const counterfactualSection = document.getElementById('counterfactual-section');
  const counterfactualText = document.getElementById('counterfactual-text');
  const evidenceText = document.getElementById('evidence-text');
  const evidenceSourceTag = document.getElementById('evidence-source-tag');

  // 1. Grab from page logic
  if (grabBtn) {
    grabBtn.addEventListener('click', async () => {
      try {
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tabs || tabs.length === 0) {
          errorMsg.textContent = "No active tab found.";
          errorMsg.style.display = 'block';
          return;
        }
        const tab = tabs[0];
        
        // 1. Try direct message for immediate selection
        chrome.tabs.sendMessage(tab.id, { action: "getSelectedText" }, (response) => {
          if (chrome.runtime.lastError) {
            console.warn("VerifAI: Could not connect to content script. Refresh the page.");
            errorMsg.innerHTML = "Cannot connect to page.<br>Please <b>refresh the website</b> (F5) and try again.";
            errorMsg.style.display = 'block';
            return;
          }
          
          if (!response || !response.text) {
            // 2. Fallback: check storage (passed from background as per requirements)
            chrome.storage.local.get(['selectedText'], (result) => {
              if (result.selectedText) {
                claimInput.value = result.selectedText;
              } else {
                errorMsg.textContent = "No text selected on page.";
                errorMsg.style.display = 'block';
                setTimeout(() => { errorMsg.style.display = 'none'; }, 3000);
              }
            });
            return;
          }
          
          claimInput.value = response.text;
        });

        chrome.tabs.sendMessage(tab.id, { action: "getPageContext" }, (response) => {
          if (response && response.title && !queryInput.value) {
            queryInput.value = response.title;
          }
        });
      } catch (err) {
        console.error(err);
      }
    });
  }

  // 2. Verify facts logic
  verifyBtn.addEventListener('click', async () => {
    const query = queryInput.value.trim();
    const claim = claimInput.value.trim();

    if (!query || !claim) {
      errorMsg.textContent = "Please provide both context and a claim.";
      errorMsg.style.display = 'block';
      return;
    }

    // Reset UI
    errorMsg.style.display = 'none';
    resultsSection.style.display = 'none';
    loading.style.display = 'block';

    try {
      const response = await fetch('http://localhost:8000/verify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, claim }),
      });

      if (!response.ok) {
        throw new Error('API Error');
      }

      const data = await response.json();
      displayResults(data, claim);
    } catch (err) {
      loading.style.display = 'none';
      errorMsg.textContent = "Backend offline — make sure FastAPI is running on port 8000.";
      errorMsg.style.display = 'block';
    } finally {
      loading.style.display = 'none';
    }
  });

  function displayResults(data, originalClaim) {
    loading.style.display = 'none';
    resultsSection.style.display = 'block';

    // Verdict Badge
    verdictBadge.textContent = data.verdict;
    verdictBadge.className = 'badge'; // Reset
    if (data.verdict === 'Factually Correct') {
      verdictBadge.classList.add('badge-green');
    } else if (data.verdict === 'Hallucinated') {
      verdictBadge.classList.add('badge-red');
    } else {
      verdictBadge.classList.add('badge-orange');
    }

    // Confidence
    confidenceVal.textContent = (data.confidence * 100).toFixed(1) + '%';
    
    // Highlighted Span
    if (data.highlighted_span && originalClaim.includes(data.highlighted_span)) {
      const parts = originalClaim.split(data.highlighted_span);
      claimDisplay.innerHTML = `${parts[0]}<span class="span-error">${data.highlighted_span}</span>${parts[1]}`;
    } else {
      claimDisplay.textContent = originalClaim;
    }

    // Reason
    reasonText.textContent = data.reason;

    // Error Type
    if (data.error_type) {
      errorTypeContainer.style.display = 'block';
      errorTypeBadge.textContent = data.error_type;
      errorTypeBadge.className = 'badge badge-red';
    } else {
      errorTypeContainer.style.display = 'none';
    }

    // Correction
    if (data.correction) {
      correctionSection.style.display = 'block';
      correctionText.textContent = data.correction;
    } else {
      correctionSection.style.display = 'none';
    }

    // Counterfactual
    if (data.counterfactual) {
      counterfactualSection.style.display = 'block';
      counterfactualText.textContent = data.counterfactual;
    } else {
      counterfactualSection.style.display = 'none';
    }

    // Evidence
    if (data.evidence && data.evidence.length > 0) {
      const bestEvidence = data.evidence[0];
      evidenceText.textContent = `"${bestEvidence.text}"`;
      evidenceSourceTag.textContent = `Source: ${bestEvidence.source}`;
    } else {
      evidenceText.textContent = "No direct evidence found in knowledge base.";
      evidenceSourceTag.textContent = "";
    }
  }

  const popupDashLink = document.getElementById('popup-dashboard-link');
  if (popupDashLink) {
    popupDashLink.addEventListener('click', (e) => {
      e.preventDefault();
      chrome.tabs.create({ url: "http://localhost:8501" });
    });
  }
});
