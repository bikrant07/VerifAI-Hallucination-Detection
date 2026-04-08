// background.js facilitates communication/storage
chrome.runtime.onInstalled.addListener(() => {
  console.log("VerifAI Extension Installed");
});

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "openDashboard") {
    chrome.tabs.create({ url: "http://localhost:8501" });
    if (sendResponse) sendResponse({ status: "success" });
    return true;
  }

  if (request.action === "storeSelection") {
    chrome.storage.local.set({ selectedText: request.text }, () => {
      console.log("Selection stored:", request.text);
      if (sendResponse) sendResponse({ status: "success" });
    });
    return true;
  }

  if (request.action === "verifyClaim") {
    const { claim, query } = request;
    console.log("VerifAI: Background received verification request.", { claim: claim.substring(0, 50), query: query?.substring(0, 50) });
    
    fetch('http://localhost:8000/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query || "General Context", claim: claim })
    })
    .then(async response => {
      console.log("VerifAI: API responded with status:", response.status);
      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`API Error ${response.status}: ${errText}`);
      }
      return response.json();
    })
    .then(data => {
      console.log("VerifAI: Success! Sending data back to content script.");
      sendResponse({ status: 'success', data });
    })
    .catch(error => {
      console.error("VerifAI: Background Fetch Failure:", error);
      sendResponse({ status: 'error', message: "Backend offline? " + error.message });
    });

    return true; // Keep channel open for async response
  }
});
