import { renderNavbar } from "./components/Navbar.js";
import { renderHero } from "./components/Hero.js";
import { renderVoiceInput } from "./components/VoiceInput.js";
import { renderQueryInput, setQueryInputLoading } from "./components/QueryInput.js";
import { renderAnswerCard } from "./components/AnswerCard.js";
import { renderSourcesPanel } from "./components/SourcesPanel.js";
import { renderSystemStatus } from "./components/SystemStatus.js";
import { renderRagPipeline } from "./components/RagPipeline.js";
import { renderWhyGyaanRag } from "./components/WhyGyaanRag.js";
import { renderResearchMetrics } from "./components/ResearchMetrics.js";

// Global App State
let activeLang = "hi";
let currentQuery = "";

// Element Container Selectors
const navbarContainer = document.querySelector("#navbar-container");
const heroContainer = document.querySelector("#hero-container");
const voiceContainer = document.querySelector("#voice-container");
const queryContainer = document.querySelector("#query-container");
const responseContainer = document.querySelector("#response-container");
const pipelineContainer = document.querySelector("#pipeline-container");
const statusContainer = document.querySelector("#status-container");
const whyContainer = document.querySelector("#why-container");
const metricsContainer = document.querySelector("#metrics-container");

function init() {
    // 1. Render Navigation and Hero Banner
    renderNavbar(navbarContainer, activeLang, handleLanguageChange);
    renderHero(heroContainer, activeLang);
    
    // 2. Render Voice Input (Primary CTA) and Text Query Fallback
    renderVoiceInput(voiceContainer, activeLang, handleVoiceSubmit);
    renderQueryInput(queryContainer, activeLang, handleQuerySubmit);
    
    // 3. Render Technical Terminals
    renderRagPipeline(pipelineContainer, activeLang);
    renderSystemStatus(statusContainer, activeLang);

    // 4. Render Why Gyaan RAG and Research Metrics sections
    renderWhyGyaanRag(whyContainer, activeLang);
    renderResearchMetrics(metricsContainer, activeLang);
}

function handleLanguageChange(lang) {
    if (activeLang === lang) return;
    activeLang = lang;
    
    // Update navbar lang display active states
    renderNavbar(navbarContainer, activeLang, handleLanguageChange);
    
    // Rerender Hero, VoiceInput, QueryInput, and Pipeline panel
    renderHero(heroContainer, activeLang);
    renderVoiceInput(voiceContainer, activeLang, handleVoiceSubmit);
    renderQueryInput(queryContainer, activeLang, handleQuerySubmit);
    
    // Keep input query string if any
    const input = document.querySelector("#query-input");
    if (input) {
        input.value = currentQuery;
    }
    
    renderRagPipeline(pipelineContainer, activeLang);
    renderSystemStatus(statusContainer, activeLang);
    renderWhyGyaanRag(whyContainer, activeLang);
    renderResearchMetrics(metricsContainer, activeLang);

    // If a response is currently visible, rerender response texts
    if (responseContainer.style.display !== "none" && window.lastRAGResponse) {
        renderRAGResponse(window.lastRAGResponse, currentQuery);
    }
}

function renderRAGResponse(data, queryText) {
    window.lastRAGResponse = data;
    responseContainer.style.display = "block";
    
    // Render Answer and Attributed Sources Panel
    responseContainer.innerHTML = `
        <div id="answer-card-mount" class="layout-section"></div>
        <div id="sources-panel-mount" class="layout-section"></div>
    `;
    
    const answerMount = responseContainer.querySelector("#answer-card-mount");
    const sourcesMount = responseContainer.querySelector("#sources-panel-mount");
    
    renderAnswerCard(answerMount, data, activeLang, queryText || data.transcript || currentQuery);
    renderSourcesPanel(sourcesMount, data.sources || [], activeLang);
    
    // Smooth scroll response into viewport
    responseContainer.scrollIntoView({ behavior: "smooth" });
}

function renderLoading(textOverride) {
    responseContainer.style.display = "block";
    const defaultText = activeLang === 'hi' 
        ? 'तथ्यों को खोजा जा रहा है (Retrieving & Generating)...' 
        : 'Retrieving context and generating grounded answer...';
    const loadingText = textOverride || defaultText;
        
    responseContainer.innerHTML = `
        <div class="sketch-card loading-box">
            <div class="loading-spinner"></div>
            <div class="loading-sketch-text font-sketch">${loadingText}</div>
        </div>
    `;
    responseContainer.scrollIntoView({ behavior: "smooth" });
}

function renderError(errMessage) {
    responseContainer.style.display = "block";
    const errorTitle = activeLang === 'hi' ? 'त्रुटि (Error)' : 'ERROR DETECTED';
    const fallbackText = activeLang === 'hi'
        ? 'अनपेक्षित सर्वर त्रुटि। कृपया पुनः प्रयास करें।'
        : 'An unexpected RAG subsystem error occurred. Please try again.';
        
    responseContainer.innerHTML = `
        <div class="sketch-card abstain-card">
            <span class="answer-tag abstain-tag font-display">${errorTitle}</span>
            <div class="answer-text" style="color: var(--hot-pink); font-weight: bold; margin-top: 10px;">
                ${errMessage || fallbackText}
            </div>
            <p class="font-sketch" style="color: var(--dark-black); margin-top: 10px;">
                ${activeLang === 'hi' ? '⚠️ कृपया जाँचें कि uvicorn बैकएंड और API कुंजियाँ सक्रिय हैं।' : '⚠️ Ensure backend server and API credentials are functional.'}
            </p>
        </div>
    `;
    responseContainer.scrollIntoView({ behavior: "smooth" });
}

// ─── Voice Submit Handler ───────────────────────────────────────────────────
async function handleVoiceSubmit(audioBlob, filename, mimeType, onStageUpdate) {
    renderLoading(activeLang === 'hi' ? '🎙️ आवाज़ का विश्लेषण किया जा रहा है...' : '🎙️ Processing voice recording...');
    
    const formData = new FormData();
    formData.append("file", audioBlob, filename);
    formData.append("language_code", activeLang === 'hi' ? "hi-IN" : "en-IN");

    try {
        if (onStageUpdate) onStageUpdate('stt');

        const response = await fetch("/api/voice", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            let errorMsg = `HTTP Error ${response.status}`;
            try {
                const errData = await response.json();
                if (errData && (errData.message || errData.detail)) {
                    errorMsg = errData.message || errData.detail;
                }
            } catch (e) {}
            throw new Error(errorMsg);
        }

        const data = await response.json();
        
        if (onStageUpdate) onStageUpdate('done');
        currentQuery = data.transcript || "";
        renderRAGResponse(data, data.transcript);

    } catch (err) {
        console.error("Voice RAG pipeline failure:", err);
        let customMessage = err.message;
        if (err instanceof TypeError && err.message === "Failed to fetch") {
            customMessage = activeLang === 'hi'
                ? "नेटवर्क त्रुटि: बैकएंड सर्वर से कनेक्शन नहीं हो सका।"
                : "Network Error: Could not connect to the backend server.";
        }
        renderError(customMessage);
    }
}

// ─── Text Query Submit Handler with Real-Time SSE Streaming ────────────────
async function handleQuerySubmit(query) {
    currentQuery = query;
    setQueryInputLoading(queryContainer, true, activeLang);
    
    // Mount initial progressive streaming card
    responseContainer.style.display = "block";
    const initialStreamLabel = activeLang === 'hi' ? '⚡ साक्ष्य स्ट्रीमिंग (Streaming)...' : '⚡ STREAMING EVIDENCE...';
    responseContainer.innerHTML = `
        <div id="answer-card-mount" class="layout-section">
            <div class="sketch-card">
                <div class="card-header-bar" style="margin-bottom: 12px;">
                    <span class="answer-tag font-display" style="background: var(--electric-yellow); color: var(--dark-black);">${initialStreamLabel}</span>
                </div>
                <div id="streaming-answer-text" class="answer-text font-serif" style="font-size: 1.15rem; line-height: 1.6; min-height: 48px; color: var(--dark-black);">
                    <span class="streaming-cursor">▊</span>
                </div>
            </div>
        </div>
        <div id="sources-panel-mount" class="layout-section"></div>
    `;
    responseContainer.scrollIntoView({ behavior: "smooth" });

    try {
        const response = await fetch("/api/stream", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ query: query })
        });

        if (!response.ok || !response.body) {
            // Fallback to synchronous endpoint
            const syncRes = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query })
            });
            const data = await syncRes.json();
            setQueryInputLoading(queryContainer, false, activeLang);
            renderRAGResponse(data, query);
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let accumulatedText = "";
        let buffer = "";
        const streamingTextEl = document.querySelector("#streaming-answer-text");

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            buffer = lines.pop(); // Keep trailing incomplete fragment

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    try {
                        const payload = JSON.parse(line.slice(6));
                        if (payload.type === "token") {
                            accumulatedText += payload.delta;
                            if (streamingTextEl) {
                                streamingTextEl.innerHTML = `${accumulatedText}<span class="streaming-cursor" style="color: var(--hot-pink);">▊</span>`;
                            }
                        } else if (payload.type === "done") {
                            setQueryInputLoading(queryContainer, false, activeLang);
                            renderRAGResponse(payload, query);
                            return;
                        }
                    } catch (e) {}
                }
            }
        }
        setQueryInputLoading(queryContainer, false, activeLang);
    } catch (err) {
        setQueryInputLoading(queryContainer, false, activeLang);
        console.error("Streaming failure:", err);
        // Fallback to synchronous query
        try {
            const syncRes = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query })
            });
            const data = await syncRes.json();
            renderRAGResponse(data, query);
        } catch (e2) {
            renderError(err.message);
        }
    }
}

// Initialise App on DOM Load
document.addEventListener("DOMContentLoaded", init);
