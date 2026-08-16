import { renderNavbar } from "./components/Navbar.js";
import { renderHero } from "./components/Hero.js";
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
    
    // 2. Render Query Input Component
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
    
    // Rerender Hero, QueryInput, and Pipeline panel language texts
    renderHero(heroContainer, activeLang);
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
        renderRAGResponse(window.lastRAGResponse);
    }
}

function renderRAGResponse(data) {
    window.lastRAGResponse = data;
    responseContainer.style.display = "block";
    
    // Render Answer and Attributed Sources Panel
    responseContainer.innerHTML = `
        <div id="answer-card-mount" class="layout-section"></div>
        <div id="sources-panel-mount" class="layout-section"></div>
    `;
    
    const answerMount = responseContainer.querySelector("#answer-card-mount");
    const sourcesMount = responseContainer.querySelector("#sources-panel-mount");
    
    renderAnswerCard(answerMount, data, activeLang, currentQuery);
    renderSourcesPanel(sourcesMount, data.sources, activeLang);
    
    // Smooth scroll response into viewport
    responseContainer.scrollIntoView({ behavior: "smooth" });
}

function renderLoading() {
    responseContainer.style.display = "block";
    const loadingText = activeLang === 'hi' 
        ? 'तथ्यों को खोजा जा रहा है (Searching indices)...' 
        : 'Retrieving context from indices...';
        
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
            <div class="answer-text" style="color: var(--hot-pink); font-weight: bold;">
                ${errMessage || fallbackText}
            </div>
            <p class="font-sketch" style="color: var(--dark-black); margin-top: 10px;">
                ${activeLang === 'hi' ? '⚠️ कृपया जाँचें कि uvicorn बैकएंड सक्रिय है।' : '⚠️ Ensure uvicorn server is active.'}
            </p>
        </div>
    `;
    responseContainer.scrollIntoView({ behavior: "smooth" });
}

async function handleQuerySubmit(query) {
    currentQuery = query;
    setQueryInputLoading(queryContainer, true, activeLang);
    renderLoading();
    
    try {
        const response = await fetch("/api/query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ query: query })
        });
        
        setQueryInputLoading(queryContainer, false, activeLang);
        
        if (!response.ok) {
            let errorMsg = `HTTP Error ${response.status}`;
            try {
                const errData = await response.json();
                if (errData && errData.detail) {
                    errorMsg += `: ${errData.detail}`;
                }
            } catch (e) {
                // Not JSON
            }
            throw new Error(errorMsg);
        }
        
        const data = await response.json();
        renderRAGResponse(data);
        
    } catch (err) {
        setQueryInputLoading(queryContainer, false, activeLang);
        console.error("RAG Query submit failure details:", err);
        
        let customMessage = err.message;
        if (err instanceof TypeError && err.message === "Failed to fetch") {
            customMessage = activeLang === 'hi'
                ? "नेटवर्क त्रुटि (Failed to fetch): बैकएंड सर्वर से कनेक्शन स्थापित नहीं हो सका।"
                : "Network Error (Failed to fetch): Could not connect to the backend server.";
        }
        renderError(customMessage);
    }
}

// Initialise App on DOM Load
document.addEventListener("DOMContentLoaded", init);
