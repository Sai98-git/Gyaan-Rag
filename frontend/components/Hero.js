export function renderHero(container, activeLang) {
    const isHi = activeLang === 'hi';
    const headline = isHi 
        ? 'बोलकर पूछें। खोजें। समझें।' 
        : 'SPEAK. RETRIEVE. EXPLAIN.';
        
    const subtitle = isHi
        ? 'भारतीय भाषाओं के लिए पहला रीयल-टाइम वॉइस-इनेबल्ड साक्ष्य-आधारित RAG मॉडल।'
        : 'An Indic-first, Voice-Enabled Retrieval-Augmented Generation system anchored to MSMARCO-XI evidence.';
        
    container.innerHTML = `
        <div class="poster-hero">
            <span class="hero-tag font-display">${isHi ? '🎙️ वॉइस RAG सक्रिय' : '🎙️ VOICE-ENABLED RAG ACTIVE'}</span>
            <h1 class="hero-headline font-display">${headline}</h1>
            <p class="hero-subtitle">${subtitle}</p>
            <div class="hero-badges">
                <span class="brutalist-badge">🎙 Sarvam Indic STT</span>
                <span class="brutalist-badge">🎯 Multilingual E5 Vectors</span>
                <span class="brutalist-badge">📦 3 Chunking Strategies</span>
                <span class="brutalist-badge">🛡️ Grounding Guardrails</span>
                <span class="brutalist-badge">⚡ Sub-second Latency</span>
            </div>
            <span class="annotation-arrow font-sketch" style="position: absolute; right: 40px; bottom: 10px; transform: rotate(-15deg);">
                ${isHi ? 'सीधे माइक से पूछें →' : 'Ask using your voice →'}
            </span>
        </div>
    `;
}
