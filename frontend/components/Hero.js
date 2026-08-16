export function renderHero(container, activeLang) {
    const headline = activeLang === 'hi' 
        ? 'पूछो। खोजो। समझो।' 
        : 'ASK. FIND. EXPLAIN.';
        
    const subtitle = activeLang === 'hi'
        ? 'सत्यापित संदर्भों पर आधारित पहला भारतीय-भाषा RAG शोध सहायक।'
        : 'An Indic-first, grounded AI research assistant anchored completely to facts.';
        
    container.innerHTML = `
        <div class="poster-hero">
            <span class="hero-tag font-display">${activeLang === 'hi' ? 'भारतीय RAG सिस्टम' : 'INDIC RAG ACTIVE'}</span>
            <h1 class="hero-headline font-display">${headline}</h1>
            <p class="hero-subtitle">${subtitle}</p>
            <div class="hero-badges">
                <span class="brutalist-badge">✓ Semantic Retrieval</span>
                <span class="brutalist-badge">✓ Multilingual E5</span>
                <span class="brutalist-badge">✓ Grounded Answers</span>
                <span class="brutalist-badge">✓ Zero Hallucinations</span>
            </div>
            <span class="annotation-arrow font-sketch" style="position: absolute; right: 40px; bottom: 10px; transform: rotate(-15deg);">
                ${activeLang === 'hi' ? 'नो रैंडम मैजिक →' : 'No Hallucinations here →'}
            </span>
        </div>
    `;
}
