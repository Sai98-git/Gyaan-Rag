export function renderSystemStatus(container, activeLang) {
    const titleText = activeLang === 'hi' ? 'सिस्टम स्थिति' : 'SYSTEM STATUS';
    
    container.innerHTML = `
        <div class="sketch-card-dark system-status-board">
            <h3 class="font-display" style="color: var(--electric-yellow); border-bottom: 2px dashed var(--electric-yellow); padding-bottom: 0.5rem; margin-bottom: 1rem;">
                ${titleText}
            </h3>
            <div class="status-row">
                <span>RETRIEVAL:</span>
                <span class="status-val">DENSE COSINE ✓</span>
            </div>
            <div class="status-row">
                <span>EMBEDDING:</span>
                <span class="status-val">E5-SMALL (LOCAL) ✓</span>
            </div>
            <div class="status-row">
                <span>CHUNKER:</span>
                <span class="status-val">SEMANTIC/STRUCTURE ✓</span>
            </div>
            <div class="status-row">
                <span>GENERATION:</span>
                <span class="status-val">SARVAM/MOCK ✓</span>
            </div>
            <div class="status-row">
                <span>GROUNDING:</span>
                <span class="status-val" style="color: var(--hot-pink);">GUARD ACTIVE ✓</span>
            </div>
        </div>
    `;
}
