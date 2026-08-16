export function renderRagPipeline(container, activeLang) {
    const titleText = activeLang === 'hi' ? 'आरएजी पाइपलाइन कैसे काम करती है' : 'HOW THE RAG PIPELINE WORKS';
    
    container.innerHTML = `
        <div class="sketch-card-dark" style="box-shadow: 8px 8px 0px var(--electric-yellow); border-color: var(--hot-pink);">
            <h3 class="font-display" style="color: var(--hot-pink); border-bottom: 2px dashed var(--hot-pink); padding-bottom: 0.5rem; margin-bottom: 1rem;">
                ${titleText}
            </h3>
            <div class="pipeline-flow">
                <div class="flow-step font-display">
                    1. USER QUERY
                    <p>${activeLang === 'hi' ? 'उपयोगकर्ता सवाल पूछता है' : 'User asks a question in Hindi or English'}</p>
                </div>
                <div class="flow-step font-display">
                    2. DENSE RETRIEVAL
                    <p>${activeLang === 'hi' ? 'E5-small मॉडल सवाल को वेक्टर में बदलता है' : 'E5-small maps query to semantic vectors'}</p>
                </div>
                <div class="flow-step font-display">
                    3. CONTEXT EXTRACT
                    <p>${activeLang === 'hi' ? 'संबंधित और सत्यापित पैराग्राफ निकाले जाते हैं' : 'Top relevant deduplicated chunks retrieved'}</p>
                </div>
                <div class="flow-step font-display">
                    4. GROUNDING GUARD
                    <p>${activeLang === 'hi' ? 'समानता स्कोर (>= 0.75) और तथ्यों की जांच होती है' : 'Validates similarity >= 0.75 and lexical overlap'}</p>
                </div>
                <div class="flow-step font-display" style="border-color: var(--electric-yellow); color: var(--electric-yellow);">
                    5. GROUNDED ANSWER
                    <p>${activeLang === 'hi' ? 'Sarvam मॉडल सत्यापित तथ्यों के साथ उत्तर देता है' : 'Sarvam generates answer referencing exact source citations'}</p>
                </div>
            </div>
            <p class="font-sketch text-center" style="color: var(--electric-yellow); margin-top: 1rem; text-align: center;">
                ⚡ ${activeLang === 'hi' ? 'कोई इंटरनेट का मनगढ़ंत सच नहीं' : 'No arbitrary internet magic.'}
            </p>
        </div>
    `;
}
