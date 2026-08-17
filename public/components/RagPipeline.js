export function renderRagPipeline(container, activeLang) {
    const isHi = activeLang === 'hi';
    const titleText = isHi ? 'वॉयस-इनेबल्ड RAG आर्किटेक्चर' : 'VOICE-ENABLED RAG ARCHITECTURE';
    
    container.innerHTML = `
        <div class="sketch-card-dark" style="box-shadow: 8px 8px 0px var(--electric-yellow); border-color: var(--hot-pink);">
            <h3 class="font-display" style="color: var(--hot-pink); border-bottom: 2px dashed var(--hot-pink); padding-bottom: 0.5rem; margin-bottom: 1rem;">
                ${titleText}
            </h3>
            <div class="pipeline-flow">
                <div class="flow-step font-display">
                    1. 🎙 MICROPHONE INPUT
                    <p>${isHi ? 'ब्राउज़र ऑडियो कैप्चर और लाइव स्ट्रीम' : 'Browser MediaRecorder live audio capture'}</p>
                </div>
                <div class="flow-step font-display">
                    2. 📝 SARVAM STT
                    <p>${isHi ? 'सर्वम Saaras मॉडल द्वारा सटीक स्पीच-टू-टेक्स्ट' : 'Sarvam Saaras ASR converts speech to transcript'}</p>
                </div>
                <div class="flow-step font-display">
                    3. 🧹 QUERY CLEANER
                    <p>${isHi ? 'नॉइज़ फ़िल्टरिंग और टेक्स्ट नॉर्मलाइज़ेशन' : 'Audio artifact filtering and query normalization'}</p>
                </div>
                <div class="flow-step font-display">
                    4. 🎯 E5 VECTOR RETRIEVAL
                    <p>${isHi ? 'MSMARCO-XI से शीर्ष-के पैराग्राफ्स का चयन' : 'Dense cosine search over indexed dataset passages'}</p>
                </div>
                <div class="flow-step font-display">
                    5. 🧠 SARVAM GENERATION
                    <p>${isHi ? 'सत्यापित संदर्भों पर आधारित उत्तर निर्माण' : 'Strict context-conditioned Indic LLM synthesis'}</p>
                </div>
                <div class="flow-step font-display" style="border-color: var(--electric-yellow); color: var(--electric-yellow);">
                    6. 🛡️ GROUNDING GUARD
                    <p>${isHi ? 'स्कोर थ्रेशोल्ड (≥0.78) व लेक्सिकल ओवरलैप' : 'Score thresholding + hallucination guardrails'}</p>
                </div>
            </div>
            <p class="font-sketch text-center" style="color: var(--electric-yellow); margin-top: 1rem; text-align: center;">
                ⚡ ${isHi ? 'शून्य मनगढ़ंत उत्तर · 100% साक्ष्य-आधारित' : 'Zero Hallucinations · Fully Evidence Grounded.'}
            </p>
        </div>
    `;
}
