export function renderWhyGyaanRag(container, activeLang) {
    const titleText = activeLang === 'hi' ? 'ज्ञान RAG क्यों?' : 'WHY GYAAN RAG?';

    container.innerHTML = `
        <div class="sketch-card-dark"
             style="box-shadow: 8px 8px 0px var(--hot-pink); border-color: var(--electric-yellow);">

            <!-- Section heading -->
            <h2 class="font-display"
                style="color: var(--electric-yellow); font-size: 2rem;
                       border-bottom: 3px solid var(--electric-yellow);
                       padding-bottom: 0.6rem; margin-bottom: 1.5rem;">
                ${titleText}
            </h2>

            <!-- One-line description -->
            <p style="font-size: 1.1rem; line-height: 1.75; color: var(--cream);
                      max-width: 70ch; margin-bottom: 2rem;">
                Gyaan RAG is an Indic-first Retrieval-Augmented Generation system designed to
                answer questions using <span class="marker-yellow">retrieved evidence</span>
                from its knowledge base — rather than relying purely on an LLM's internal knowledge.
            </p>

            <!-- Pipeline flow (compact, horizontal on desktop) -->
            <div style="margin-bottom: 2.5rem;">
                <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
                            letter-spacing: 0.1em; color: var(--hot-pink); margin-bottom: 1rem;">
                    ${activeLang === 'hi' ? 'पाइपलाइन प्रवाह' : 'PIPELINE FLOW'}
                </div>
                <div class="why-flow">
                    ${[
                        ["USER QUESTION", activeLang === 'hi' ? 'उपयोगकर्ता प्रश्न' : null],
                        ["RETRIEVE KNOWLEDGE", activeLang === 'hi' ? 'प्रासंगिक ज्ञान खोजें' : null],
                        ["BUILD CONTEXT", activeLang === 'hi' ? 'संदर्भ बनाएं' : null],
                        ["GENERATE ANSWER", activeLang === 'hi' ? 'उत्तर तैयार करें' : null],
                        ["GROUNDING GUARD", activeLang === 'hi' ? 'ग्राउंडिंग जांच' : null],
                        ["RESPOND / ABSTAIN", activeLang === 'hi' ? 'उत्तर / अस्वीकृति' : null],
                    ].map(([label, hi], i, arr) => `
                        <div class="why-flow-step font-display" style="${i === arr.length - 1
                            ? 'border-color: var(--electric-yellow); color: var(--electric-yellow);'
                            : ''}">
                            ${hi || label}
                        </div>
                        ${i < arr.length - 1 ? '<div class="why-flow-arrow">↓</div>' : ''}
                    `).join('')}
                </div>
            </div>

            <!-- Three feature cards -->
            <div class="why-features-grid">

                <div class="why-feature-card">
                    <div class="why-feature-icon font-display">01</div>
                    <div class="why-feature-title font-display">
                        ${activeLang === 'hi' ? 'भारतीय-भाषा प्रथम' : 'INDIC-FIRST'}
                    </div>
                    <p class="why-feature-body">
                        ${activeLang === 'hi'
                            ? 'हिंदी और अन्य भारतीय भाषाओं में प्रश्न पूछना और पुनः प्राप्त करना समर्थित है।'
                            : 'Supports Hindi and Indic-language querying and retrieval, using a multilingual embedding model trained on Indic text.'}
                    </p>
                </div>

                <div class="why-feature-card">
                    <div class="why-feature-icon font-display">02</div>
                    <div class="why-feature-title font-display">
                        ${activeLang === 'hi' ? 'साक्ष्य-आधारित' : 'EVIDENCE-GROUNDED'}
                    </div>
                    <p class="why-feature-body">
                        ${activeLang === 'hi'
                            ? 'उत्तर ज्ञान आधार के पुनः प्राप्त किए गए अंशों से उत्पन्न होते हैं।'
                            : 'Answers are generated from retrieved passages in the knowledge base. The LLM is grounded to the retrieved context — not its parametric memory.'}
                    </p>
                </div>

                <div class="why-feature-card">
                    <div class="why-feature-icon font-display">03</div>
                    <div class="why-feature-title font-display">
                        ${activeLang === 'hi' ? 'अस्वीकृति क्षमता' : 'ABSTENTION'}
                    </div>
                    <p class="why-feature-body">
                        ${activeLang === 'hi'
                            ? 'जब विश्वसनीय संदर्भ उपलब्ध नहीं होता, तो सिस्टम असमर्थित जानकारी देने से इनकार कर सकता है।'
                            : 'When the grounding guard determines that retrieved context is insufficient, the system abstains rather than generating unsupported information.'}
                    </p>
                </div>

            </div>

        </div>
    `;
}
