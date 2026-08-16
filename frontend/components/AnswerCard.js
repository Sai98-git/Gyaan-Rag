// Strip the trailing mock-provider notice from the answer string.
// This note is useful in development logs but must not appear in the user-facing card.
function stripMockNote(text) {
    return text
        .replace(/\(Note: This is the mock offline provider\.[^)]*\)/gi, "")
        .replace(/Note: This is the mock offline provider\.[^\n]*/gi, "")
        .trim();
}

export function renderAnswerCard(container, data, activeLang, queryText) {
    const { answer: rawAnswer, guard_triggered, guard_reason, retrieval, generation, provider } = data;

    // Remove mock provider notice from display text only (backend answer unchanged)
    const answer = stripMockNote(rawAnswer);

    const isAbstained = guard_triggered
        || answer.toLowerCase().includes("don't have enough information")
        || answer.toLowerCase().includes("पर्याप्त नहीं");

    const cardClass = isAbstained ? "sketch-card abstain-card" : "sketch-card";

    let statusLabel;
    if (guard_triggered) {
        statusLabel = "GROUNDING GUARD · ABSTAINED";
    } else if (isAbstained) {
        statusLabel = "ABSTAINED · INSUFFICIENT EVIDENCE";
    } else {
        statusLabel = "GROUNDED ANSWER";
    }

    const tagBg = isAbstained ? "var(--hot-pink)" : "var(--electric-yellow)";
    const tagColor = isAbstained ? "white" : "var(--dark-black)";

    const retLatency = retrieval?.latency_ms?.toFixed(1) ?? "—";
    const genLatency = generation?.latency_ms?.toFixed(1) ?? "—";
    const providerLabel = provider?.toUpperCase() ?? "—";

    container.innerHTML = `
        <div class="${cardClass}">

            <!-- Status tag row -->
            <div style="display:flex; justify-content:space-between; align-items:flex-start;
                        flex-wrap:wrap; gap:0.5rem; margin-bottom:1.75rem;">
                <span class="answer-tag font-display"
                      style="background:${tagBg}; color:${tagColor}; border-color:var(--dark-black);">
                    ${statusLabel}
                </span>
                <span class="font-sketch"
                      style="color:var(--hot-pink); font-size:1rem;">
                    ${activeLang === 'hi' ? 'स्रोत-आधारित उत्तर' : 'Source-backed response'}
                </span>
            </div>

            <!-- Query echo (smaller, secondary) -->
            ${queryText ? `
            <div style="border-left:4px solid var(--hot-pink); padding-left:1rem; margin-bottom:1.75rem;">
                <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase;
                            color:var(--hot-pink); letter-spacing:0.08em; margin-bottom:4px;">
                    ${activeLang === 'hi' ? 'प्रश्न' : 'QUERY'}
                </div>
                <p style="font-size:1rem; font-weight:600; color:var(--dark-black); line-height:1.5;">
                    ${queryText}
                </p>
            </div>
            ` : ''}

            <!-- ANSWER heading -->
            ${!isAbstained ? `
            <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase;
                        letter-spacing:0.1em; color:var(--dark-black); margin-bottom:0.6rem;
                        border-bottom:2px solid var(--dark-black); padding-bottom:4px;">
                ${activeLang === 'hi' ? 'उत्तर' : 'ANSWER'}
            </div>
            ` : ''}

            <!-- Main answer text — the visual focal point -->
            <div class="answer-text"
                 style="color:var(--dark-black); font-size:1.22rem; line-height:1.85;
                        max-width:72ch; margin-bottom:${isAbstained ? '1.5rem' : '1rem'};">
                ${answer.replace(/\n/g, "<br>")}
            </div>

            <!-- Grounding badge (only when answer is grounded) -->
            ${!isAbstained ? `
            <div style="display:inline-flex; align-items:center; gap:0.4rem;
                        background:rgba(0,107,60,0.1); border:2px solid var(--deep-green);
                        border-radius:6px; padding:4px 12px; margin-bottom:1.5rem;">
                <span style="color:var(--deep-green); font-weight:900; font-size:1rem;">✓</span>
                <span style="font-size:0.82rem; font-weight:700; text-transform:uppercase;
                             letter-spacing:0.07em; color:var(--deep-green);">
                    ${activeLang === 'hi' ? 'पुनः प्राप्त स्रोतों में आधारित' : 'Grounded in retrieved context'}
                </span>
            </div>
            ` : ''}

            <!-- Guard abstention note -->
            ${guard_triggered && guard_reason ? `
            <div style="margin-bottom:1.5rem; padding:0.75rem 1rem;
                        border:2px dashed var(--hot-pink);
                        background:rgba(255,0,128,0.06); border-radius:8px;">
                <p style="color:var(--hot-pink); font-weight:700; font-size:0.82rem;
                           text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">
                    ⚠ Limited evidence
                </p>
                <p style="font-size:0.9rem; color:var(--dark-black);">
                    The retrieved passages did not contain sufficient information to answer
                    this question confidently.
                </p>
            </div>
            ` : ''}

            <!-- Latency / provider footer — de-emphasised -->
            <div class="latency-annotation font-sketch"
                 style="margin-top:0.5rem; padding-top:1rem;
                        border-top:2px dashed rgba(0,20,13,0.25);
                        display:flex; flex-wrap:wrap; gap:1.5rem;
                        font-size:1.1rem; color:rgba(0,20,13,0.55);">
                <span>Retrieval: ${retLatency} ms</span>
                <span>Generation: ${genLatency} ms</span>
                <span>Provider: ${providerLabel}</span>
            </div>
        </div>
    `;
}
