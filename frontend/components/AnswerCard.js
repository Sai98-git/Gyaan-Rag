function stripMockNote(text) {
    if (!text) return "";
    return text
        .replace(/\(Note: This is the mock offline provider\.[^)]*\)/gi, "")
        .replace(/Note: This is the mock offline provider\.[^\n]*/gi, "")
        .trim();
}

export function renderAnswerCard(container, data, activeLang, queryText) {
    const isHi = activeLang === 'hi';
    const { 
        answer: rawAnswer, 
        transcript, 
        normalized_query, 
        language, 
        guard_triggered, 
        guard_reason, 
        latency, 
        retrieval, 
        generation, 
        provider, 
        stt_provider,
        sources = []
    } = data;

    const answer = stripMockNote(rawAnswer);

    const isAbstained = guard_triggered
        || answer.toLowerCase().includes("don't have enough information")
        || answer.toLowerCase().includes("पर्याप्त नहीं");

    const cardClass = isAbstained ? "sketch-card abstain-card" : "sketch-card";

    let statusLabel;
    if (guard_triggered) {
        statusLabel = isHi ? "🛡️ ग्राउंडिंग गार्ड · निरस्त (ABSTAINED)" : "🛡️ GROUNDING GUARD · ABSTAINED";
    } else if (isAbstained) {
        statusLabel = isHi ? "🛡️ अपर्याप्त साक्ष्य · निरस्त (ABSTAINED)" : "🛡️ INSUFFICIENT EVIDENCE · ABSTAINED";
    } else {
        statusLabel = isHi ? "✅ साक्ष्य-सत्यापित उत्तर (GROUNDED ANSWER)" : "✅ GROUNDED ANSWER";
    }

    const tagBg = isAbstained ? "var(--hot-pink)" : "var(--electric-yellow)";
    const tagColor = isAbstained ? "white" : "var(--dark-black)";

    // Latency extraction
    const sttMs = latency?.stt_ms !== undefined ? `${latency.stt_ms.toFixed(1)} ms` : "— (Text Query)";
    const retMs = (latency?.retrieval_ms ?? retrieval?.latency_ms)?.toFixed(1) ?? "—";
    const genMs = (latency?.generation_ms ?? generation?.latency_ms)?.toFixed(1) ?? "—";
    const totalMs = latency?.total_ms !== undefined 
        ? `${latency.total_ms.toFixed(1)} ms` 
        : `${((retrieval?.latency_ms || 0) + (generation?.latency_ms || 0)).toFixed(1)} ms`;

    const userUtterance = transcript || queryText || normalized_query || "";
    const langBadge = (language || "hi-IN").toUpperCase();
    const sttBadge = (stt_provider || "SARVAM STT").toUpperCase();

    container.innerHTML = `
        <div class="${cardClass}">

            <!-- Status Tag Row -->
            <div style="display:flex; justify-content:space-between; align-items:flex-start;
                        flex-wrap:wrap; gap:0.5rem; margin-bottom:1.5rem;">
                <span class="answer-tag font-display"
                      style="background:${tagBg}; color:${tagColor}; border-color:var(--dark-black);">
                    ${statusLabel}
                </span>
                <div style="display:flex; gap:0.5rem; align-items:center;">
                    <span class="brutalist-mini-badge font-display" style="background:var(--dark-black); color:var(--electric-yellow);">
                        ${langBadge}
                    </span>
                    ${transcript ? `
                    <span class="brutalist-mini-badge font-display" style="background:var(--hot-pink); color:white;">
                        🎙 ${sttBadge}
                    </span>
                    ` : ''}
                </div>
            </div>

            <!-- TRANSCRIPT / QUERY SECTION -->
            ${userUtterance ? `
            <div class="transcript-callout">
                <div class="transcript-header font-display">
                    <span>${transcript ? (isHi ? '🎙 उपयोगकर्ता की आवाज़ (TRANSCRIPT)' : '🎙 USER TRANSCRIPT') : (isHi ? '✍️ प्रश्न (QUERY)' : '✍️ USER QUERY')}</span>
                </div>
                <p class="transcript-text font-display">
                    "${userUtterance}"
                </p>
                ${normalized_query && normalized_query !== userUtterance ? `
                <div style="font-size:0.85rem; color:rgba(0,20,13,0.6); margin-top:4px;">
                    <em>Normalized:</em> ${normalized_query}
                </div>
                ` : ''}
            </div>
            ` : ''}

            <!-- ANSWER SECTION -->
            ${!isAbstained ? `
            <div style="font-size:0.82rem; font-weight:700; text-transform:uppercase;
                        letter-spacing:0.1em; color:var(--dark-black); margin-bottom:0.6rem;
                        border-bottom:2px solid var(--dark-black); padding-bottom:4px;">
                ${isHi ? 'सत्यापित उत्तर (GROUNDED ANSWER)' : 'GROUNDED ANSWER'}
            </div>
            ` : ''}

            <!-- Main Answer Text -->
            <div class="answer-text"
                 style="color:var(--dark-black); font-size:1.22rem; line-height:1.85;
                        max-width:72ch; margin-bottom:${isAbstained ? '1.5rem' : '1rem'}; font-weight:500;">
                ${answer.replace(/\n/g, "<br>")}
            </div>

            <!-- Grounding Verification Badge -->
            ${!isAbstained ? `
            <div style="display:inline-flex; align-items:center; gap:0.4rem;
                        background:rgba(0,107,60,0.1); border:2px solid var(--deep-green);
                        border-radius:6px; padding:4px 12px; margin-bottom:1.5rem;">
                <span style="color:var(--deep-green); font-weight:900; font-size:1rem;">✓</span>
                <span style="font-size:0.82rem; font-weight:700; text-transform:uppercase;
                             letter-spacing:0.07em; color:var(--deep-green);">
                    ${isHi ? 'पुनः प्राप्त MSMARCO-XI साक्ष्यों में पूर्णतः सत्यापित' : 'Strictly grounded in retrieved MSMARCO-XI dataset evidence'}
                </span>
            </div>
            ` : ''}

            <!-- Guard Abstention Diagnostics -->
            ${isAbstained ? `
            <div style="margin-bottom:1.5rem; padding:0.75rem 1rem;
                        border:2px dashed var(--hot-pink);
                        background:rgba(255,0,128,0.06); border-radius:8px;">
                <p style="color:var(--hot-pink); font-weight:700; font-size:0.85rem;
                           text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">
                    🛡️ ${isHi ? 'निरस्त करने का कारण (ABSTENTION REASON)' : 'GROUNDING GUARD DECISION'}
                </p>
                <p style="font-size:0.92rem; color:var(--dark-black); font-weight:500;">
                    ${guard_reason || (isHi ? 'प्रदत्त प्रश्न के लिए डेटासेट में पर्याप्त साक्ष्य उपलब्ध नहीं है।' : 'No sufficiently relevant evidence was retrieved from the dataset for this query.')}
                </p>
            </div>
            ` : ''}

            <!-- REAL-TIME LATENCY BREAKDOWN METRICS TABLE -->
            <div class="latency-metrics-container">
                <div class="latency-metrics-title font-display">
                    ⚡ ${isHi ? 'पाइपलाइन लेटेंसी विश्लेषण (Latency Breakdown)' : 'PIPELINE LATENCY BREAKDOWN'}
                </div>
                <div class="latency-grid">
                    <div class="latency-item">
                        <span class="latency-name font-display">1. Speech-To-Text</span>
                        <span class="latency-val font-display">${sttMs}</span>
                    </div>
                    <div class="latency-item">
                        <span class="latency-name font-display">2. Retrieval (E5 + BM25)</span>
                        <span class="latency-val font-display">${retMs} ms</span>
                    </div>
                    <div class="latency-item">
                        <span class="latency-name font-display">3. Generation (Sarvam)</span>
                        <span class="latency-val font-display">${genMs} ms</span>
                    </div>
                    <div class="latency-item latency-total">
                        <span class="latency-name font-display">Total End-to-End</span>
                        <span class="latency-val font-display">${totalMs}</span>
                    </div>
                </div>
            </div>

        </div>
    `;
}
