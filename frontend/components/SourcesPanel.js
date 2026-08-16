export function renderSourcesPanel(container, sources, activeLang) {
    if (!sources || sources.length === 0) {
        container.innerHTML = "";
        return;
    }

    const titleText = activeLang === 'hi'
        ? `सत्यापन स्रोत · ${sources.length}`
        : `SOURCES · ${sources.length}`;

    let gridHtml = "";

    sources.forEach((source, idx) => {
        const chunkId = source.chunk_id;
        const score = typeof source.score === "number" ? source.score.toFixed(3) : "—";
        // Use preview field (passage text) if available; fall back gracefully
        const preview = source.preview
            || (source.metadata && source.metadata.text && source.metadata.text.slice(0, 200))
            || "No passage preview available.";

        // Build a clean metadata object — exclude 'text' to avoid duplication
        const metaCopy = { ...source.metadata };
        delete metaCopy.text;
        const metadataJson = JSON.stringify(metaCopy, null, 2);

        // Score colour: green when clearly confident, amber when borderline
        const scoreNum = parseFloat(score);
        const scoreColour = scoreNum >= 0.85
            ? "#006B3C"    /* deep-green: strongly retrieved */
            : scoreNum >= 0.78
                ? "#E07B00"    /* amber: borderline */
                : "var(--hot-pink)"; /* pink: low */

        gridHtml += `
            <div class="source-card">
                <div class="source-header">
                    <span class="source-id font-display">SOURCE ${String(idx + 1).padStart(2, '0')}</span>
                    <span class="source-score font-display"
                          style="color:${scoreColour};">
                        Retrieval similarity: ${score}
                    </span>
                </div>

                <div class="source-body">
                    "${preview.trim()}"
                </div>

                <div style="margin-top:0.75rem; display:flex; gap:1rem; flex-wrap:wrap; align-items:center;">
                    <button class="metadata-trigger font-display" data-idx="${idx}">
                        [+] VIEW PASSAGE →
                    </button>
                    ${source.metadata && source.metadata.language
                        ? `<span style="font-size:0.8rem; font-weight:bold; text-transform:uppercase;
                                        border:1px solid var(--dark-black); padding:2px 6px; border-radius:4px;">
                                LANG: ${source.metadata.language.toUpperCase()}
                           </span>`
                        : ""}
                </div>

                <div id="metadata-block-${idx}" class="source-metadata-block" style="display:none;">
                    <div style="margin-bottom:0.5rem;">
                        <strong>Chunk ID:</strong> ${chunkId}
                    </div>
                    <pre>${metadataJson}</pre>
                </div>
            </div>
        `;
    });

    container.innerHTML = `
        <div class="sources-title font-display">${titleText}</div>
        <div class="sources-grid">${gridHtml}</div>
    `;

    // Wire expand/collapse buttons
    container.querySelectorAll(".metadata-trigger").forEach(btn => {
        btn.addEventListener("click", () => {
            const idx = btn.getAttribute("data-idx");
            const block = container.querySelector(`#metadata-block-${idx}`);
            if (block.style.display === "none") {
                block.style.display = "block";
                btn.textContent = "[-] HIDE PASSAGE ↑";
            } else {
                block.style.display = "none";
                btn.textContent = "[+] VIEW PASSAGE →";
            }
        });
    });
}

