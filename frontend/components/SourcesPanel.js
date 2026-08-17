export function renderSourcesPanel(container, sources, activeLang) {
    if (!sources || sources.length === 0) {
        container.innerHTML = "";
        return;
    }

    const titleText = activeLang === 'hi'
        ? `सत्यापन स्रोत एवं स्कोर (EVIDENCE SOURCES & SCORES · ${sources.length})`
        : `RETRIEVED DATASET SOURCES & SCORES · ${sources.length}`;

    let gridHtml = "";

    sources.forEach((source, idx) => {
        const chunkId = source.chunk_id;
        const normScore = typeof source.score === "number" ? source.score.toFixed(3) : "—";
        const denseScore = source.dense_score !== undefined ? source.dense_score.toFixed(3) : (source.metadata?.dense_similarity !== undefined ? source.metadata.dense_similarity.toFixed(3) : "—");
        const bm25Score = source.bm25_score !== undefined ? source.bm25_score.toFixed(3) : (source.metadata?.bm25_relevance !== undefined ? source.metadata.bm25_relevance.toFixed(3) : "—");
        const rrfScore = source.rrf_score !== undefined ? source.rrf_score.toFixed(4) : "—";

        const preview = source.text
            || source.preview
            || (source.metadata && source.metadata.text && source.metadata.text.slice(0, 250))
            || "No passage preview available.";

        const metaCopy = { ...source.metadata };
        delete metaCopy.text;
        const metadataJson = JSON.stringify(metaCopy, null, 2);

        gridHtml += `
            <div class="source-card">
                <div class="source-header" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:4px;">
                    <span class="source-id font-display">SOURCE ${String(idx + 1).padStart(2, '0')} · [${chunkId}]</span>
                    <span class="source-score font-display" style="color:var(--deep-green); font-size:0.88rem;">
                        RRF Confidence: ${normScore}
                    </span>
                </div>

                <div style="display:flex; gap:0.5rem; margin:0.4rem 0; flex-wrap:wrap; font-size:0.78rem;">
                    <span style="background:rgba(0,107,60,0.1); border:1px solid var(--deep-green); padding:2px 6px; border-radius:4px;">
                        <strong>Dense E5:</strong> ${denseScore}
                    </span>
                    <span style="background:rgba(0,32,96,0.1); border:1px solid #002060; padding:2px 6px; border-radius:4px;">
                        <strong>BM25:</strong> ${bm25Score}
                    </span>
                    <span style="background:rgba(255,225,53,0.3); border:1px solid #c9b000; padding:2px 6px; border-radius:4px;">
                        <strong>RRF Score:</strong> ${rrfScore}
                    </span>
                </div>

                <div class="source-body" style="font-size:0.92rem; line-height:1.6; margin-top:0.4rem;">
                    "${preview.trim()}"
                </div>

                <div style="margin-top:0.6rem; display:flex; gap:0.5rem; flex-wrap:wrap; align-items:center;">
                    <button class="metadata-trigger font-display" data-idx="${idx}" style="font-size:0.75rem; padding:3px 8px;">
                        [+] VIEW RAW METADATA →
                    </button>
                </div>

                <div id="metadata-block-${idx}" class="source-metadata-block" style="display:none; margin-top:0.5rem;">
                    <pre style="font-size:0.75rem; background:#f4f4f4; padding:6px; border-radius:4px;">${metadataJson}</pre>
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
                btn.textContent = "[-] HIDE RAW METADATA ↑";
            } else {
                block.style.display = "none";
                btn.textContent = "[+] VIEW RAW METADATA →";
            }
        });
    });
}
