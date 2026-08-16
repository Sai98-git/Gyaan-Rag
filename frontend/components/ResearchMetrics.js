export function renderResearchMetrics(container, activeLang) {
    const titleText = activeLang === 'hi' ? 'शोध मेट्रिक्स' : 'RESEARCH METRICS';
    const subtitleText = activeLang === 'hi'
        ? 'हिंदी पुनः प्राप्ति मूल्यांकन · 104 प्रश्न'
        : 'HINDI RETRIEVAL EVALUATION · 104 QUERIES';
    const evalNote = activeLang === 'hi'
        ? 'हिंदी प्रश्न उप-समूह पर मूल्यांकन परिणाम'
        : 'Evaluation results from the Hindi query split of MSMARCO-XI';

    const metrics = [
        { label: 'R@1',   value: '41.3%', note: 'Recall at 1' },
        { label: 'R@5',   value: '92.3%', note: 'Recall at 5' },
        { label: 'R@10',  value: '97.1%', note: 'Recall at 10' },
        { label: 'MRR@10', value: '62.5%', note: 'Mean Reciprocal Rank' },
    ];

    container.innerHTML = `
        <div class="sketch-card"
             style="border-color: var(--dark-black); box-shadow: 8px 8px 0px var(--electric-yellow);">

            <!-- Header -->
            <div style="margin-bottom: 0.4rem;">
                <h2 class="font-display"
                    style="color: var(--dark-black); font-size: 2rem; line-height: 1.1;">
                    ${titleText}
                </h2>
                <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase;
                            letter-spacing: 0.1em; color: var(--hot-pink); margin-top: 0.4rem;">
                    ${subtitleText}
                </div>
            </div>

            <!-- Eval note -->
            <p class="font-sketch"
               style="color: rgba(0,20,13,0.55); font-size: 1.15rem; margin-bottom: 2rem;
                      border-bottom: 2px dashed rgba(0,20,13,0.2); padding-bottom: 1rem;">
                ${evalNote}
            </p>

            <!-- Main metric cards -->
            <div class="metrics-grid" style="margin-bottom: 2rem;">
                ${metrics.map(m => `
                    <div class="metric-card">
                        <div class="metric-value font-display">${m.value}</div>
                        <div class="metric-label font-display">${m.label}</div>
                        <div class="metric-note">${m.note}</div>
                    </div>
                `).join('')}
            </div>

            <!-- Secondary stats row -->
            <div class="metrics-secondary-row">
                <div class="metrics-stat-block">
                    <div class="metrics-stat-label font-display">
                        ${activeLang === 'hi' ? 'औसत पुनः प्राप्ति विलंबता' : 'MEAN RETRIEVAL LATENCY'}
                    </div>
                    <div class="metrics-stat-value font-display">88.81 ms</div>
                </div>
                <div class="metrics-stat-block">
                    <div class="metrics-stat-label font-display">
                        ${activeLang === 'hi' ? 'इंडेक्स आकार' : 'INDEX SIZE'}
                    </div>
                    <div class="metrics-stat-value font-display">9.20 MB</div>
                </div>
            </div>

            <!-- Method footnote -->
            <div style="margin-top: 2rem; padding: 1rem;
                        border: 2px solid var(--dark-black); border-radius: 6px;
                        background: rgba(0,20,13,0.05);">
                <div style="display: flex; flex-wrap: wrap; gap: 1.5rem;">
                    <div>
                        <span style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
                                     letter-spacing: 0.08em; color: var(--hot-pink);">
                            ${activeLang === 'hi' ? 'पुनः प्राप्ति:' : 'RETRIEVAL:'}
                        </span>
                        <span style="font-size: 0.9rem; font-weight: 600; color: var(--dark-black);
                                     margin-left: 6px;">
                            Semantic Chunker + Dense Retrieval
                        </span>
                    </div>
                    <div>
                        <span style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
                                     letter-spacing: 0.08em; color: var(--hot-pink);">
                            ${activeLang === 'hi' ? 'एम्बेडिंग:' : 'EMBEDDING:'}
                        </span>
                        <span style="font-size: 0.9rem; font-weight: 600; color: var(--dark-black);
                                     margin-left: 6px;">
                            Multilingual E5-small
                        </span>
                    </div>
                </div>
            </div>

        </div>
    `;
}
