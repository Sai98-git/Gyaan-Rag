export function renderQueryInput(container, activeLang, onSubmit) {
    const isHi = activeLang === 'hi';
    const toggleText = isHi ? '⌨️ या कीबोर्ड से टाइप करें (Text Fallback)' : '⌨️ Or Type Your Query (Text Fallback)';
    const labelText = isHi ? 'लिखित प्रश्न दर्ज करें:' : 'Type query manually:';
    const buttonText = isHi ? 'पूछो →' : 'ASK →';
    const placeholderText = isHi 
        ? 'यहाँ अपना सवाल लिखें (उदा. कॉर्पोरेशन क्या है?)...' 
        : 'Type your question here (e.g. what is a corporation?)...';
        
    container.innerHTML = `
        <div class="text-fallback-wrapper">
            <details class="text-fallback-accordion">
                <summary class="font-display text-fallback-summary">
                    ${toggleText}
                </summary>
                <div class="sketch-card query-box" style="margin-top: 1rem;">
                    <label for="query-input" class="query-label font-display">
                        <span class="marker-yellow">${labelText}</span>
                    </label>
                    <div class="input-wrapper">
                        <input 
                            type="text" 
                            id="query-input" 
                            class="brutalist-input" 
                            placeholder="${placeholderText}"
                            autocomplete="off"
                        />
                        <button id="query-submit-btn" class="brutalist-btn font-display" type="button">${buttonText}</button>
                    </div>
                    <p class="font-sketch" style="color: var(--hot-pink); margin-top: 5px; margin-left: 5px;">
                        ${isHi ? '✍️ सीधे MSMARCO-XI इंडेक्स से खोजें' : '✍️ Queries directly against indexed knowledge'}
                    </p>
                </div>
            </details>
        </div>
    `;
    
    const input = container.querySelector("#query-input");
    const submitBtn = container.querySelector("#query-submit-btn");
    
    if (submitBtn && input) {
        const triggerSubmit = () => {
            const query = input.value.trim();
            if (query) {
                onSubmit(query);
            }
        };
        
        submitBtn.addEventListener("click", triggerSubmit);
        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                triggerSubmit();
            }
        });
    }
}

export function setQueryInputLoading(container, isLoading, activeLang) {
    const submitBtn = container.querySelector("#query-submit-btn");
    const input = container.querySelector("#query-input");
    if (!submitBtn || !input) return;
    
    if (isLoading) {
        submitBtn.disabled = true;
        input.disabled = true;
        submitBtn.textContent = activeLang === 'hi' ? 'खोज रहा हूँ...' : 'SEARCHING...';
        submitBtn.style.opacity = "0.7";
        submitBtn.style.cursor = "not-allowed";
    } else {
        submitBtn.disabled = false;
        input.disabled = false;
        submitBtn.textContent = activeLang === 'hi' ? 'पूछो →' : 'ASK →';
        submitBtn.style.opacity = "1";
        submitBtn.style.cursor = "pointer";
    }
}
