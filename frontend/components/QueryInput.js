export function renderQueryInput(container, activeLang, onSubmit) {
    const labelText = activeLang === 'hi' ? 'आप क्या जानना चाहते हैं?' : 'Ask the Knowledge Base...';
    const buttonText = activeLang === 'hi' ? 'पूछो →' : 'ASK →';
    const placeholderText = activeLang === 'hi' 
        ? 'यहाँ अपना सवाल लिखें (जैसे: कॉर्पोरेशन क्या है?)...' 
        : 'Type your question here (e.g. what is a corporation?)...';
        
    container.innerHTML = `
        <div class="sketch-card query-box">
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
                <button id="query-submit-btn" class="brutalist-btn font-display">${buttonText}</button>
            </div>
            <p class="font-sketch" style="color: var(--hot-pink); margin-top: 5px; margin-left: 5px;">
                ${activeLang === 'hi' ? '✍️ सीधे सोर्स से सत्यापित तथ्य' : '✍️ Structured factual query context'}
            </p>
        </div>
    `;
    
    const input = container.querySelector("#query-input");
    const submitBtn = container.querySelector("#query-submit-btn");
    
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

export function setQueryInputLoading(container, isLoading, activeLang) {
    const submitBtn = container.querySelector("#query-submit-btn");
    const input = container.querySelector("#query-input");
    if (!submitBtn || !input) return;
    
    if (isLoading) {
        submitBtn.disabled = true;
        input.disabled = true;
        submitBtn.textContent = activeLang === 'hi' ? 'सोच रहा हूँ...' : 'RETRIEVING...';
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

