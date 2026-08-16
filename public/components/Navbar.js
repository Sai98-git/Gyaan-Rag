export function renderNavbar(container, activeLang, onLanguageChange) {
    const brandText = activeLang === 'hi' ? 'ज्ञान' : 'Gyaān';
    
    container.innerHTML = `
        <div class="brutalist-nav">
            <a href="#" class="nav-brand">${brandText} <span>RAG</span></a>
            <ul class="nav-links">
                <li><a href="#query-container" class="nav-link">${activeLang === 'hi' ? 'पूछो' : 'Ask'}</a></li>
                <li><a href="#pipeline-container" class="nav-link">${activeLang === 'hi' ? 'कैसे काम करता है' : 'How it works'}</a></li>
                <li id="lang-selector-container"></li>
            </ul>
        </div>
    `;
    
    // Mount Language Selector inside navbar
    const langContainer = container.querySelector("#lang-selector-container");
    renderLanguageSelector(langContainer, activeLang, onLanguageChange);
}

function renderLanguageSelector(container, activeLang, onLanguageChange) {
    container.innerHTML = `
        <div class="lang-selector">
            <button class="lang-btn ${activeLang === 'hi' ? 'active' : ''}" data-lang="hi">हिन्दी</button>
            <button class="lang-btn ${activeLang === 'en' ? 'active' : ''}" data-lang="en">EN</button>
        </div>
    `;
    
    container.querySelectorAll(".lang-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const lang = btn.getAttribute("data-lang");
            onLanguageChange(lang);
        });
    });
}
