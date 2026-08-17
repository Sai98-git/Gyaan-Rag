/**
 * VoiceInput.js — Live browser microphone recording, visualizer,
 * multi-stage pipeline state progression, and audio submission.
 */

let mediaRecorder = null;
let audioChunks = [];
let recordingInterval = null;
let recordingSeconds = 0;
let audioContext = null;
let analyser = null;
let animationFrameId = null;

export function renderVoiceInput(container, activeLang, onVoiceSubmit) {
    const isHi = activeLang === 'hi';
    const mainBtnText = isHi ? '🎙 बोलकर पूछें (Ask by Voice)' : '🎙 ASK BY VOICE';
    const subHint = isHi 
        ? 'माइक बटन दबाएं और हिंदी या अंग्रेजी में अपना सवाल बोलें' 
        : 'Click microphone and speak your question in Hindi or English';
    const stopBtnText = isHi ? '⏹ रोकें और उत्तर पाएं' : '⏹ STOP & GET ANSWER';
    const cancelBtnText = isHi ? '✖ रद्द करें' : '✖ CANCEL';

    container.innerHTML = `
        <div class="sketch-card voice-box" id="voice-card">
            <div class="voice-header">
                <span class="voice-badge font-display">${isHi ? 'वॉयस-इनेबल्ड RAG' : 'VOICE-FIRST RAG INTERFACE'}</span>
                <span class="font-sketch" style="color: var(--hot-pink); font-size: 1.1rem;">
                    ${isHi ? '⚡ सर्वम AI Indic स्पीच-टू-टेक्स्ट' : '⚡ Powered by Sarvam AI Indic STT'}
                </span>
            </div>

            <!-- IDLE STATE: Big Mic CTA -->
            <div id="voice-idle-state" class="voice-state-view">
                <button id="voice-start-btn" class="brutalist-voice-btn font-display" type="button">
                    <span class="mic-icon">🎙</span>
                    <span class="mic-label">${mainBtnText}</span>
                </button>
                <p class="voice-hint font-sketch">${subHint}</p>
            </div>

            <!-- RECORDING STATE: Live Timer & Pulsing Waveform -->
            <div id="voice-recording-state" class="voice-state-view" style="display: none;">
                <div class="recording-indicator">
                    <div class="recording-dot"></div>
                    <span class="recording-label font-display">${isHi ? 'सुन रहा हूँ...' : 'LISTENING...'}</span>
                    <span id="recording-timer" class="recording-timer font-display">00:00</span>
                </div>

                <!-- Live Audio Visualizer Canvas -->
                <canvas id="voice-visualizer" width="400" height="60" class="visualizer-canvas"></canvas>

                <div class="recording-actions">
                    <button id="voice-stop-btn" class="brutalist-btn btn-stop font-display" type="button">
                        ${stopBtnText}
                    </button>
                    <button id="voice-cancel-btn" class="brutalist-btn btn-cancel font-display" type="button">
                        ${cancelBtnText}
                    </button>
                </div>
            </div>

            <!-- PROCESSING PIPELINE STEP TRACKER -->
            <div id="voice-pipeline-tracker" class="voice-state-view" style="display: none;">
                <div class="pipeline-progress-box">
                    <div class="pipeline-step-item" id="p-step-stt">
                        <span class="step-icon">📝</span>
                        <span class="step-text font-display">${isHi ? '1. आवाज़ को टेक्स्ट में बदला जा रहा है (Sarvam STT)...' : '1. Transcribing Speech (Sarvam STT)...'}</span>
                        <span class="step-status">⏳</span>
                    </div>
                    <div class="pipeline-step-item" id="p-step-retrieval">
                        <span class="step-icon">🔍</span>
                        <span class="step-text font-display">${isHi ? '2. ज्ञानकोष से साक्ष्य खोजे जा रहे हैं (E5 Vector Search)...' : '2. Retrieving Knowledge (E5 Vector Search)...'}</span>
                        <span class="step-status">⏳</span>
                    </div>
                    <div class="pipeline-step-item" id="p-step-gen">
                        <span class="step-icon">🧠</span>
                        <span class="step-text font-display">${isHi ? '3. सत्यापित उत्तर तैयार किया जा रहा है (Sarvam AI)...' : '3. Generating Grounded Answer (Sarvam AI)...'}</span>
                        <span class="step-status">⏳</span>
                    </div>
                </div>
            </div>

            <!-- ERROR DISPLAY -->
            <div id="voice-error-box" class="voice-error-box" style="display: none;">
                <p id="voice-error-text" class="voice-error-text font-display"></p>
            </div>
        </div>
    `;

    const startBtn = container.querySelector("#voice-start-btn");
    const stopBtn = container.querySelector("#voice-stop-btn");
    const cancelBtn = container.querySelector("#voice-cancel-btn");

    startBtn.addEventListener("click", () => startRecording(container, activeLang, onVoiceSubmit));
    stopBtn.addEventListener("click", () => stopRecording(container, activeLang, onVoiceSubmit));
    cancelBtn.addEventListener("click", () => cancelRecording(container, activeLang));
}

async function startRecording(container, activeLang, onVoiceSubmit) {
    const idleView = container.querySelector("#voice-idle-state");
    const recView = container.querySelector("#voice-recording-state");
    const errorBox = container.querySelector("#voice-error-box");
    const timerDisplay = container.querySelector("#recording-timer");
    const canvas = container.querySelector("#voice-visualizer");

    errorBox.style.display = "none";

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showVoiceError(container, activeLang === 'hi' 
            ? 'आपका ब्राउज़र ऑडियो रिकॉर्डिंग का समर्थन नहीं करता।' 
            : 'Your browser does not support audio recording.');
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Determine supported audio mime type
        let mimeType = 'audio/webm';
        if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
            mimeType = 'audio/webm;codecs=opus';
        } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
            mimeType = 'audio/mp4';
        } else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
            mimeType = 'audio/ogg;codecs=opus';
        }

        mediaRecorder = new MediaRecorder(stream, { mimeType });
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        // Initialize AudioContext visualizer
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const sourceNode = audioContext.createMediaStreamSource(stream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 64;
            sourceNode.connect(analyser);
            startVisualizer(canvas);
        } catch (visErr) {
            console.warn("Visualizer init failed (non-fatal):", visErr);
        }

        mediaRecorder.start(250); // Collect slice every 250ms

        idleView.style.display = "none";
        recView.style.display = "flex";

        // Start timer
        recordingSeconds = 0;
        timerDisplay.textContent = "00:00";
        clearInterval(recordingInterval);
        recordingInterval = setInterval(() => {
            recordingSeconds++;
            const mins = String(Math.floor(recordingSeconds / 60)).padStart(2, '0');
            const secs = String(recordingSeconds % 60).padStart(2, '0');
            timerDisplay.textContent = `${mins}:${secs}`;

            // Auto-stop at 30 seconds max
            if (recordingSeconds >= 30) {
                stopRecording(container, activeLang, onVoiceSubmit);
            }
        }, 1000);

    } catch (err) {
        console.error("Microphone access error:", err);
        let msg = activeLang === 'hi'
            ? 'माइक्रोफ़ोन की अनुमति अस्वीकृत या उपलब्ध नहीं है।'
            : 'Microphone permission was denied or is unavailable.';
        showVoiceError(container, msg);
    }
}

function stopRecording(container, activeLang, onVoiceSubmit) {
    if (!mediaRecorder || mediaRecorder.state === "inactive") return;

    clearInterval(recordingInterval);
    stopVisualizer();

    const recView = container.querySelector("#voice-recording-state");
    const tracker = container.querySelector("#voice-pipeline-tracker");

    recView.style.display = "none";
    tracker.style.display = "block";
    setPipelineStep(container, "stt", "active");

    mediaRecorder.onstop = async () => {
        // Stop all audio stream tracks
        if (mediaRecorder.stream) {
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }

        const mimeType = mediaRecorder.mimeType || 'audio/webm';
        const audioBlob = new Blob(audioChunks, { type: mimeType });

        if (audioBlob.size === 0) {
            showVoiceError(container, activeLang === 'hi' ? 'खाली ऑडियो रिकॉर्डिंग।' : 'Empty audio recording.');
            resetVoiceUI(container);
            return;
        }

        const fileExt = mimeType.includes('mp4') ? 'mp4' : (mimeType.includes('ogg') ? 'ogg' : 'webm');
        const filename = `recording_${Date.now()}.${fileExt}`;

        // Send to backend
        try {
            await onVoiceSubmit(audioBlob, filename, mimeType, (stage) => {
                if (stage === 'retrieval') {
                    setPipelineStep(container, "stt", "done");
                    setPipelineStep(container, "retrieval", "active");
                } else if (stage === 'generation') {
                    setPipelineStep(container, "retrieval", "done");
                    setPipelineStep(container, "gen", "active");
                } else if (stage === 'done') {
                    setPipelineStep(container, "gen", "done");
                }
            });
        } finally {
            resetVoiceUI(container);
        }
    };

    mediaRecorder.stop();
}

function cancelRecording(container, activeLang) {
    clearInterval(recordingInterval);
    stopVisualizer();

    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.onstop = null;
        if (mediaRecorder.stream) {
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        mediaRecorder.stop();
    }

    audioChunks = [];
    resetVoiceUI(container);
}

function resetVoiceUI(container) {
    const idleView = container.querySelector("#voice-idle-state");
    const recView = container.querySelector("#voice-recording-state");
    const tracker = container.querySelector("#voice-pipeline-tracker");

    if (idleView) idleView.style.display = "flex";
    if (recView) recView.style.display = "none";
    if (tracker) tracker.style.display = "none";

    // Reset steps
    ["stt", "retrieval", "gen"].forEach(step => {
        const el = container.querySelector(`#p-step-${step}`);
        if (el) {
            el.classList.remove("active", "done");
            const statusEl = el.querySelector(".step-status");
            if (statusEl) statusEl.textContent = "⏳";
        }
    });
}

function setPipelineStep(container, stepName, state) {
    const el = container.querySelector(`#p-step-${stepName}`);
    if (!el) return;

    if (state === "active") {
        el.classList.add("active");
        el.classList.remove("done");
        const statusEl = el.querySelector(".step-status");
        if (statusEl) statusEl.textContent = "🔄";
    } else if (state === "done") {
        el.classList.remove("active");
        el.classList.add("done");
        const statusEl = el.querySelector(".step-status");
        if (statusEl) statusEl.textContent = "✓";
    }
}

function showVoiceError(container, message) {
    const errorBox = container.querySelector("#voice-error-box");
    const errorText = container.querySelector("#voice-error-text");
    if (errorBox && errorText) {
        errorText.textContent = `⚠️ ${message}`;
        errorBox.style.display = "block";
    }
}

function startVisualizer(canvas) {
    if (!canvas || !analyser) return;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
        animationFrameId = requestAnimationFrame(draw);
        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#00140d';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLength) * 2.2;
        let barHeight;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            barHeight = (dataArray[i] / 255) * canvas.height * 0.85;
            ctx.fillStyle = '#f5ff00'; // Electric yellow
            ctx.fillRect(x, canvas.height - barHeight - 4, barWidth - 2, barHeight + 4);
            x += barWidth + 2;
        }
    }
    draw();
}

function stopVisualizer() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (audioContext && audioContext.state !== 'closed') {
        try {
            audioContext.close();
        } catch (e) {}
    }
}
