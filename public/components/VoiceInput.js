/**
 * VoiceInput.js — High-fidelity browser microphone recording, live audio visualizer,
 * native 16-bit 16kHz mono WAV PCM encoding, multi-stage pipeline state progression,
 * and audio submission to /api/voice.
 */

let audioContext = null;
let scriptProcessor = null;
let mediaStreamSource = null;
let mediaStream = null;
let pcmBuffers = [];
let recordingInterval = null;
let recordingSeconds = 0;
let analyser = null;
let animationFrameId = null;

// Fallback MediaRecorder state
let mediaRecorder = null;
let audioChunks = [];

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
                    ${isHi ? '⚡ सर्वम AI Indic स्पीच-टू-टेक्स्ट (16kHz PCM WAV)' : '⚡ Powered by Sarvam AI Indic STT (16kHz PCM WAV)'}
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
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            }
        });

        // Initialize AudioContext PCM recording
        pcmBuffers = [];
        const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
        audioContext = new AudioCtxClass({ sampleRate: 16000 });
        
        mediaStreamSource = audioContext.createMediaStreamSource(mediaStream);
        
        // Setup visualizer analyser
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 64;
        mediaStreamSource.connect(analyser);
        startVisualizer(canvas);

        // Setup ScriptProcessor for PCM extraction
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        scriptProcessor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            pcmBuffers.push(new Float32Array(inputData));
        };
        
        mediaStreamSource.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

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

async function stopRecording(container, activeLang, onVoiceSubmit) {
    clearInterval(recordingInterval);
    stopVisualizer();

    const recView = container.querySelector("#voice-recording-state");
    const tracker = container.querySelector("#voice-pipeline-tracker");

    recView.style.display = "none";
    tracker.style.display = "block";
    setPipelineStep(container, "stt", "active");

    // Clean up AudioContext & mediaStream
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor.onaudioprocess = null;
    }
    if (mediaStreamSource) {
        mediaStreamSource.disconnect();
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }

    const inputSampleRate = audioContext ? audioContext.sampleRate : 16000;
    if (audioContext && audioContext.state !== 'closed') {
        try { await audioContext.close(); } catch (e) {}
    }

    // Merge Float32 PCM buffers
    let totalLength = 0;
    for (const buf of pcmBuffers) {
        totalLength += buf.length;
    }

    if (totalLength === 0) {
        showVoiceError(container, activeLang === 'hi' ? 'खाली ऑडियो रिकॉर्डिंग।' : 'Empty audio recording.');
        resetVoiceUI(container);
        return;
    }

    const mergedFloat32 = new Float32Array(totalLength);
    let offset = 0;
    for (const buf of pcmBuffers) {
        mergedFloat32.set(buf, offset);
        offset += buf.length;
    }

    // Downsample to 16000Hz if needed
    const downsampledSamples = downsampleBuffer(mergedFloat32, inputSampleRate, 16000);

    // Encode to 16-bit mono 16kHz WAV Blob
    const wavBlob = encodeWAV(downsampledSamples, 16000);
    const filename = `recording_${Date.now()}.wav`;
    const mimeType = "audio/wav";

    // Send to backend
    try {
        await onVoiceSubmit(wavBlob, filename, mimeType, (stage) => {
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
    } catch (err) {
        console.error("Voice pipeline submission failed:", err);
        showVoiceError(container, err.message || 'Voice RAG failed');
    } finally {
        resetVoiceUI(container);
    }
}

function cancelRecording(container, activeLang) {
    clearInterval(recordingInterval);
    stopVisualizer();

    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor.onaudioprocess = null;
    }
    if (mediaStreamSource) {
        mediaStreamSource.disconnect();
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    if (audioContext && audioContext.state !== 'closed') {
        try { audioContext.close(); } catch (e) {}
    }

    pcmBuffers = [];
    resetVoiceUI(container);
}

function resetVoiceUI(container) {
    const idleView = container.querySelector("#voice-idle-state");
    const recView = container.querySelector("#voice-recording-state");
    const tracker = container.querySelector("#voice-pipeline-tracker");

    if (idleView) idleView.style.display = "flex";
    if (recView) recView.style.display = "none";
    if (tracker) tracker.style.display = "none";

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
            ctx.fillStyle = '#f5ff00';
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
}

// ─── WAV Encoding Helpers ───────────────────────────────────────────────────

function downsampleBuffer(buffer, inputSampleRate, outputSampleRate = 16000) {
    if (inputSampleRate === outputSampleRate) {
        return buffer;
    }
    const sampleRateRatio = inputSampleRate / outputSampleRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        let accum = 0, count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
        }
        result[offsetResult] = count > 0 ? accum / count : 0;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
    }
    return result;
}

function encodeWAV(samples, sampleRate = 16000) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    /* RIFF identifier */
    writeString(view, 0, 'RIFF');
    /* file length */
    view.setUint32(4, 36 + samples.length * 2, true);
    /* RIFF type */
    writeString(view, 8, 'WAVE');
    /* format chunk identifier */
    writeString(view, 12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw PCM) */
    view.setUint16(20, 1, true);
    /* channel count (1 mono) */
    view.setUint16(22, 1, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * 2, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, 2, true);
    /* bits per sample */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    writeString(view, 36, 'data');
    /* data chunk length */
    view.setUint32(40, samples.length * 2, true);

    // Write PCM 16-bit samples
    floatTo16BitPCM(view, 44, samples);

    return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}
