// Import Transformers.js from a global CDN
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// ==========================================================================
// Application State
// ==========================================================================
let speechRecognizer = null;
let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;

// Hardcoded array of script files (since static GitHub Pages can't read folder contents natively)
const scriptFiles = ['script1.txt', 'script2.txt', 'script3.txt'];
let currentScriptText = "";
let originalWordsArray = [];

// ==========================================================================
// DOM Elements
// ==========================================================================
const statusBadge = document.getElementById('status-badge');
const instructionText = document.getElementById('instruction-text');
const wordsContainer = document.getElementById('words-container');
const recordBtn = document.getElementById('record-btn');
const nextBtn = document.getElementById('next-btn');
const statsContainer = document.getElementById('stats');
const accuracyVal = document.getElementById('accuracy-val');
const wordsVal = document.getElementById('words-val');

// ==========================================================================
// 1. Initialization & Model Loading
// ==========================================================================
async function init() {
    try {
        // Initialize the Whisper pipeline using the ultra-lightweight English model
        speechRecognizer = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
        
        // Update UI states once loaded
        statusBadge.textContent = "ready";
        statusBadge.className = "status-badge ready";
        instructionText.textContent = "Press the microphone button and read the script aloud.";
        recordBtn.disabled = false;

        // Load the first script
        loadRandomScript();
    } catch (error) {
        console.error("Failed to initialize Whisper model:", error);
        statusBadge.textContent = "Initialization Failed";
        instructionText.textContent = "Could not load the AI model. Check your internet connection.";
    }
}

// ==========================================================================
// 2. Script Management
// ==========================================================================
async function loadRandomScript() {
    // Hide stats from previous run
    statsContainer.classList.add('hidden');
    
    const randomIndex = Math.floor(Math.random() * scriptFiles.length);
    const targetScript = scriptFiles[randomIndex];

    try {
        const response = await fetch(`./scripts/${targetScript}`);
        if (!response.ok) throw new Error("Script file not found");
        
        currentScriptText = await response.text();
        displayScript(currentScriptText);
    } catch (error) {
        console.error(error);
        // Fallback script if text files aren't created yet
        currentScriptText = "Hello team, I wanted to provide a quick update on our software implementation architecture. Let me know if you have any questions.";
        displayScript(currentScriptText);
    }
}

function displayScript(text) {
    wordsContainer.innerHTML = "";
    // Split by whitespace to keep words intact with their original punctuation for visual display
    originalWordsArray = text.trim().split(/\s+/);

    originalWordsArray.forEach((word, index) => {
        const span = document.createElement('span');
        span.className = 'word';
        span.id = `word-${index}`;
        span.textContent = word;
        wordsContainer.appendChild(span);
    });
}

// ==========================================================================
// 3. Audio Recording Logic
// ==========================================================================
async function toggleRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        // Stop Recording
        mediaRecorder.stop();
        recordBtn.classList.remove('recording-active');
        recordBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
        statusBadge.textContent = "processing audio...";
        statusBadge.className = "status-badge loading";
        instructionText.textContent = "Analyzing your pronunciation, please wait...";
        recordBtn.disabled = true;
    } else {
        // Start Recording
        audioChunks = [];
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                // Stop all tracks to release microphone hardware immediately
                stream.getTracks().forEach(track => track.stop());
                await processAudio();
            };

            mediaRecorder.start();
            recordBtn.classList.add('recording-active');
            recordBtn.innerHTML = '<i class="fa-solid fa-stop"></i>';
            statusBadge.textContent = "recording";
            statusBadge.className = "status-badge recording";
            instructionText.textContent = "Reading now... press stop when finished.";
        } catch (err) {
            console.error("Microphone access denied:", err);
            alert("Please allow microphone access to use MonkeyRead.");
        }
    }
}

// ==========================================================================
// 4. Client-Side Machine Learning Processing
// ==========================================================================
async function processAudio() {
    try {
        const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
        
        // Force resampling to 16kHz Mono as strictly required by Whisper AI
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        }
        
        const arrayBuffer = await audioBlob.arrayBuffer();
        const decodedAudio = await audioContext.decodeAudioData(arrayBuffer);
        
        // Extract raw Float32 audio channel data
        const rawAudioData = decodedAudio.getChannelData(0);

        // Feed the local audio directly into Transformers.js Whisper pipeline
        const result = await speechRecognizer(rawAudioData, {
            chunk_length_s: 30,
            stride_length_s: 5,
            return_timestamps: false
        });

        evaluatePronunciation(result.text);

    } catch (error) {
        console.error("Error processing audio:", error);
        instructionText.textContent = "Error parsing your audio. Please try reading again.";
    } finally {
        // Reset buttons back to operational state
        recordBtn.disabled = false;
        statusBadge.textContent = "ready";
        statusBadge.className = "status-badge ready";
    }
}

// ==========================================================================
// 5. Clean Text & Alignment Evaluation Logic
// ==========================================================================
function cleanText(text) {
    return text
        .toLowerCase()
        .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?"']/g, "") // Removes all punctuation completely
        .replace(/\s+/g, " ")                           // Merges duplicate spacing
        .trim();
}

function evaluatePronunciation(transcribedText) {
    const cleanTranscript = cleanText(transcribedText);
    const spokenWordsArray = cleanTranscript.split(" ").filter(w => w.length > 0);

    let correctCount = 0;
    let transcriptPointer = 0;

    // Moving window alignment algorithm to handle stuttering or skipped words without collapsing the layout
    originalWordsArray.forEach((originalWord, index) => {
        const wordDom = document.getElementById(`word-${index}`);
        const cleanOriginalWord = cleanText(originalWord);

        let wordMatched = false;

        // Look ahead up to 4 words in the transcribed array to see if the word exists
        for (let lookAhead = 0; lookAhead < 5; lookAhead++) {
            const currentCheckIndex = transcriptPointer + lookAhead;
            if (currentCheckIndex < spokenWordsArray.length && spokenWordsArray[currentCheckIndex] === cleanOriginalWord) {
                wordMatched = true;
                transcriptPointer = currentCheckIndex + 1; // Advance pointer past this matched token
                break;
            }
        }

        if (wordMatched) {
            wordDom.className = "word correct";
            correctCount++;
        } else {
            wordDom.className = "word incorrect";
            // Attach individual Text-to-Speech playback capability natively to red words
            wordDom.onclick = () => speakWord(cleanOriginalWord);
        }
    });

    // Display Results Summary
    const finalAccuracy = Math.round((correctCount / originalWordsArray.length) * 100);
    accuracyVal.textContent = `${finalAccuracy}%`;
    wordsVal.textContent = `${correctCount}/${originalWordsArray.length}`;
    
    statsContainer.classList.remove('hidden');
    instructionText.textContent = "Review your results below. Click on any red word to hear how it sounds.";
}

// ==========================================================================
// 6. Text-to-Speech (Built-in Free Engine)
// ==========================================================================
function speakWord(word) {
    if ('speechSynthesis' in window) {
        // Cancel any lingering spoken utterances immediately
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(word);
        
        // Select a standard English voice profile
        const voices = window.speechSynthesis.getVoices();
        const naturalVoice = voices.find(voice => voice.lang.startsWith('en-US') || voice.lang.startsWith('en-GB'));
        if (naturalVoice) utterance.voice = naturalVoice;
        
        utterance.rate = 0.85; // Slightly slower pacing to ensure clean structural separation
        window.speechSynthesis.speak(utterance);
    }
}

// ==========================================================================
// Event Listeners
// ==========================================================================
recordBtn.addEventListener('click', toggleRecording);
nextBtn.addEventListener('click', loadRandomScript);

// Crucial workaround: Speech Synthesis engines load asynchronously on some web engines (Chrome/Safari)
if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = () => {};
}

// Boot up the application
init();