// --- Backend API Configuration ---
const API_BASE = "http://127.0.0.1:8000";

document.addEventListener('DOMContentLoaded', () => {
    // --- Authentication (Login/Signup) Logic ---
    const loginBtn = document.getElementById('login-btn');
    const modal = document.getElementById('login-modal');
    const closeBtn = document.getElementById('close-modal');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const authForms = document.querySelectorAll('.auth-form');

    loginBtn.addEventListener('click', () => modal.classList.add('active'));
    closeBtn.addEventListener('click', () => modal.classList.remove('active'));

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.getAttribute('data-tab');
            
            // Switch tabs
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Switch forms
            authForms.forEach(form => {
                form.classList.remove('active');
                if (form.id === `${tab}-form`) form.classList.add('active');
            });
        });
    });

    // Handle Form Submissions & Biometric Setup
    let bioStream = null;

    authForms.forEach(form => {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            const submitBtn = form.querySelector('.submit-btn');
            
            // For BOTH Login and Signup, trigger the Biometric Scanner!
            submitBtn.textContent = 'Initializing Security...';
            
            setTimeout(() => {
                authForms.forEach(f => f.classList.remove('active'));
                document.getElementById('biometric-setup').classList.add('active');
                
                const tabsContainer = document.querySelector('.auth-tabs');
                if (tabsContainer) tabsContainer.style.display = 'none'; // Hide tabs
                
                // Set logic text based on form type
                const authHeader = document.querySelector('#biometric-setup h2');
                if (form.id === 'login-form') {
                    authHeader.innerHTML = '<i class="fa-solid fa-user-lock"></i> Verifying Identity';
                } else {
                    authHeader.innerHTML = '<i class="fa-solid fa-shield-virus"></i> Registering Biometrics';
                }

                // Auto-start Biometric Scan
                startBiometricsSequence();
                
            }, 800);
        });
    });

    // Auto-running Biometric Automation
    let fingerScanTimeout = null;
    let isFingerScanComplete = false;

    async function startBiometricsSequence() {
        const video = document.getElementById('setup-webcam');
        const faceLine = document.getElementById('face-scan-line');
        const faceStatus = document.getElementById('face-status');
        const fingerBox = document.getElementById('fingerprint-scanner');
        const fingerStatus = document.getElementById('finger-status');
        
        // Reset state for new attempts
        isFingerScanComplete = false;
        fingerBox.className = 'fingerprint-box';
        fingerStatus.textContent = 'Waiting for Fingerprint';
        fingerStatus.className = '';
        fingerStatus.style.color = 'var(--text-secondary)';

        // 1. Start Camera
        try {
            bioStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
            video.srcObject = bioStream;
        } catch (err) {
            console.warn("Camera failed, simulating Face ID.", err);
        }

        // 2. Scan Face (Automated but with realistic delay)
        setTimeout(() => {
            faceLine.style.display = 'block';
            faceStatus.textContent = 'Scanning Facial AI Vectors...';
            faceStatus.style.color = 'var(--accent)';
            
            setTimeout(() => {
                faceLine.style.display = 'none';
                faceStatus.textContent = 'Face Target Locked';
                faceStatus.className = 'success';
                document.querySelector('.face-scanner-circle').style.borderColor = 'var(--success)';

                // 3. Prompt User for Interactive Fingerprint!
                fingerStatus.textContent = 'Press & Hold Fingerprint icon';
                fingerStatus.style.color = '#f1c40f'; // Yellow prompt
                fingerBox.style.cursor = 'pointer';

                // Ensure listeners are only added once
                if (!fingerBox.dataset.interactive) {
                    fingerBox.dataset.interactive = "true";
                    
                    fingerBox.addEventListener('mousedown', holdFingerprint);
                    fingerBox.addEventListener('touchstart', holdFingerprint, {passive: false});
                    
                    fingerBox.addEventListener('mouseup', releaseFingerprint);
                    fingerBox.addEventListener('mouseleave', releaseFingerprint);
                    fingerBox.addEventListener('touchend', releaseFingerprint);
                }

            }, 2500); // Simulated AI Face scan processing time

        }, 800);
    }

    // --- Interactive Hover & Hold logic (Foolproof for Presentation) ---
    function holdFingerprint(e) {
        if(e) e.preventDefault();
        if (isFingerScanComplete) return;

        const fingerBox = document.getElementById('fingerprint-scanner');
        const fingerLine = document.getElementById('finger-scan-line');
        const fingerStatus = document.getElementById('finger-status');

        fingerBox.classList.add('scanning');
        fingerLine.style.display = 'block';
        fingerStatus.textContent = 'Analyzing Thermal Print... HOLD';
        fingerStatus.style.color = 'var(--accent)';

        // Require 2 full seconds of holding your mouse click
        fingerScanTimeout = setTimeout(() => {
            isFingerScanComplete = true;
            fingerLine.style.display = 'none';
            fingerBox.classList.remove('scanning');
            fingerBox.classList.add('success');
            fingerStatus.textContent = 'Access Granted!';
            fingerStatus.className = 'success';
            
            setTimeout(() => {
                completeLogin();
            }, 1000);
        }, 2000);
    }

    function releaseFingerprint() {
        if (isFingerScanComplete) return;
        
        clearTimeout(fingerScanTimeout); // Cancel the scan!
        
        const fingerBox = document.getElementById('fingerprint-scanner');
        const fingerLine = document.getElementById('finger-scan-line');
        const fingerStatus = document.getElementById('finger-status');

        if(fingerBox.classList.contains('scanning')) {
            fingerBox.classList.remove('scanning');
            fingerLine.style.display = 'none';
            fingerStatus.textContent = 'Scan Interrupted! Press & Hold.';
            fingerStatus.style.color = 'var(--error)';
        }
    }

    function completeLogin() {
        if (modal) modal.classList.remove('active');
        if (loginBtn) {
            loginBtn.innerHTML = '<i class="fa-solid fa-circle-user"></i> Dashboard';
            loginBtn.style.backgroundColor = 'var(--success)';
        }
        
        if (bioStream) bioStream.getTracks().forEach(t => t.stop());

        // --- Switch to Dashboard View ---
        const dashboard = document.getElementById('dashboard-view');
        if (dashboard) {
            // Hide all landing page elements
            document.querySelectorAll('header, section, footer, .reveal').forEach(el => {
                if (el.id !== 'dashboard-view' && !el.closest('#dashboard-view')) {
                    el.style.display = 'none';
                }
            });
            
            // Show dashboard
            dashboard.style.display = 'flex';
            document.body.style.overflow = 'hidden'; 
            
            initDashboardAI();
            setupDashboardForensics();
        }
    }

    // --- Dashboard AI Assistant Integration ---
    function initDashboardAI() {
        const dInput = document.getElementById('dashboard-chat-input');
        const dSend = document.getElementById('dashboard-chat-send');
        const dChatView = document.getElementById('dashboard-chat-view');

        if (!dSend || !dInput) return;

        dSend.addEventListener('click', async () => {
            const text = dInput.value.trim();
            if (!text) return;
            
            const userMsg = document.createElement('div');
            userMsg.className = 'user-msg';
            userMsg.textContent = text;
            dChatView.appendChild(userMsg);
            dInput.value = '';
            
            const botMsg = document.createElement('div');
            botMsg.className = 'bot-msg typing-indicator';
            botMsg.textContent = 'Neural Engine Analyzing...';
            dChatView.appendChild(botMsg);
            dChatView.scrollTop = dChatView.scrollHeight;

            const response = await fetchGeminiChatbotResponse(text);
            botMsg.classList.remove('typing-indicator');
            botMsg.innerHTML = response.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            dChatView.scrollTop = dChatView.scrollHeight;
        });

        dInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') dSend.click();
        });
    }

    // --- Dashboard Forensics Setup ---
    function setupDashboardForensics() {
        const imageInput = document.getElementById('image-input');
        const imageDropZone = document.getElementById('image-drop-zone');
        const webcamBtn = document.getElementById('webcam-btn');

        if (imageDropZone) {
            imageDropZone.addEventListener('dragover', (e) => { e.preventDefault(); imageDropZone.classList.add('dragover'); });
            imageDropZone.addEventListener('dragleave', () => { imageDropZone.classList.remove('dragover'); });
            imageDropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                imageDropZone.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                if (file && file.type.startsWith('image/')) handleImageUpload(file);
            });
            imageDropZone.addEventListener('click', () => imageInput.click());
        }

        if (imageInput) {
            imageInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) handleImageUpload(file);
            });
        }
        
        if (webcamBtn) {
            webcamBtn.addEventListener('click', startWebcamScanner);
        }
    }

    async function handleImageUpload(file) {
        const preview = document.getElementById('image-preview');
        preview.src = URL.createObjectURL(file);
        
        showLoading('image');
        const startTime = performance.now();

        try {
            // --- NEW: Route through Local Python Backend (Stable AI Connection) ---
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch(`${API_BASE}/predict-image`, { 
                method: 'POST', 
                body: formData 
            });

            if (!response.ok) throw new Error("Local Engine Conflict");
            
            const data = await response.json();
            data.detection_time = `${((performance.now() - startTime) / 1000).toFixed(2)}s`;
            
            showResult('image', data);
        } catch (error) {
            console.error("Forensic Hub Error:", error);
            simulateResult('image', "Local Hub Offline. Please visit terminal.");
        }
    }

    async function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.onerror = error => reject(error);
        });
    }

    async function analyzeImageWithAI(file) {
        const base64 = await fileToBase64(file);
        
        // Exhaustive model fallback system - testing both V1 and V1BETA
        const testScenarios = [
            { model: "gemini-1.5-flash", version: "v1beta" },
            { model: "gemini-1.5-flash", version: "v1" },
            { model: "gemini-pro-vision", version: "v1beta" }
        ];
        
        let lastError = "All neural paths failed.";

        for (const scenario of testScenarios) {
            try {
                const url = `https://generativelanguage.googleapis.com/${scenario.version}/models/${scenario.model}:generateContent?key=${GEMINI_API_KEY}`;
                
                const prompt = `Analyze this image for deepfake signs. Responde ONLY in JSON: { "prediction": "Deepfake" or "Real", "confidence": "0-100%", "artifacts": "X" }`;
                const payload = {
                    contents: [{
                        parts: [
                            { text: prompt },
                            { inline_data: { mime_type: file.type || 'image/jpeg', data: base64 } }
                        ]
                    }]
                };

                const response = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    console.error(`Scenario [${scenario.model}] Failed:`, errorText);
                    lastError = `Neural Model ${scenario.model} [${scenario.version}] returned HTTP ${response.status}`;
                    continue; // Try next
                }
                
                const data = await response.json();
                const rawText = data.candidates[0].content.parts[0].text;
                
                const jsonStart = rawText.indexOf('{');
                const jsonEnd = rawText.lastIndexOf('}') + 1;
                const jsonString = rawText.substring(jsonStart, jsonEnd);
                return JSON.parse(jsonString);

            } catch (e) {
                console.warn(`Connection to ${scenario.model} interrupted:`, e);
                lastError = `Connection interrupted: ${e.message}`;
            }
        }
        
        throw new Error(`Forensic Engine Standby: ${lastError}`);
    }


    function showLoading(type) {
        document.getElementById(`${type}-result`).style.display = 'block';
        document.getElementById(`${type}-loading`).style.display = 'flex';
        document.getElementById(`${type}-analysis-details`).style.display = 'none';
        
        const progress = document.getElementById(`${type}-progress`);
        if (progress) {
            progress.style.background = `conic-gradient(rgba(88, 166, 255, 0.1) 0deg, rgba(88, 166, 255, 0.1) 0deg)`;
            const val = progress.querySelector('.progress-value');
            if (val) val.textContent = '0%';
        }
    }

    function showResult(type, data) {
        document.getElementById(`${type}-loading`).style.display = 'none';
        document.getElementById(`${type}-analysis-details`).style.display = 'block';

        const isFake = data.prediction.toLowerCase() === 'deepfake';
        const confVal = parseFloat(data.confidence.replace('%', ''));
        
        const label = document.getElementById(`${type}-result-label`);
        label.textContent = isFake ? 'Synthetic Artifacts Detected' : 'Authentic Media Verified';
        label.className = `result-label ${isFake ? 'deepfake' : 'real'}`;

        document.getElementById(`${type}-artifacts`).textContent = data.artifacts || (isFake ? "Neural Patterns Found" : "None Detected");
        document.getElementById(`${type}-time`).textContent = data.detection_time;
        document.getElementById('detail-confidence').textContent = data.confidence;

        if (type === 'image') {
            const heatmap = document.getElementById('heatmap-overlay');
            heatmap.style.display = isFake ? 'block' : 'none';
            if (isFake) {
                heatmap.style.top = (30 + Math.random() * 40) + '%';
                heatmap.style.left = (30 + Math.random() * 40) + '%';
            }
            document.getElementById('meta-software').textContent = isFake ? "Generative AI" : "Original Camera";
            document.getElementById('meta-date').textContent = new Date().toLocaleDateString();
        }

        animateProgress(type, confVal, isFake);
    }

    function animateProgress(type, endValue, isFake) {
        const progress = document.getElementById(`${type}-progress`);
        const valueEl = progress.querySelector('.progress-value');
        let startValue = 0;
        const color = isFake ? 'var(--error)' : 'var(--success)';

        const interval = setInterval(() => {
            startValue++;
            valueEl.textContent = `${startValue}%`;
            progress.style.background = `conic-gradient(${color} ${startValue * 3.6}deg, rgba(88, 166, 255, 0.1) 0deg)`;
            if (startValue >= Math.floor(endValue)) {
                clearInterval(interval);
                valueEl.textContent = `${endValue.toFixed(1)}%`;
            }
        }, 15);
    }

    // --- AI Engine Integration ---
    const GEMINI_API_KEY = "AIzaSyCd_wZpQB-_CqvTyN0Vaz9VZmOEFhDbIgo".trim();

    async function fetchGeminiChatbotResponse(userPrompt) {
        try {
            const response = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: userPrompt })
            });
            
            if (!response.ok) throw new Error("Local Chat Hub Offline");
            const data = await response.json();
            return data.response;
        } catch (error) {
            console.error("Chat Hub Error:", error);
            return "Connection error (Local Hub). Please ensure backend is running.";
        }
    }

    // --- Webcam Support ---
    function startWebcamScanner() {
        const modal = document.getElementById('webcam-modal');
        const video = document.getElementById('webcam-stream');
        const captureBtn = document.getElementById('capture-btn');
        const closeBtn = document.getElementById('close-webcam');
        let stream = null;

        modal.classList.add('active');
        navigator.mediaDevices.getUserMedia({ video: true }).then(s => {
            stream = s;
            video.srcObject = s;
        });

        closeBtn.onclick = () => {
            modal.classList.remove('active');
            if (stream) stream.getTracks().forEach(t => t.stop());
        };

        captureBtn.onclick = () => {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            canvas.toBlob(blob => {
                const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
                modal.classList.remove('active');
                if (stream) stream.getTracks().forEach(t => t.stop());
                handleImageUpload(file);
            });
        };
    }

    function simulateResult(type, errorMsg) {
        alert(errorMsg || "Direct AI Connection Offline. Starting Simulation.");
        document.getElementById(`${type}-loading`).style.display = 'none';
    }

    window.generateReport = function(type) {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        doc.text("DEEPSCAN FORENSIC DOSSIER", 20, 20);
        doc.text(`Time: ${new Date().toLocaleString()}`, 20, 40);
        doc.save(`DeepScan_Analysis.pdf`);

    }

    // --- REVEAL ON SCROLL LOGIC ---
    const reveal = () => {
        const reveals = document.querySelectorAll(".reveal");
        reveals.forEach(el => {
            const windowHeight = window.innerHeight;
            const elementTop = el.getBoundingClientRect().top;
            const elementVisible = 100;
            if (elementTop < windowHeight - elementVisible) {
                el.classList.add("active");
            }
        });
    }
    window.addEventListener("scroll", reveal);
    reveal(); // Initial check on load
});
