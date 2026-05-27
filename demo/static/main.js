const API_BASE = "http://localhost:5000/api";

let auvs = [];
let currentAUVId = null;
let uploadedFile = null;

const auvListEl = document.getElementById("auv-list");
const currentAUVNameEl = document.getElementById("current-auv-name");
const imageUploadEl = document.getElementById("image-upload");
const previewImgEl = document.getElementById("preview-img");
const emptyStateEl = document.querySelector(".empty-state");
const btnDetect = document.getElementById("btn-detect");
const cameraFeedEl = document.getElementById("camera-feed");
const resultsBoxEl = document.getElementById("results-box");
const telLatencyEl = document.getElementById("tel-latency");

// Fetch auvs on load
async function fetchAUVs() {
    try {
        const res = await fetch(`${API_BASE}/auvs`);
        const data = await res.json();
        auvs = data.auvs;
        renderAUVs();
    } catch (e) {
        console.error("Failed to fetch auvs:", e);
        // Fallback for UI if backend is not running
        auvs = [
            { id: 1, name: "AUV Alpha (Mock)", battery: 85, status: "Active" },
            { id: 2, name: "AUV Beta (Mock)", battery: 62, status: "Active" }
        ];
        renderAUVs();
    }
}

function renderAUVs() {
    auvListEl.innerHTML = "";
    auvs.forEach(s => {
        const li = document.createElement("li");
        li.className = `auv-item ${currentAUVId === s.id ? "active" : ""}`;
        li.innerHTML = `
            <div class="auv-name">${s.name}</div>
            <div class="auv-meta">
                <span><i class="fa-solid fa-battery-three-quarters"></i> ${s.battery}%</span>
                <span style="color: ${s.status === 'Active' ? 'var(--success)' : 'var(--accent)'}">${s.status}</span>
            </div>
        `;
        li.onclick = () => selectAUV(s.id);
        auvListEl.appendChild(li);
    });
}

function selectAUV(id) {
    currentAUVId = id;
    const auv = auvs.find(s => s.id === id);
    currentAUVNameEl.textContent = `Camera View: ${auv.name}`;
    renderAUVs();
    checkReadyState();
}

// Handle Image Upload
imageUploadEl.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    uploadedFile = file;
    const reader = new FileReader();
    reader.onload = (event) => {
        previewImgEl.src = event.target.result;
        previewImgEl.style.display = "block";
        emptyStateEl.style.display = "none";
        
        // Reset results
        resultsBoxEl.innerHTML = `<p class="placeholder-text">Ready for detection...</p>`;
        telLatencyEl.textContent = "-- ms";
    };
    reader.readAsDataURL(file);
    checkReadyState();
});

function checkReadyState() {
    if (currentAUVId && uploadedFile) {
        btnDetect.disabled = false;
    } else {
        btnDetect.disabled = true;
    }
}

// Run Detection
btnDetect.addEventListener("click", async () => {
    if (!currentAUVId || !uploadedFile) return;
    
    // UI Loading state
    btnDetect.disabled = true;
    btnDetect.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing...`;
    cameraFeedEl.classList.add("scanning");
    
    const startTime = performance.now();
    
    const formData = new FormData();
    formData.append("file", uploadedFile);
    
    try {
        const res = await fetch(`${API_BASE}/detect/${currentAUVId}`, {
            method: "POST",
            body: formData
        });
        
        if (!res.ok) throw new Error("API Error");
        
        const data = await res.json();
        
        // Render Image
        previewImgEl.src = `data:image/jpeg;base64,${data.image_b64}`;
        
        // Render Results
        if (data.detections.length === 0) {
            resultsBoxEl.innerHTML = `<p class="placeholder-text">No objects detected.</p>`;
        } else {
            resultsBoxEl.innerHTML = data.detections.map(d => `
                <div class="detection-item">
                    <span class="det-label"><i class="fa-solid fa-fish"></i> ${d.label}</span>
                    <span class="det-conf">${(d.confidence * 100).toFixed(1)}%</span>
                </div>
            `).join("");
        }
        
        // Render Telemetry
        const endTime = performance.now();
        telLatencyEl.textContent = `${(endTime - startTime).toFixed(0)} ms`;
        
    } catch (e) {
        console.error("Detection failed:", e);
        resultsBoxEl.innerHTML = `<p class="placeholder-text" style="color: var(--accent);">Error connecting to API.</p>`;
    } finally {
        btnDetect.disabled = false;
        btnDetect.innerHTML = `<i class="fa-solid fa-radar"></i> Run Detection`;
        cameraFeedEl.classList.remove("scanning");
    }
});

// Init
fetchAUVs();
