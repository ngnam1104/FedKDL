const DEFAULT_API_BASE = `${window.location.origin}/api`;
let API_BASE = getConfiguredApiBase();

let auvs = [];
let currentAUVId = null;
let uploadedFile = null;
let summary = null;

const fmt = (value, digits = 3) => {
    const n = Number(value);
    if (!Number.isFinite(n)) return "--";
    return n.toFixed(digits);
};

const fmtKb = (kb) => {
    const n = Number(kb);
    if (!Number.isFinite(n)) return "--";
    if (n >= 1024) return `${(n / 1024).toFixed(2)} MB`;
    return `${n.toFixed(1)} KB`;
};

function normalizeApiBase(value) {
    const text = String(value || "").trim().replace(/\/+$/, "");
    if (!text) return DEFAULT_API_BASE;
    return text.endsWith("/api") ? text : `${text}/api`;
}

function getConfiguredApiBase() {
    const params = new URLSearchParams(window.location.search);
    return normalizeApiBase(params.get("api") || localStorage.getItem("fedkdl_demo_api") || DEFAULT_API_BASE);
}

async function getJson(path) {
    const res = await fetch(`${API_BASE}${path}`);
    if (!res.ok) throw new Error(`API ${path} failed`);
    return res.json();
}

function setupApiConfig() {
    const input = document.getElementById("api-base-input");
    const button = document.getElementById("api-connect-btn");
    input.value = API_BASE;
    button.addEventListener("click", async () => {
        API_BASE = normalizeApiBase(input.value);
        input.value = API_BASE;
        localStorage.setItem("fedkdl_demo_api", API_BASE);
        document.getElementById("model-path").textContent = "Connecting...";
        await Promise.all([loadAUVs(), loadSummary()]);
    });
}

function setupTabs() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const tab = btn.dataset.tab;
            document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
            document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(`tab-${tab}`).classList.add("active");
        });
    });
}

async function loadSummary() {
    summary = await getJson("/demo/summary");
    document.getElementById("model-path").textContent = `Model: ${summary.model_path}`;
    renderRoundButtons("fedavg");
    renderRoundButtons("fedkdl");
    await renderReplay("fedavg", 1);
    await renderReplay("fedkdl", 1);
    await renderCentralized();
}

async function loadAUVs() {
    try {
        const data = await getJson("/auvs");
        auvs = data.auvs;
    } catch {
        auvs = [
            { id: 1, name: "AUV 1", battery: 85, status: "Active" },
            { id: 2, name: "AUV 2", battery: 73, status: "Active" },
        ];
    }
    currentAUVId = auvs[0]?.id ?? null;
    renderAUVs();
    updateSelectedAUV();
}

function renderAUVs() {
    const list = document.getElementById("auv-list");
    list.innerHTML = auvs.map((auv) => `
        <li class="auv-item ${auv.id === currentAUVId ? "active" : ""}" data-id="${auv.id}">
            <div class="auv-name">${auv.name}</div>
            <div class="auv-meta">
                <span><i class="fa-solid fa-battery-three-quarters"></i> ${auv.battery}%</span>
                <span>${auv.status}</span>
            </div>
        </li>
    `).join("");
    list.querySelectorAll(".auv-item").forEach((item) => {
        item.addEventListener("click", () => {
            currentAUVId = Number(item.dataset.id);
            renderAUVs();
            updateSelectedAUV();
            checkDetectReady();
        });
    });
}

function updateSelectedAUV() {
    const auv = auvs.find((item) => item.id === currentAUVId);
    document.getElementById("selected-auv").textContent = auv ? auv.name : "None";
}

function checkDetectReady() {
    document.getElementById("btn-detect").disabled = !(currentAUVId && uploadedFile);
}

function setupDetection() {
    const input = document.getElementById("image-upload");
    const preview = document.getElementById("preview-img");
    const empty = document.querySelector(".empty-state");
    const results = document.getElementById("results-box");
    const button = document.getElementById("btn-detect");
    const camera = document.getElementById("camera-feed");

    input.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (!file) return;
        uploadedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.style.display = "block";
            empty.style.display = "none";
            results.innerHTML = `<p class="muted">Ready for detection.</p>`;
            document.getElementById("tel-latency").textContent = "-- ms";
        };
        reader.readAsDataURL(file);
        checkDetectReady();
    });

    button.addEventListener("click", async () => {
        if (!uploadedFile || !currentAUVId) return;
        button.disabled = true;
        button.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing`;
        camera.classList.add("scanning");
        const start = performance.now();
        const form = new FormData();
        form.append("file", uploadedFile);
        try {
            const res = await fetch(`${API_BASE}/detect/${currentAUVId}`, { method: "POST", body: form });
            if (!res.ok) throw new Error("Detection API failed");
            const data = await res.json();
            preview.src = `data:image/jpeg;base64,${data.image_b64}`;
            results.innerHTML = renderDetections(data.detections, data.model_path);
            document.getElementById("tel-latency").textContent = `${(performance.now() - start).toFixed(0)} ms`;
        } catch (error) {
            results.innerHTML = `<p class="error">Detection failed. Check the API URL or SSH tunnel to the GPU server.</p>`;
        } finally {
            button.disabled = false;
            button.innerHTML = `<i class="fa-solid fa-radar"></i> Run detection`;
            camera.classList.remove("scanning");
        }
    });
}

function renderDetections(detections, modelPath) {
    const rows = detections.length
        ? detections.map((d) => `
            <div class="detection-item">
                <span>${d.label}</span>
                <strong>${(d.confidence * 100).toFixed(1)}%</strong>
            </div>
        `).join("")
        : `<p class="muted">No object detected at confidence 0.25.</p>`;
    return `
        ${rows}
        <div class="model-note">Detector: ${modelPath || "loaded model"}</div>
    `;
}

async function renderCentralized() {
    const data = await getJson("/demo/centralized");
    document.getElementById("centralized-metrics").innerHTML = `
        ${metricCard("Raw images", data.raw_images.toLocaleString())}
        ${metricCard("Avg image", `${data.avg_image_kb.toFixed(0)} KB`)}
        ${metricCard("Raw payload", `${data.payload_mb.toFixed(1)} MB`)}
        ${metricCard("Training site", "Gateway")}
    `;
    document.getElementById("centralized-note").textContent = data.note;
}

function renderRoundButtons(caseName) {
    const container = document.getElementById(`${caseName}-rounds`);
    const rounds = summary?.cases?.[caseName]?.rounds ?? [1, 2, 3];
    container.innerHTML = rounds.map((round, index) => `
        <button class="round-btn ${index === 0 ? "active" : ""}" data-round="${round}" data-case="${caseName}">
            Round ${round}
        </button>
    `).join("");
    container.querySelectorAll(".round-btn").forEach((btn) => {
        btn.addEventListener("click", async () => {
            container.querySelectorAll(".round-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            await renderReplay(caseName, Number(btn.dataset.round));
        });
    });
}

async function renderReplay(caseName, round) {
    const data = await getJson(`/demo/round/${caseName}/${round}`);
    const target = document.getElementById(`${caseName}-replay`);
    target.innerHTML = `
        <div class="flow-card">
            ${renderPipeline(data)}
        </div>
        <div class="metric-grid">
            ${metricCard("mAP50", fmt(data.metrics.mAP50, 4))}
            ${metricCard("mAP50-95", fmt(data.metrics.mAP50_95, 4))}
            ${metricCard(data.payload_label, fmtKb(data.metrics.payload_kb))}
            ${metricCard("Latency", `${fmt(data.metrics.latency_s, 1)} s`)}
            ${metricCard("Energy", `${fmt(data.metrics.energy_j, 1)} J`)}
            ${metricCard("Alive AUVs", fmt(data.metrics.alive, 0))}
        </div>
        <div class="loss-card">
            <h3>AUV local loss matrix</h3>
            <div class="loss-grid">
                ${data.losses.slice(0, 30).map(renderLossCell).join("")}
            </div>
        </div>
    `;
}

function renderPipeline(data) {
    if (data.case === "fedkdl") {
        const relayHtml = data.flow.relays.map((relay) => `
            <div class="relay-node">
                <strong>${relay.name}</strong>
                <span>${relay.auv_ids.length} AUVs</span>
            </div>
        `).join("");
        return `
            <h3>${data.title} - Round ${data.round}</h3>
            <div class="pipeline fedkdl-flow">
                <div class="node">AUV LoRA train</div>
                <div class="arrow">→</div>
                <div class="relay-column">${relayHtml}</div>
                <div class="arrow">→</div>
                <div class="node gateway">Gateway KD</div>
            </div>
            <ol class="steps">${data.flow.steps.map((step) => `<li>${step}</li>`).join("")}</ol>
        `;
    }
    return `
        <h3>${data.title} - Round ${data.round}</h3>
        <div class="pipeline fedavg-flow">
            <div class="node">AUV full train</div>
            <div class="arrow">→</div>
            <div class="node gateway">Gateway FedAvg</div>
            <div class="arrow">→</div>
            <div class="node">Broadcast</div>
        </div>
        <ol class="steps">${data.flow.steps.map((step) => `<li>${step}</li>`).join("")}</ol>
    `;
}

function renderLossCell(item) {
    const loss = Number(item.loss);
    const level = loss > 5 ? "high" : loss > 4.4 ? "mid" : "low";
    return `
        <div class="loss-cell ${level}">
            <span>AUV ${item.id}</span>
            <strong>${loss.toFixed(3)}</strong>
        </div>
    `;
}

function metricCard(label, value) {
    return `
        <div class="metric-card">
            <span>${label}</span>
            <strong>${value}</strong>
        </div>
    `;
}

async function init() {
    setupApiConfig();
    setupTabs();
    setupDetection();
    await Promise.all([loadAUVs(), loadSummary()]);
}

init().catch((error) => {
    console.error(error);
    document.getElementById("model-path").textContent = "Backend not available";
});
