const DEFAULT_API_BASE = `${window.location.origin}/api`;
const API_BASE = normalizeApiBase(
    new URLSearchParams(window.location.search).get("api")
    || localStorage.getItem("fedkdl_demo_api")
    || DEFAULT_API_BASE
);

const SCENARIO_COLORS = {
    centralized: "#ff7a72",
    fedavg: "#f6c85f",
    fedkdl: "#32d3d8",
};

let uploadedFile = null;
let activeScenario = "detect";
let currentRound = 1;
let scenarioData = null;
let simulationRunning = false;

function normalizeApiBase(value) {
    const text = String(value || "").trim().replace(/\/+$/, "");
    if (!text) return DEFAULT_API_BASE;
    return text.endsWith("/api") ? text : `${text}/api`;
}

async function getJson(path) {
    const response = await fetch(`${API_BASE}${path}`);
    if (!response.ok) throw new Error(`API request failed: ${path}`);
    return response.json();
}

const sleep = (durationMs) => new Promise((resolve) => setTimeout(resolve, durationMs));

function formatPayload(kb) {
    const value = Number(kb);
    if (!Number.isFinite(value)) return "--";
    if (value >= 1024 * 1024) return `${(value / 1024 / 1024).toFixed(2)} GB`;
    if (value >= 1024) return `${(value / 1024).toFixed(2)} MB`;
    return `${value.toFixed(1)} KB`;
}

function formatTime(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value)) return "--";
    if (value >= 86400) return `${(value / 86400).toFixed(1)} days`;
    if (value >= 3600) return `${(value / 3600).toFixed(1)} h`;
    if (value >= 60) return `${(value / 60).toFixed(1)} min`;
    return `${value.toFixed(1)} s`;
}

function metricCard(label, value) {
    return `
        <div class="metric-card">
            <span>${label}</span>
            <strong>${value}</strong>
        </div>
    `;
}

function setBackendState(online, modelPath = "") {
    const dot = document.getElementById("backend-dot");
    dot.classList.toggle("online", online);
    dot.classList.toggle("offline", !online);
    document.getElementById("backend-label").textContent = online ? "GPU backend online" : "Backend unavailable";
    document.getElementById("model-path").textContent = modelPath || "Global model";
}

function setupScenarioTabs() {
    document.querySelectorAll(".scenario-btn").forEach((button) => {
        button.addEventListener("click", async () => {
            if (simulationRunning) return;
            document.querySelectorAll(".scenario-btn").forEach((item) => item.classList.remove("active"));
            button.classList.add("active");
            activeScenario = button.dataset.scenario;

            document.getElementById("detect-view").classList.toggle("active", activeScenario === "detect");
            document.getElementById("simulation-view").classList.toggle("active", activeScenario !== "detect");

            if (activeScenario !== "detect") {
                currentRound = 1;
                await loadScenario();
            }
        });
    });
}

function setupDetection() {
    const input = document.getElementById("image-upload");
    const preview = document.getElementById("preview-img");
    const emptyState = document.querySelector(".empty-state");
    const resultsBox = document.getElementById("results-box");
    const runButton = document.getElementById("btn-detect");
    const camera = document.getElementById("camera-feed");

    input.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (!file) return;
        uploadedFile = file;
        const reader = new FileReader();
        reader.onload = (readerEvent) => {
            preview.src = readerEvent.target.result;
            preview.style.display = "block";
            emptyState.style.display = "none";
            resultsBox.innerHTML = `<p class="muted">Image ready for global-model inference.</p>`;
            document.getElementById("tel-latency").textContent = "-- ms";
        };
        reader.readAsDataURL(file);
        runButton.disabled = false;
    });

    runButton.addEventListener("click", async () => {
        if (!uploadedFile) return;
        runButton.disabled = true;
        runButton.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i><span>Processing</span>`;
        camera.classList.add("scanning");
        const startedAt = performance.now();
        const form = new FormData();
        form.append("file", uploadedFile);

        try {
            const response = await fetch(`${API_BASE}/detect`, { method: "POST", body: form });
            if (!response.ok) throw new Error("Detection request failed");
            const data = await response.json();
            preview.src = `data:image/jpeg;base64,${data.image_b64}`;
            resultsBox.innerHTML = renderDetections(data.detections, data.model_path);
            document.getElementById("tel-latency").textContent = `${(performance.now() - startedAt).toFixed(0)} ms`;
        } catch (error) {
            resultsBox.innerHTML = `<p class="error">Inference failed. Check the GPU backend and SSH tunnel.</p>`;
        } finally {
            runButton.disabled = false;
            runButton.innerHTML = `<i class="fa-solid fa-crosshairs"></i><span>Run inference</span>`;
            camera.classList.remove("scanning");
        }
    });
}

function renderDetections(detections, modelPath) {
    const rows = detections.length
        ? detections.map((detection) => `
            <div class="detection-item">
                <span>${detection.label}</span>
                <strong>${(detection.confidence * 100).toFixed(1)}%</strong>
            </div>
        `).join("")
        : `<p class="muted">No object detected at confidence 0.25.</p>`;

    return `${rows}<p class="model-note">Detector: ${modelPath || "global student model"}</p>`;
}

async function loadScenario() {
    const round = activeScenario === "centralized" ? 1 : currentRound;
    scenarioData = await getJson(`/demo/scenario/${activeScenario}/${round}`);
    renderScenario();
}

function renderScenario() {
    document.getElementById("scenario-title").textContent = scenarioData.title;
    document.getElementById("round-label").textContent = activeScenario === "centralized"
        ? "Upload cycle"
        : `Round ${scenarioData.round} / 3`;
    document.getElementById("scenario-eyebrow").textContent = activeScenario === "centralized"
        ? "Raw-data communication"
        : "One federated learning round";

    renderTopology();
    renderTimeline();
    renderScenarioMetrics();
    setPhaseStatus("Ready", "Select Run simulation");

    const runButton = document.getElementById("run-simulation");
    runButton.onclick = runSimulation;
    runButton.disabled = false;
    runButton.innerHTML = `<i class="fa-solid fa-play"></i><span>Run simulation</span>`;
}

function renderTopology() {
    const relayLayer = document.getElementById("relay-layer");
    const auvLayer = document.getElementById("auv-layer");
    const lossById = new Map((scenarioData.losses || []).map((item) => [Number(item.id), Number(item.loss)]));
    const topology = scenarioData.topology;

    relayLayer.innerHTML = topology.relays.map((relay) => `
        <div
            id="relay-${relay.id}"
            class="device relay-device ${topology.use_relays ? "" : "bypassed"}"
            style="left:${relay.x_pct}%;top:${relay.y_pct}%"
        >
            <i class="fa-solid fa-tower-broadcast"></i>
            <span>R${relay.id}</span>
        </div>
    `).join("");

    auvLayer.innerHTML = topology.auvs.map((auv) => {
        const loss = lossById.get(Number(auv.id));
        const lossBadge = auv.connected && Number.isFinite(loss)
            ? `<span class="loss-badge">L ${loss.toFixed(2)}</span>`
            : "";
        return `
            <div
                id="auv-${auv.id}"
                class="device auv-device ${auv.connected ? "" : "disconnected"}"
                style="left:${auv.x_pct}%;top:${auv.y_pct}%"
                title="${auv.connected ? `AUV ${auv.id} to Relay ${auv.relay_id}` : `AUV ${auv.id}: out of range`}"
            >
                ${lossBadge}
                <i class="fa-solid fa-robot"></i>
                <span>A${auv.id}</span>
            </div>
        `;
    }).join("");
}

function renderTimeline() {
    document.getElementById("phase-timeline").innerHTML = scenarioData.phases.map((phase, index) => `
        <div class="phase-step" data-phase-index="${index}">
            ${index + 1}. ${phase.label}
        </div>
    `).join("");
}

function renderScenarioMetrics() {
    const metrics = scenarioData.metrics;
    const demoDuration = scenarioData.phases.reduce((total, phase) => total + phase.duration_ms, 0) / 1000;
    const accuracy = activeScenario === "centralized"
        ? `${(metrics.mAP50 * 100).toFixed(2)}% upper bound`
        : `${(metrics.mAP50 * 100).toFixed(2)}%`;

    document.getElementById("scenario-metrics").innerHTML = [
        metricCard("Uplink payload", formatPayload(metrics.uplink_payload_kb)),
        metricCard("Downlink payload", metrics.downlink_payload_kb > 0 ? formatPayload(metrics.downlink_payload_kb) : "None"),
        metricCard("Training latency", formatTime(metrics.train_latency_s)),
        metricCard("Communication latency", formatTime(metrics.communication_latency_s)),
        metricCard("Physical total", formatTime(metrics.round_latency_s)),
        metricCard("Compressed demo", `${demoDuration.toFixed(1)} s`),
        metricCard("mAP50", accuracy),
        metricCard("Energy", metrics.energy_j > 0 ? `${metrics.energy_j.toFixed(0)} J` : "Not modeled"),
    ].join("");
}

function setPhaseStatus(index, label) {
    document.getElementById("phase-index").textContent = index;
    document.getElementById("phase-label").textContent = label;
}

function setTimelineState(activeIndex) {
    document.querySelectorAll(".phase-step").forEach((step, index) => {
        step.classList.toggle("active", index === activeIndex);
        step.classList.toggle("done", index < activeIndex);
    });
}

function clearNodeStates() {
    document.querySelectorAll(".device").forEach((node) => {
        node.classList.remove("training", "transmitting", "receiving", "processing");
    });
}

function clearLinks() {
    document.getElementById("communication-layer").innerHTML = "";
}

function nodeCenter(node, sceneRect) {
    const rect = node.getBoundingClientRect();
    return {
        x: rect.left - sceneRect.left + rect.width / 2,
        y: rect.top - sceneRect.top + rect.height / 2,
    };
}

function createCommunicationLink(fromNode, toNode, durationMs, color) {
    const scene = document.getElementById("ocean-scene");
    const layer = document.getElementById("communication-layer");
    const sceneRect = scene.getBoundingClientRect();
    const start = nodeCenter(fromNode, sceneRect);
    const end = nodeCenter(toNode, sceneRect);
    const deltaX = end.x - start.x;
    const deltaY = end.y - start.y;
    const length = Math.hypot(deltaX, deltaY);
    const angle = Math.atan2(deltaY, deltaX) * 180 / Math.PI;

    const link = document.createElement("div");
    link.className = "comm-link";
    link.style.left = `${start.x}px`;
    link.style.top = `${start.y}px`;
    link.style.width = `${length}px`;
    link.style.transform = `rotate(${angle}deg)`;
    link.style.setProperty("--link-color", color);

    const packet = document.createElement("div");
    packet.className = "signal-packet";
    packet.style.animationDuration = `${durationMs}ms`;
    link.appendChild(packet);
    layer.appendChild(link);

    fromNode.classList.add("transmitting");
    toNode.classList.add("receiving");
}

function relayForAuv(auvId) {
    const auv = scenarioData.topology.auvs.find((item) => Number(item.id) === Number(auvId));
    if (!auv || !auv.connected || auv.relay_id === null || auv.relay_id === undefined) return null;
    return document.getElementById(`relay-${auv.relay_id}`);
}

function communicationPairs(phaseId) {
    const gateway = document.getElementById("gateway");
    const topology = scenarioData.topology;
    const auvs = topology.auvs.map((auv) => document.getElementById(`auv-${auv.id}`));
    const connectedAuvs = topology.auvs.filter((auv) => auv.connected);
    const activeRelayIds = new Set(connectedAuvs.map((auv) => Number(auv.relay_id)));
    const activeRelays = topology.relays
        .filter((relay) => activeRelayIds.has(Number(relay.id)))
        .map((relay) => document.getElementById(`relay-${relay.id}`));

    if (phaseId === "uplink_direct") return auvs.map((auv) => [auv, gateway]);
    if (phaseId === "uplink_a2r") {
        return connectedAuvs.map((auv) => [
            document.getElementById(`auv-${auv.id}`),
            relayForAuv(auv.id),
        ]);
    }
    if (phaseId === "uplink_r2g") return activeRelays.map((relay) => [relay, gateway]);
    if (phaseId === "downlink_g2r") return activeRelays.map((relay) => [gateway, relay]);
    if (phaseId === "downlink_r2a") {
        return connectedAuvs.map((auv) => [
            relayForAuv(auv.id),
            document.getElementById(`auv-${auv.id}`),
        ]);
    }
    return [];
}

async function animatePhase(phase) {
    clearNodeStates();
    clearLinks();
    const auvs = scenarioData.topology.auvs.map((auv) => document.getElementById(`auv-${auv.id}`));
    const relays = scenarioData.topology.relays.map((relay) => document.getElementById(`relay-${relay.id}`));
    const gateway = document.getElementById("gateway");

    if (phase.id === "capture") {
        auvs.forEach((auv) => auv.classList.add("training"));
    } else if (phase.id === "train") {
        scenarioData.topology.auvs
            .filter((auv) => auv.connected)
            .forEach((auv) => document.getElementById(`auv-${auv.id}`).classList.add("training"));
    } else if (phase.id === "relay_aggregate") {
        const activeRelayIds = new Set(
            scenarioData.topology.auvs
                .filter((auv) => auv.connected)
                .map((auv) => Number(auv.relay_id))
        );
        relays.forEach((relay, relayId) => {
            if (activeRelayIds.has(relayId)) relay.classList.add("processing");
        });
    } else if (phase.id.startsWith("gateway_")) {
        gateway.classList.add("processing");
    } else {
        const pairs = communicationPairs(phase.id);
        pairs.forEach(([fromNode, toNode]) => {
            createCommunicationLink(fromNode, toNode, phase.duration_ms, SCENARIO_COLORS[activeScenario]);
        });
    }

    await sleep(phase.duration_ms);
    clearLinks();
}

async function runSimulation() {
    if (!scenarioData || simulationRunning) return;
    simulationRunning = true;
    const runButton = document.getElementById("run-simulation");
    const resetButton = document.getElementById("reset-simulation");
    runButton.disabled = true;
    resetButton.disabled = true;
    runButton.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i><span>Running</span>`;

    for (let index = 0; index < scenarioData.phases.length; index += 1) {
        const phase = scenarioData.phases[index];
        setTimelineState(index);
        setPhaseStatus(`Phase ${index + 1}/${scenarioData.phases.length}`, phase.label);
        await animatePhase(phase);
    }

    clearNodeStates();
    setTimelineState(scenarioData.phases.length);
    setPhaseStatus("Complete", activeScenario === "centralized" ? "Raw dataset received" : `Round ${currentRound} completed`);
    simulationRunning = false;
    resetButton.disabled = false;

    if (activeScenario !== "centralized" && currentRound < 3) {
        currentRound += 1;
        runButton.disabled = false;
        runButton.innerHTML = `<i class="fa-solid fa-forward-step"></i><span>Load round ${currentRound}</span>`;
        runButton.onclick = async () => {
            runButton.onclick = runSimulation;
            await loadScenario();
        };
    } else {
        runButton.disabled = false;
        runButton.innerHTML = `<i class="fa-solid fa-rotate-right"></i><span>Run again</span>`;
    }
}

async function resetSimulation() {
    if (simulationRunning) return;
    currentRound = 1;
    clearLinks();
    clearNodeStates();
    await loadScenario();
}

async function init() {
    setupScenarioTabs();
    setupDetection();
    document.getElementById("run-simulation").onclick = runSimulation;
    document.getElementById("reset-simulation").addEventListener("click", resetSimulation);

    try {
        const summary = await getJson("/demo/summary");
        setBackendState(true, summary.model_path);
    } catch (error) {
        setBackendState(false);
    }
}

init();
