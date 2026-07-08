const DEFAULT_API_BASE = `${window.location.origin}/api`;
const API_BASE = normalizeApiBase(
    new URLSearchParams(window.location.search).get("api")
    || localStorage.getItem("fedkdl_demo_api")
    || DEFAULT_API_BASE
);

const SCENARIO_COLORS = {
    centralized: "#ff7a72",
    fedavg_flat: "#4d8eff",
    fedavg_hfl: "#f6c85f",
    fedkdl: "#32d3d8",
};

const TRAIN_LOG_DELAY_MS = 100;
const ROUND_TRANSITION_DELAY_MS = 420;
const CENTRALIZED_TRAIN_STEPS = 18;
const STANDALONE_LOG_REPLAY_DELAY_MS = TRAIN_LOG_DELAY_MS;

let uploadedFile = null;
let uploadedFileName = "upload.jpg";
let selectedSampleId = null;
let activeScenario = "detect";
let currentRound = 1;
let scenarioData = null;
let simulationRunning = false;
let simulationPaused = false;
let simulationStopRequested = false;
let replayAllRunning = false;
let liveDemoAvailableByCase = {};
let liveTrainingAvailable = false;
let liveTrainingJobId = null;
let liveTrainingRunning = false;
let logReplayAvailableByCase = {};
let logReplayRunning = false;
let logReplayStopRequested = false;
let logReplayLatestLoss = new Map();
let logReplayPreviousLoss = new Map();
let logReplayLatestEvent = new Map();
let trainingLogCacheByCase = {};
let selectedInspectorNode = null;

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

async function postJson(path) {
    const response = await fetch(`${API_BASE}${path}`, { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `API request failed: ${path}`);
    return data;
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

function setBackendState(online, modelPath = "", replayOnline = online) {
    const dot = document.getElementById("backend-dot");
    dot.classList.toggle("online", replayOnline);
    dot.classList.toggle("offline", !replayOnline);
    document.getElementById("backend-label").textContent = online
        ? "GPU backend online"
        : replayOnline
            ? "Replay backend online · detection unavailable"
            : "Backend unavailable";
    document.getElementById("model-path").textContent = modelPath || "Global model";
}

function setupScenarioTabs() {
    document.querySelectorAll(".scenario-btn").forEach((button) => {
        button.addEventListener("click", async () => {
            if (simulationRunning || liveTrainingRunning || logReplayRunning) return;
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

    input.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (!file) return;
        uploadedFile = file;
        uploadedFileName = file.name || "upload.jpg";
        selectedSampleId = null;
        document.querySelectorAll(".sample-image-btn").forEach((button) => button.classList.remove("active"));
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

    runButton.addEventListener("click", runDetection);
    loadSampleImages();
}

async function loadSampleImages() {
    const grid = document.getElementById("sample-image-grid");
    try {
        const data = await getJson("/demo/sample-images");
        const images = data.images || [];
        if (!images.length) {
            grid.innerHTML = `<p class="muted">No dataset images found on this server.</p>`;
            return;
        }
        grid.innerHTML = images.map((image) => {
            const source = `${API_BASE}${image.url.replace(/^\/api/, "")}`;
            return `
                <button class="sample-image-btn" type="button" data-sample-id="${image.id}" data-sample-url="${source}" data-sample-name="${image.name}" title="${image.name}">
                    <img src="${source}" alt="${image.name}" loading="lazy">
                    <span>${image.name}</span>
                </button>
            `;
        }).join("");
        grid.querySelectorAll(".sample-image-btn").forEach((button) => {
            button.addEventListener("click", () => selectSampleImage(button));
        });
    } catch (error) {
        grid.innerHTML = `<p class="muted">Sample images unavailable.</p>`;
    }
}

async function selectSampleImage(button) {
    const preview = document.getElementById("preview-img");
    const emptyState = document.querySelector(".empty-state");
    const resultsBox = document.getElementById("results-box");
    const runButton = document.getElementById("btn-detect");
    const imageUrl = button.dataset.sampleUrl;
    selectedSampleId = Number(button.dataset.sampleId);
    uploadedFileName = button.dataset.sampleName || `sample-${selectedSampleId}.jpg`;
    document.querySelectorAll(".sample-image-btn").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    preview.src = imageUrl;
    preview.style.display = "block";
    emptyState.style.display = "none";
    resultsBox.innerHTML = `<p class="muted">Sample image selected. Running inference.</p>`;
    document.getElementById("tel-latency").textContent = "-- ms";

    try {
        const response = await fetch(imageUrl);
        if (!response.ok) throw new Error("Sample image request failed");
        uploadedFile = await response.blob();
        runButton.disabled = false;
        await runDetection();
    } catch (error) {
        uploadedFile = null;
        runButton.disabled = true;
        resultsBox.innerHTML = `<p class="error">Could not load the selected sample image.</p>`;
    }
}

async function runDetection() {
    if (!uploadedFile) return;
    const preview = document.getElementById("preview-img");
    const resultsBox = document.getElementById("results-box");
    const runButton = document.getElementById("btn-detect");
    const camera = document.getElementById("camera-feed");
    runButton.disabled = true;
    runButton.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i><span>Processing</span>`;
    camera.classList.add("scanning");
    const startedAt = performance.now();
    const form = new FormData();
    form.append("file", uploadedFile, uploadedFileName);

    try {
        const response = await fetch(`${API_BASE}/detect`, { method: "POST", body: form });
        if (!response.ok) throw new Error("Detection request failed");
        const data = await response.json();
        preview.src = `data:image/jpeg;base64,${data.image_b64}`;
        const endToEndMs = performance.now() - startedAt;
        resultsBox.innerHTML = renderDetections(
            data.detections,
            data.model_path,
            data.server_inference_ms,
            endToEndMs,
            data.confidence_threshold,
            data.fallback_threshold_used,
        );
        document.getElementById("tel-latency").textContent = `${endToEndMs.toFixed(0)} ms`;
    } catch (error) {
        resultsBox.innerHTML = `<p class="error">Inference failed. Check the GPU backend and SSH tunnel.</p>`;
    } finally {
        runButton.disabled = false;
        runButton.innerHTML = `<i class="fa-solid fa-crosshairs"></i><span>Run inference</span>`;
        camera.classList.remove("scanning");
    }
}

function renderDetections(detections, modelPath, serverInferenceMs, endToEndMs, confidenceThreshold = 0.25, fallbackThresholdUsed = false) {
    const rows = detections.length
        ? detections.map((detection) => `
            <div class="detection-item">
                <span>${detection.label}</span>
                <strong>${(detection.confidence * 100).toFixed(1)}%</strong>
            </div>
        `).join("")
        : `<p class="muted">No object detected at confidence ${Number(confidenceThreshold).toFixed(2)}.</p>`;
    const fallbackNote = fallbackThresholdUsed
        ? `<p class="model-note">Display threshold relaxed for demo visibility.</p>`
        : "";

    return `
        ${rows}
        ${fallbackNote}
        <p class="model-note">GPU inference: ${Number(serverInferenceMs || 0).toFixed(1)} ms</p>
        <p class="model-note">Browser round trip: ${Number(endToEndMs || 0).toFixed(1)} ms</p>
        <p class="model-note">Detector: ${modelPath || "global student model"}</p>
    `;
}

async function loadScenario() {
    const round = currentRound;
    scenarioData = await getJson(`/demo/scenario/${activeScenario}/${round}`);
    renderScenario();
}

async function getTrainingLogReplay(caseName, maxRounds = getRequestedRounds()) {
    if (!logReplayAvailableByCase[caseName]) return null;
    const cacheKey = `${caseName}:${maxRounds}`;
    if (!trainingLogCacheByCase[cacheKey]) {
        trainingLogCacheByCase[cacheKey] = getJson(`/demo/training-log/${caseName}?max_rounds=${maxRounds}`)
            .catch(() => null);
    }
    return trainingLogCacheByCase[cacheKey];
}

function getRequestedRounds() {
    const input = document.getElementById("round-count");
    const value = Number(input?.value || 40);
    return Math.min(Math.max(Math.round(value) || 1, 1), 40);
}

function compactTrainingEvents(events, targetCount) {
    if (events.length <= targetCount) return events;
    const stride = Math.max(1, Math.ceil(events.length / targetCount));
    const selected = events.filter((_, index) => index % stride === 0);
    const last = events[events.length - 1];
    if (selected[selected.length - 1] !== last) selected.push(last);
    return selected;
}

function renderScenario() {
    document.getElementById("scenario-title").textContent = scenarioData.title;
    document.getElementById("round-label").textContent = `Round ${scenarioData.round} / 40`;
    document.getElementById("scenario-eyebrow").textContent = activeScenario === "centralized"
        ? "Centralized training replay"
        : "One federated learning round";
    document.getElementById("round-control").classList.remove("hidden");
    document.getElementById("run-all-simulation").classList.add("hidden");
    document.getElementById("run-log-replay").classList.add("hidden");
    document.getElementById("run-simulation").classList.add("hidden");
    const liveButton = document.getElementById("run-live-training");
    const supportsLive = Boolean(liveDemoAvailableByCase[activeScenario])
        || activeScenario === "centralized"
        || Boolean(logReplayAvailableByCase[activeScenario]);
    liveButton.classList.remove("hidden");
    liveButton.disabled = !supportsLive || simulationRunning || liveTrainingRunning;

    renderTopology();
    renderTimeline();
    renderScenarioMetrics();
    renderPayloadManifest();
    setPhaseStatus("Ready", "Select Train live");

    const runButton = document.getElementById("run-simulation");
    runButton.onclick = runLiveTraining;
    runButton.disabled = true;
    runButton.innerHTML = `<i class="fa-solid fa-play"></i><span>Run round</span>`;
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
            role="button"
            tabindex="0"
            data-node-type="relay"
            data-node-id="${relay.id}"
            title="Inspect Relay ${relay.id}"
        >
            <i class="fa-solid fa-tower-broadcast"></i>
            <span>R${relay.id}</span>
        </div>
    `).join("");

    auvLayer.innerHTML = topology.auvs.map((auv) => {
        const auvId = Number(auv.id);
        const replayLoss = logReplayLatestLoss.get(auvId);
        const csvLoss = lossById.get(auvId);
        const loss = Number.isFinite(replayLoss) ? replayLoss : csvLoss;
        const previousLoss = logReplayPreviousLoss.get(auvId);
        const trendClass = Number.isFinite(replayLoss) && Number.isFinite(previousLoss)
            ? replayLoss <= previousLoss ? "loss-down" : "loss-up"
            : "";
        const sourceClass = Number.isFinite(replayLoss) ? "training-log" : "";
        const lossBadge = auv.connected && Number.isFinite(loss)
            ? `<span class="loss-badge ${sourceClass} ${trendClass}">L ${loss.toFixed(2)}</span>`
            : "";
        return `
            <div
                id="auv-${auv.id}"
                class="device auv-device ${auv.connected ? "" : "disconnected"}"
                style="left:${auv.x_pct}%;top:${auv.y_pct}%"
                title="${
                    auv.connected
                        ? topology.use_relays
                            ? `AUV ${auv.id} to Relay ${auv.relay_id}`
                            : `AUV ${auv.id} directly to Gateway`
                        : `AUV ${auv.id}: out of range`
                }"
                role="button"
                tabindex="0"
                data-node-type="auv"
                data-node-id="${auv.id}"
            >
                ${lossBadge}
                <i class="fa-solid fa-robot"></i>
                <span>A${auv.id}</span>
            </div>
        `;
    }).join("");

    document.querySelectorAll("[data-node-type][data-node-id]").forEach((node) => {
        const open = () => openNodeInspector(node.dataset.nodeType, Number(node.dataset.nodeId));
        node.addEventListener("click", open);
        node.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                open();
            }
        });
    });
    renderStaticTopologyLinks();
    if (selectedInspectorNode) {
        openNodeInspector(selectedInspectorNode.type, selectedInspectorNode.id, true);
    }
}

function closeNodeInspector() {
    selectedInspectorNode = null;
    document.getElementById("node-inspector").classList.add("hidden");
}

function factRows(rows) {
    return `<dl class="node-facts">${rows.map(([label, value]) => `
        <dt>${label}</dt><dd>${value}</dd>
    `).join("")}</dl>`;
}

function openNodeInspector(nodeType, nodeId, restoring = false) {
    const inspector = document.getElementById("node-inspector");
    const title = document.getElementById("inspector-title");
    const type = document.getElementById("inspector-type");
    const body = document.getElementById("inspector-body");

    if (!restoring) {
        selectedInspectorNode = { type: nodeType, id: nodeId };
    }

    if (nodeType === "auv") {
        const detail = (scenarioData.auv_details || []).find((item) => Number(item.id) === nodeId);
        if (!detail) return;
        type.textContent = detail.connected ? "Participating AUV" : "Disconnected AUV";
        title.textContent = `AUV ${nodeId} - round ${scenarioData.round}`;
        const localLoss = detail.local_loss !== null
            && detail.local_loss !== undefined
            && Number.isFinite(Number(detail.local_loss))
            ? Number(detail.local_loss).toFixed(4)
            : "Not applicable";
        const liveEvent = logReplayLatestEvent.get(nodeId);
        const liveLoss = logReplayLatestLoss.get(nodeId);
        const previousLiveLoss = logReplayPreviousLoss.get(nodeId);
        const liveTrend = Number.isFinite(liveLoss) && Number.isFinite(previousLiveLoss)
            ? liveLoss <= previousLiveLoss ? "down" : "up"
            : "n/a";
        const sampleImages = (detail.sample_image_urls || []).map((url, index) => {
            const source = `${API_BASE}${url.replace(/^\/api/, "")}`;
            const name = (detail.sample_names || [])[index] || `Sample ${index + 1}`;
            return `
                <figure>
                    <img src="${source}" alt="AUV ${nodeId} sample ${index + 1}" loading="lazy">
                    <figcaption title="${name}">${name}</figcaption>
                </figure>
            `;
        }).join("");
        const facts = [
            ["Status", detail.connected ? "Connected" : "Out of range"],
            ["Route", detail.route],
            ["Local images", Number(detail.image_count).toLocaleString()],
            ["Local loss", localLoss],
        ];
        if (Number.isFinite(liveLoss)) {
            facts.push(
                ["Live loss", Number(liveLoss).toFixed(4)],
                ["Previous loss", Number.isFinite(previousLiveLoss) ? Number(previousLiveLoss).toFixed(4) : "--"],
                ["Trend", liveTrend],
            );
        }
        if (liveEvent) {
            facts.push([
                "Batch",
                `epoch ${liveEvent.epoch}/${liveEvent.epochs}, batch ${liveEvent.batch}/${liveEvent.batches}`,
            ]);
        }
        if (detail.loss_components) {
            facts.push([
                "Loss components",
                `box ${Number(detail.loss_components.box).toFixed(3)} / `
                + `cls ${Number(detail.loss_components.cls).toFixed(3)} / `
                + `dfl ${Number(detail.loss_components.dfl).toFixed(3)}`,
            ]);
        }
        facts.push(
            ["Model state", detail.model_state],
            ["Transmits", detail.transmitted_object],
        );
        body.innerHTML = `
            ${factRows(facts)}
            ${sampleImages ? `<div class="sample-images">${sampleImages}</div>` : ""}
        `;
    } else {
        const detail = (scenarioData.relay_details || []).find((item) => Number(item.id) === nodeId);
        if (!detail) return;
        type.textContent = "Relay node";
        title.textContent = `Relay ${nodeId} - round ${scenarioData.round}`;
        const auvList = detail.auv_ids.length
            ? detail.auv_ids.map((id) => `A${id}`).join(", ")
            : "No attached AUV";
        const neighbors = detail.cooperation_neighbors.length
            ? detail.cooperation_neighbors.map((id) => `R${id}`).join(", ")
            : "R2R disabled";
        body.innerHTML = factRows([
            ["Attached AUVs", auvList],
            ["Local images", Number(detail.image_count).toLocaleString()],
            ["Aggregation", detail.aggregation],
            ["R2R neighbors", neighbors],
            ["Gateway link", detail.gateway_link],
        ]);
    }
    inspector.classList.remove("hidden");
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
        metricCard("Local training", formatTime(metrics.train_latency_s)),
        metricCard("AUV to Relay/GW", formatTime(metrics.tau_a2r)),
        metricCard("Relay cooperation", formatTime(metrics.tau_r2r)),
        metricCard("Relay to Gateway", formatTime(metrics.tau_r2g)),
        metricCard("Relay SVD", formatTime(metrics.tau_svd)),
        metricCard("Physical total", formatTime(metrics.round_latency_s)),
        metricCard("Global loss", Number(metrics.loss || 0).toFixed(4)),
        metricCard("mAP50", accuracy),
        metricCard("mAP50-95", `${(Number(metrics.mAP50_95 || 0) * 100).toFixed(2)}%`),
        metricCard("Precision", `${(Number(metrics.precision || 0) * 100).toFixed(2)}%`),
        metricCard("Recall", `${(Number(metrics.recall || 0) * 100).toFixed(2)}%`),
        metricCard("Energy", metrics.energy_j > 0 ? `${metrics.energy_j.toFixed(0)} J` : "Not modeled"),
        metricCard("Compressed demo", `${demoDuration.toFixed(1)} s`),
    ].join("");
}

function renderPayloadManifest() {
    const payload = scenarioData.payload || {};
    document.getElementById("payload-manifest").innerHTML = `
        <h3>Payload being transmitted</h3>
        <dl>
            <dt>Object</dt><dd>${payload.name || "--"}</dd>
            <dt>Encoding</dt><dd>${payload.encoding || "--"}</dd>
            <dt>Contents</dt><dd>${payload.contents || "--"}</dd>
            <dt>Source</dt><dd>${payload.source || "--"}</dd>
        </dl>
    `;
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

function normalizeRelayPair(pair) {
    if (Array.isArray(pair) && pair.length >= 2) {
        return { from: Number(pair[0]), to: Number(pair[1]) };
    }
    if (pair && pair.from !== undefined && pair.to !== undefined) {
        return { from: Number(pair.from), to: Number(pair.to) };
    }
    return null;
}

function createTopologyLink(fromNode, toNode, highlighted = false) {
    if (!fromNode || !toNode) return;
    const scene = document.getElementById("ocean-scene");
    const layer = document.getElementById("topology-link-layer");
    const sceneRect = scene.getBoundingClientRect();
    const start = nodeCenter(fromNode, sceneRect);
    const end = nodeCenter(toNode, sceneRect);
    const deltaX = end.x - start.x;
    const deltaY = end.y - start.y;
    const link = document.createElement("div");
    link.className = `topology-link ${highlighted ? "cooperation" : "feasible"}`;
    link.style.left = `${start.x}px`;
    link.style.top = `${start.y}px`;
    link.style.width = `${Math.hypot(deltaX, deltaY)}px`;
    link.style.transform = `rotate(${Math.atan2(deltaY, deltaX) * 180 / Math.PI}deg)`;
    layer.appendChild(link);
}

function renderStaticTopologyLinks() {
    const layer = document.getElementById("topology-link-layer");
    layer.innerHTML = "";
    if (!scenarioData?.topology?.use_relays) return;

    const cooperationKeys = new Set(
        (scenarioData.topology.cooperation_pairs || [])
            .map(normalizeRelayPair)
            .filter(Boolean)
            .map((pair) => [pair.from, pair.to].sort((a, b) => a - b).join(":"))
    );

    (scenarioData.topology.relay_links || [])
        .map(normalizeRelayPair)
        .filter(Boolean)
        .forEach((pair) => {
            const key = [pair.from, pair.to].sort((a, b) => a - b).join(":");
            createTopologyLink(
                document.getElementById(`relay-${pair.from}`),
                document.getElementById(`relay-${pair.to}`),
                cooperationKeys.has(key)
            );
        });
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
    const connectedAuvs = topology.auvs.filter((auv) => auv.connected);
    const activeRelayIds = new Set(
        connectedAuvs
            .filter((auv) => auv.relay_id !== null && auv.relay_id !== undefined)
            .map((auv) => Number(auv.relay_id))
    );
    const activeRelays = topology.relays
        .filter((relay) =>
            activeRelayIds.has(Number(relay.id))
            && relay.gateway_connected !== false
        )
        .map((relay) => document.getElementById(`relay-${relay.id}`));

    if (phaseId === "uplink_direct") {
        return connectedAuvs.map((auv) => [document.getElementById(`auv-${auv.id}`), gateway]);
    }
    if (phaseId === "downlink_direct") {
        return connectedAuvs.map((auv) => [gateway, document.getElementById(`auv-${auv.id}`)]);
    }
    if (phaseId === "uplink_a2r") {
        return connectedAuvs.map((auv) => [
            document.getElementById(`auv-${auv.id}`),
            relayForAuv(auv.id),
        ]);
    }
    if (phaseId === "relay_cooperate") {
        return (topology.cooperation_pairs || [])
            .map(normalizeRelayPair)
            .filter(Boolean)
            .filter((pair) =>
                activeRelayIds.has(pair.from) && activeRelayIds.has(pair.to)
            )
            .map((pair) => [
                document.getElementById(`relay-${pair.from}`),
                document.getElementById(`relay-${pair.to}`),
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

async function waitForAnimation(durationMs) {
    let remainingMs = durationMs;
    while (remainingMs > 0) {
        if (simulationStopRequested) return false;
        if (simulationPaused) {
            await sleep(60);
            continue;
        }
        const tickMs = Math.min(60, remainingMs);
        await sleep(tickMs);
        remainingMs -= tickMs;
    }
    return !simulationStopRequested;
}

async function animatePhase(phase, speedScale = 1) {
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
    } else if (phase.id === "relay_aggregate" || phase.id === "relay_forward") {
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
            if (fromNode && toNode) {
                createCommunicationLink(
                    fromNode,
                    toNode,
                    Math.max(120, phase.duration_ms * speedScale),
                    SCENARIO_COLORS[activeScenario]
                );
            }
        });
    }

    const completed = await waitForAnimation(Math.max(120, phase.duration_ms * speedScale));
    clearLinks();
    return completed;
}

async function animateTrainingLogPhase(phase, speedScale = 1) {
    const replay = await getTrainingLogReplay(activeScenario, getRequestedRounds());
    const roundEvents = (replay?.events || [])
        .filter((event) => Number(event.round) === Number(currentRound));
    if (!roundEvents.length) {
        return animatePhase(phase, speedScale);
    }

    clearNodeStates();
    clearLinks();
    scenarioData.topology.auvs
        .filter((auv) => auv.connected)
        .forEach((auv) => document.getElementById(`auv-${auv.id}`)?.classList.add("training"));

    for (const event of roundEvents) {
        if (simulationStopRequested) return false;
        while (simulationPaused) {
            await sleep(60);
            if (simulationStopRequested) return false;
        }

        const auvId = Number(event.auv_id);
        const previous = logReplayLatestLoss.get(auvId);
        if (Number.isFinite(previous)) logReplayPreviousLoss.set(auvId, previous);
        logReplayLatestLoss.set(auvId, Number(event.loss));
        logReplayLatestEvent.set(auvId, event);
        renderTopology();
        document.getElementById(`auv-${auvId}`)?.classList.add("training");

        const metrics = scenarioData.metrics || {};
        setPhaseStatus(
            `Round ${currentRound}/40 · Training log`,
            `AUV ${auvId} epoch ${event.epoch}/${event.epochs}, batch ${event.batch}/${event.batches}, local loss ${Number(event.loss).toFixed(4)} · Gateway mAP50 ${(Number(metrics.mAP50 || 0) * 100).toFixed(2)}%`
        );
        await sleep(TRAIN_LOG_DELAY_MS);
    }
    clearNodeStates();
    return true;
}

async function animateCentralizedTrainingPhase(phase, speedScale = 1) {
    clearNodeStates();
    clearLinks();
    const gateway = document.getElementById("gateway");
    gateway.classList.add("processing");

    const metrics = scenarioData.metrics || {};
    const startLoss = Math.max(Number(metrics.loss || 0) + 0.35, 0);
    const endLoss = Number(metrics.loss || 0);
    const steps = CENTRALIZED_TRAIN_STEPS;
    const delayMs = Math.max(70, Math.round((phase.duration_ms * speedScale) / steps));
    for (let step = 1; step <= steps; step += 1) {
        if (simulationStopRequested) return false;
        while (simulationPaused) {
            await sleep(60);
            if (simulationStopRequested) return false;
        }
        const progress = step / steps;
        const shownLoss = startLoss + (endLoss - startLoss) * progress;
        setPhaseStatus(
            `Round ${currentRound}/40 - Gateway training`,
            `Centralized loss ${shownLoss.toFixed(4)} - mAP50 ${(Number(metrics.mAP50 || 0) * 100).toFixed(2)}% - Recall ${(Number(metrics.recall || 0) * 100).toFixed(2)}%`
        );
        await sleep(delayMs);
    }
    clearNodeStates();
    return true;
}

function setSimulationControls(running) {
    const runButton = document.getElementById("run-simulation");
    const resetButton = document.getElementById("reset-simulation");
    const runAllButton = document.getElementById("run-all-simulation");
    const pauseButton = document.getElementById("pause-simulation");
    const stopButton = document.getElementById("stop-simulation");
    const liveButton = document.getElementById("run-live-training");
    const logReplayButton = document.getElementById("run-log-replay");
    runButton.disabled = running || logReplayRunning;
    resetButton.disabled = running || logReplayRunning;
    runAllButton.disabled = running || logReplayRunning;
    logReplayButton.disabled = running
        || logReplayRunning
        || !logReplayAvailableByCase[activeScenario];
    pauseButton.disabled = !running;
    stopButton.disabled = !running && !logReplayRunning;
    liveButton.disabled = running
        || logReplayRunning
        || !(liveDemoAvailableByCase[activeScenario] || activeScenario === "centralized");
    if (!running) {
        pauseButton.innerHTML = `<i class="fa-solid fa-pause"></i>`;
        pauseButton.title = "Pause replay";
    }
}

async function playLoadedRound(speedScale = 1) {
    for (let index = 0; index < scenarioData.phases.length; index += 1) {
        if (simulationStopRequested) return false;
        const phase = scenarioData.phases[index];
        setTimelineState(index);
        const roundText = activeScenario === "centralized" ? "" : `Round ${currentRound}/40 - `;
        setPhaseStatus(
            `${roundText}Phase ${index + 1}/${scenarioData.phases.length}`,
            phase.label
        );
        if (phase.id === "train" && activeScenario !== "centralized") {
            if (!await animateTrainingLogPhase(phase, speedScale)) return false;
        } else if (phase.id === "gateway_train" && activeScenario === "centralized") {
            if (!await animateCentralizedTrainingPhase(phase, speedScale)) return false;
        } else if (!await animatePhase(phase, speedScale)) {
            return false;
        }
    }
    clearNodeStates();
    setTimelineState(scenarioData.phases.length);
    return true;
}

async function runSimulation() {
    if (!scenarioData || simulationRunning || liveTrainingRunning || logReplayRunning) return;
    simulationRunning = true;
    simulationPaused = false;
    simulationStopRequested = false;
    replayAllRunning = false;
    logReplayLatestLoss = new Map();
    logReplayPreviousLoss = new Map();
    setSimulationControls(true);
    document.getElementById("run-simulation").innerHTML =
        `<i class="fa-solid fa-spinner fa-spin"></i><span>Running</span>`;

    const completed = await playLoadedRound(1);
    finishSimulation(completed);
}

function updateLiveTrainingButton(job = null) {
    const button = document.getElementById("run-live-training");
    if (liveTrainingRunning) {
        button.disabled = true;
        button.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i><span>Training</span>`;
        return;
    }
    button.disabled = !(liveDemoAvailableByCase[activeScenario] || activeScenario === "centralized");
    button.innerHTML = `<i class="fa-solid fa-microchip"></i><span>Train live</span>`;
}

async function runLiveTraining() {
    if (!liveTrainingAvailable) return;
    if (liveTrainingRunning && liveTrainingJobId) {
        await postJson(`/demo/live-round/${liveTrainingJobId}/cancel`);
        return;
    }

    try {
        const job = await postJson(`/demo/live-round/start?baseline=${encodeURIComponent(activeScenario)}`);
        liveTrainingJobId = job.job_id;
        liveTrainingRunning = true;
        updateLiveTrainingButton(job);

        while (liveTrainingRunning && liveTrainingJobId) {
            const current = await getJson(`/demo/live-round/${liveTrainingJobId}`);
            const completed = (current.completed_auvs || []).length;
            setPhaseStatus(
                `Live training · ${current.status}`,
                current.current_auv === null
                    ? current.message
                    : `${current.message} · completed AUV jobs: ${completed}`
            );
            updateLiveTrainingButton(current);
            if (["completed", "failed", "cancelled"].includes(current.status)) {
                liveTrainingRunning = false;
                liveTrainingJobId = null;
                updateLiveTrainingButton();
                break;
            }
            await sleep(1200);
        }
    } catch (error) {
        liveTrainingRunning = false;
        liveTrainingJobId = null;
        updateLiveTrainingButton();
        setPhaseStatus("Live training unavailable", error.message);
    }
}

async function runDemoTraining() {
    if (!scenarioData || simulationRunning || liveTrainingRunning || logReplayRunning) return;
    if (!(liveDemoAvailableByCase[activeScenario] || activeScenario === "centralized")) return;

    try {
        const requestedRounds = getRequestedRounds();
        simulationRunning = true;
        liveTrainingRunning = true;
        simulationPaused = false;
        simulationStopRequested = false;
        replayAllRunning = true;
        logReplayLatestLoss = new Map();
        logReplayPreviousLoss = new Map();
        logReplayLatestEvent = new Map();
        setSimulationControls(true);
        updateLiveTrainingButton();

        for (let round = 1; round <= requestedRounds; round += 1) {
            if (simulationStopRequested) break;
            currentRound = round;
            logReplayLatestLoss = new Map();
            logReplayPreviousLoss = new Map();
            logReplayLatestEvent = new Map();
            await loadScenario();
            setSimulationControls(true);
            updateLiveTrainingButton();
            if (!await playLoadedRound(1)) break;
            if (round < requestedRounds) await sleep(ROUND_TRANSITION_DELAY_MS);
        }

        finishSimulation(!simulationStopRequested);
    } catch (error) {
        simulationRunning = false;
        liveTrainingRunning = false;
        setSimulationControls(false);
        updateLiveTrainingButton();
        setPhaseStatus("Train live unavailable", error.message);
    }
}

function updateLogReplayButton() {
    const button = document.getElementById("run-log-replay");
    if (logReplayRunning) {
        button.disabled = false;
        button.innerHTML = `<i class="fa-solid fa-stop"></i><span>Stop log</span>`;
        return;
    }
    button.disabled = simulationRunning
        || liveTrainingRunning
        || !logReplayAvailableByCase[activeScenario];
    button.innerHTML = `<i class="fa-solid fa-chart-line"></i><span>Replay log</span>`;
}

async function runTrainingLogReplay() {
    if (logReplayRunning) {
        logReplayStopRequested = true;
        return;
    }
    if (
        !scenarioData
        || simulationRunning
        || liveTrainingRunning
        || !logReplayAvailableByCase[activeScenario]
    ) return;

    logReplayRunning = true;
    logReplayStopRequested = false;
    logReplayLatestLoss = new Map();
    logReplayPreviousLoss = new Map();
    setSimulationControls(false);
    updateLogReplayButton();
    clearLinks();
    clearNodeStates();

    try {
        const replay = await getJson(`/demo/training-log/${activeScenario}?max_rounds=${getRequestedRounds()}`);
        const events = replay.events || [];
        if (!events.length) throw new Error("No batch-loss events found in log.");
        const firstRound = events[0].round;
        currentRound = firstRound;
        await loadScenario();
        setSimulationControls(false);
        updateLogReplayButton();

        for (const event of events) {
            if (logReplayStopRequested) break;
            if (event.round !== currentRound) {
                currentRound = event.round;
                await loadScenario();
                setSimulationControls(false);
                updateLogReplayButton();
            }

            const auvId = Number(event.auv_id);
            const previous = logReplayLatestLoss.get(auvId);
            if (Number.isFinite(previous)) logReplayPreviousLoss.set(auvId, previous);
            logReplayLatestLoss.set(auvId, Number(event.loss));
            renderTopology();

            const node = document.getElementById(`auv-${auvId}`);
            if (node) node.classList.add("training");
            setPhaseStatus(
                `Log replay · round ${event.round}/10`,
                `AUV ${auvId} epoch ${event.epoch}/${event.epochs}, batch ${event.batch}/${event.batches}, loss ${Number(event.loss).toFixed(4)}`
            );
            await sleep(STANDALONE_LOG_REPLAY_DELAY_MS);
        }

        var finalIndex = logReplayStopRequested ? "Stopped" : "Complete";
        var finalLabel = logReplayStopRequested
            ? `Training-log replay stopped at round ${currentRound}`
            : `${getRequestedRounds()} rounds replayed from real training log`;
    } catch (error) {
        var finalIndex = "Log replay unavailable";
        var finalLabel = error.message;
    } finally {
        clearNodeStates();
        logReplayRunning = false;
        logReplayStopRequested = false;
        logReplayLatestLoss = new Map();
        logReplayPreviousLoss = new Map();
        await loadScenario();
        setSimulationControls(false);
        updateLogReplayButton();
        setPhaseStatus(finalIndex, finalLabel);
    }
}

async function runAllSimulations() {
    if (
        !scenarioData
        || simulationRunning
        || liveTrainingRunning
        || logReplayRunning
        || activeScenario === "centralized"
    ) return;
    simulationRunning = true;
    simulationPaused = false;
    simulationStopRequested = false;
    replayAllRunning = true;
    logReplayLatestLoss = new Map();
    logReplayPreviousLoss = new Map();
    setSimulationControls(true);
    document.getElementById("run-simulation").innerHTML =
        `<i class="fa-solid fa-spinner fa-spin"></i><span>Replaying</span>`;

    for (let round = currentRound; round <= 40; round += 1) {
        if (simulationStopRequested) break;
        currentRound = round;
        logReplayLatestLoss = new Map();
        logReplayPreviousLoss = new Map();
        await loadScenario();
        setSimulationControls(true);
        if (!await playLoadedRound(0.08)) break;
    }
    finishSimulation(!simulationStopRequested);
}

function finishSimulation(completed) {
    const runButton = document.getElementById("run-simulation");
    clearLinks();
    clearNodeStates();
    simulationRunning = false;
    simulationPaused = false;
    liveTrainingRunning = false;
    replayAllRunning = false;
    setSimulationControls(false);
    updateLiveTrainingButton();
    if (completed) {
        setPhaseStatus(
            "Complete",
            activeScenario === "centralized"
                ? "Raw dataset received at Gateway"
                : `Round ${currentRound} completed`
        );
    } else {
        setPhaseStatus("Stopped", `Replay stopped at round ${currentRound}`);
    }
    runButton.innerHTML = `<i class="fa-solid fa-play"></i><span>Run round</span>`;
}

function togglePauseSimulation() {
    if (!simulationRunning) return;
    simulationPaused = !simulationPaused;
    const pauseButton = document.getElementById("pause-simulation");
    pauseButton.innerHTML = simulationPaused
        ? `<i class="fa-solid fa-play"></i>`
        : `<i class="fa-solid fa-pause"></i>`;
    pauseButton.title = simulationPaused ? "Resume replay" : "Pause replay";
    setPhaseStatus(
        simulationPaused ? "Paused" : "Running",
        simulationPaused ? `Paused at round ${currentRound}` : `Round ${currentRound} resumed`
    );
}

function stopSimulation() {
    if (logReplayRunning) {
        logReplayStopRequested = true;
        return;
    }
    if (!simulationRunning) return;
    simulationStopRequested = true;
    simulationPaused = false;
}

async function resetSimulation() {
    if (simulationRunning || liveTrainingRunning || logReplayRunning) return;
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
    document.getElementById("run-all-simulation").addEventListener("click", runAllSimulations);
    document.getElementById("run-log-replay").addEventListener("click", runTrainingLogReplay);
    document.getElementById("run-live-training").addEventListener("click", runDemoTraining);
    document.getElementById("pause-simulation").addEventListener("click", togglePauseSimulation);
    document.getElementById("stop-simulation").addEventListener("click", stopSimulation);
    document.getElementById("close-inspector").addEventListener("click", closeNodeInspector);
    document.getElementById("round-count").addEventListener("input", async (event) => {
        if (simulationRunning || liveTrainingRunning || logReplayRunning) return;
        event.target.value = String(getRequestedRounds());
        currentRound = Math.min(currentRound, getRequestedRounds());
        await loadScenario();
    });
    window.addEventListener("resize", () => {
        if (scenarioData && !simulationRunning) renderStaticTopologyLinks();
    });

    try {
        const summary = await getJson("/demo/summary");
        liveDemoAvailableByCase = Object.fromEntries(
            Object.entries(summary.cases || {}).map(([name, item]) => [
                name,
                Boolean(item.live_demo_available || item.log_replay_available),
            ])
        );
        logReplayAvailableByCase = Object.fromEntries(
            Object.entries(summary.cases || {}).map(([name, item]) => [
                name,
                Boolean(item.log_replay_available),
            ])
        );
        setBackendState(Boolean(summary.ml_available), summary.model_path, true);
        updateLiveTrainingButton();
        updateLogReplayButton();
    } catch (error) {
        liveDemoAvailableByCase = {};
        logReplayAvailableByCase = {};
        setBackendState(false);
        updateLiveTrainingButton();
        updateLogReplayButton();
    }
}

init();
