document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('csvFile');
    const dropArea = document.getElementById('dropArea');
    const targetSection = document.getElementById('targetSection');
    const targetSelect = document.getElementById('targetColumn');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const form = document.getElementById('uploadForm');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const storyContainer = document.getElementById('storyContainer');

    // Drag & Drop Handling
    if (dropArea) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, e => {
                e.preventDefault();
                e.stopPropagation();
            }, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
        });

        dropArea.addEventListener('drop', e => {
            const files = e.dataTransfer.files;
            if (files.length > 0) handleFiles(files);
        }, false);
    }

    if (fileInput) {
        fileInput.addEventListener('change', function() {
            handleFiles(this.files);
        });
    }

    function handleFiles(files) {
        const file = files[0];
        if (!file.name.endsWith('.csv')) {
            showNotification('Invalid format. Please provide a CSV stream.', 'error');
            return;
        }

        // Update UI
        fileNameDisplay.textContent = `Vectored: ${file.name}`;
        fileNameDisplay.classList.remove('d-none');
        
        parseHeaders(file);
    }

    function parseHeaders(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const text = e.target.result;
            const headers = parseCSVLine(text.split('\n')[0]);
            
            targetSelect.innerHTML = '<option value="" selected disabled>Select objective column...</option>';
            headers.forEach(header => {
                if (header?.trim()) {
                    const option = document.createElement('option');
                    option.value = header.trim();
                    option.textContent = header.trim();
                    targetSelect.appendChild(option);
                }
            });

            targetSection.classList.remove('d-none');
        };
        reader.readAsText(file.slice(0, 8000));
    }

    function parseCSVLine(text) {
        const result = [];
        let cell = '', quote = false;
        for (let i = 0; i < text.length; i++) {
            let char = text[i];
            if (char === '"') quote = !quote;
            else if (char === ',' && !quote) {
                result.push(cell.trim().replace(/^"|"$/g, ''));
                cell = '';
            } else cell += char;
        }
        result.push(cell.trim().replace(/^"|"$/g, ''));
        return result;
    }

    // Form Intercept for Asynchronous Non-Blocking Processing
    form?.addEventListener('submit', async function(e) {
        e.preventDefault();

        const file = fileInput?.files?.[0];
        const targetVal = targetSelect?.value;
        const checkedModels = document.querySelectorAll('input[name="models"]:checked');

        if (!file) {
            showNotification('Please select a CSV file first.', 'error');
            return;
        }
        if (!targetVal) {
            showNotification('Please select an objective prediction target.', 'error');
            return;
        }
        if (checkedModels.length === 0) {
            showNotification('Please select at least one model architecture.', 'error');
            return;
        }

        // Show loading overlay and initialize dynamic table
        loadingOverlay.classList.remove('d-none');
        initializeOrchestrationTable();

        const formData = new FormData(form);

        try {
            const response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Accept': 'application/json'
                },
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `Server returned error (${response.status})`);
            }

            const data = await response.json();
            const jobId = data.job_id;
            
            // Poll for completion
            pollJobStatus(jobId);

        } catch (err) {
            loadingOverlay.classList.add('d-none');
            showNotification(err.message, 'error');
        }
    });

    let currentStepList = [];

    // Dynamically build the loading table based on selected models
    function initializeOrchestrationTable() {
        currentStepList = [
            { id: 'ingest', match: 'ingest', t: "Ingesting dataset stream...", icon: "fa-file-import" },
            { id: 'eda', match: 'generating', t: "Feature correlation & spectrum analysis...", icon: "fa-chart-network" },
            { id: 'config', match: 'configuring', t: "Configuring model architectures...", icon: "fa-sliders" }
        ];

        // Collect all checked models dynamically
        const selectedCheckboxes = document.querySelectorAll('input[name="models"]:checked');
        const selectedModelNames = [];
        selectedCheckboxes.forEach(cb => {
            cb.value.split(',').forEach(m => {
                const trimmed = m.strip ? m.strip() : m.trim();
                if (trimmed && !selectedModelNames.includes(trimmed)) {
                    selectedModelNames.push(trimmed);
                }
            });
        });

        if (selectedModelNames.length > 0) {
            selectedModelNames.forEach((name, i) => {
                currentStepList.push({
                    id: `model_${i}`,
                    match: name.toLowerCase(),
                    t: `Training ${name}...`,
                    icon: "fa-microchip"
                });
            });
        } else {
            currentStepList.push({ id: 'model_generic', match: 'training', t: "Training candidate models...", icon: "fa-microchip" });
        }

        currentStepList.push(
            { id: 'diag', match: 'rendering', t: "Model diagnostics & SHAP explainability...", icon: "fa-gauge-high" },
            { id: 'export', match: 'exporting', t: "Ranking leaderboard & model export...", icon: "fa-trophy" }
        );

        storyContainer.innerHTML = '';
        currentStepList.forEach(s => {
            const tr = document.createElement('tr');
            tr.id = `step-${s.id}`;
            tr.innerHTML = `
                <td class="text-white small">
                    <i class="fas ${s.icon} me-3 opacity-30"></i> ${s.t}
                </td>
                <td>
                    <div class="status-indicator">
                        <div class="dot"></div>
                        <span class="x-small text-muted">Awaiting</span>
                    </div>
                </td>
            `;
            storyContainer.appendChild(tr);
        });

        // Activate first step
        updateStep(currentStepList[0].id, 'active');
    }

    // Real-time backend status poller
    function pollJobStatus(jobId) {
        const pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/job_status/${jobId}`);
                if (!res.ok) {
                    clearInterval(pollInterval);
                    loadingOverlay.classList.add('d-none');
                    showNotification("Job status check failed. Please refresh.", "error");
                    return;
                }

                const data = await res.json();

                if (data.status === 'running') {
                    handleStepProgress(data.step);
                } else if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    markAllComplete();
                    setTimeout(() => {
                        window.location.href = `/dashboard?job_id=${jobId}`;
                    }, 400);
                } else if (data.status === 'failed') {
                    clearInterval(pollInterval);
                    loadingOverlay.classList.add('d-none');
                    showNotification(data.error || "Pipeline execution failed.", "error");
                }
            } catch (pollErr) {
                console.warn("Poll connection retry:", pollErr);
            }
        }, 800);
    }

    function handleStepProgress(stepMessage) {
        if (!stepMessage) return;
        const lowerMsg = stepMessage.toLowerCase();
        let matchedIdx = -1;

        for (let i = 0; i < currentStepList.length; i++) {
            if (lowerMsg.includes(currentStepList[i].match)) {
                matchedIdx = i;
                break;
            }
        }

        if (matchedIdx !== -1) {
            for (let i = 0; i < matchedIdx; i++) {
                updateStep(currentStepList[i].id, 'complete');
            }
            updateStep(currentStepList[matchedIdx].id, 'active');
        }
    }

    function markAllComplete() {
        currentStepList.forEach(s => {
            updateStep(s.id, 'complete');
        });
    }

    function updateStep(id, state) {
        const tr = document.getElementById(`step-${id}`);
        if (!tr) return;
        
        const indicator = tr.querySelector('.status-indicator');
        const text = indicator.querySelector('span');
        
        indicator.classList.remove('active', 'complete');
        if (state === 'active') {
            indicator.classList.add('active');
            text.textContent = 'Processing';
            text.classList.remove('text-muted');
            text.classList.add('text-primary');
        } else if (state === 'complete') {
            indicator.classList.add('complete');
            text.textContent = 'OK';
            text.classList.remove('text-primary');
            text.classList.add('text-success');
        }
    }

    function showNotification(msg, type) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type === 'error' ? 'danger' : 'warning'} alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-4 shadow-lg`;
        alertDiv.style.zIndex = '99999';
        alertDiv.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas fa-triangle-exclamation me-2"></i>
                <div class="small fw-semibold">${msg}</div>
                <button type="button" class="btn-close ms-3" data-bs-dismiss="alert"></button>
            </div>
        `;
        document.body.appendChild(alertDiv);
        setTimeout(() => alertDiv.remove(), 6000);
    }
});

