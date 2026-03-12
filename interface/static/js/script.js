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

    // Form Intercept
    form?.addEventListener('submit', function() {
        loadingOverlay.classList.remove('d-none');
        startEngineOrchestration();
    });

    // Advanced "Story" Table Sequence
    function startEngineOrchestration() {
        const sequences = [
            { id: 'data', t: "Establishing data link...", icon: "fa-link" },
            { id: 'import', t: "Injecting CSV stream...", icon: "fa-file-import" },
            { id: 'nulls', t: "Heuristic null scanning...", icon: "fa-filter" },
            { id: 'corr', t: "Correlation matrix calc...", icon: "fa-chart-network" },
            { id: 'rf', t: "Random Forest assembly...", icon: "fa-microchip" },
            { id: 'svm', t: "SVM hyperplane search...", icon: "fa-vector-square" },
            { id: 'roc', t: "ROC/AUC benchmarking...", icon: "fa-gauge-high" },
            { id: 'sync', t: "Diagnostic sync...", icon: "fa-sync" }
        ];

        // Initialize table
        storyContainer.innerHTML = '';
        sequences.forEach(s => {
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

        let idx = 0;
        const interval = setInterval(() => {
            if (idx > 0) {
                updateStep(sequences[idx-1].id, 'complete');
            }
            
            if (idx < sequences.length) {
                updateStep(sequences[idx].id, 'active');
                idx++;
            } else {
                clearInterval(interval);
            }
        }, 1200);
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
        // Simple fallback alert for now, could be a premium toast later
        alert(`${type.toUpperCase()}: ${msg}`);
    }
});

