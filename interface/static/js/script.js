document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('csvFile');
    const dropArea = document.getElementById('dropArea');
    const targetSection = document.getElementById('targetSection');
    const targetSelect = document.getElementById('targetColumn');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const form = document.getElementById('uploadForm');
    const loadingOverlay = document.getElementById('loadingOverlay');

    // Drag & Drop Visuals
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropArea.classList.add('dragover');
    }

    function unhighlight(e) {
        dropArea.classList.remove('dragover');
    }

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            fileNameDisplay.textContent = `Selected: ${file.name}`;
            
            // Read headers for dropdown
            if (file.name.endsWith('.csv')) {
                parseHeaders(file);
            } else {
                alert('Please upload a valid CSV file.');
            }
        }
    }

    function parseHeaders(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const text = e.target.result;
            // Get first line
            const firstLine = text.split('\n')[0];
            // Split by comma (handles basic CSV, not quotes/complex)
            // For robust parsing, use a library, but this suffices for demo
            const headers = firstLine.split(',').map(h => h.trim().replace(/['"]+/g, ''));
            
            // Populate Dropdown
            targetSelect.innerHTML = '<option value="" selected disabled>Choose the column to predict...</option>';
            headers.forEach(header => {
                if (header) {
                    const option = document.createElement('option');
                    option.value = header;
                    option.textContent = header;
                    targetSelect.appendChild(option);
                }
            });

            // Show section
            targetSection.classList.remove('d-none');
        };
        // Read just the first 5kb to get headers
        reader.readAsText(file.slice(0, 5000));
    }

    // Handle Form Submit
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show Loading
        loadingOverlay.classList.remove('d-none');
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('target_column', targetSelect.value);
        
        // Collect selected models
        const selectedModels = document.querySelectorAll('input[name="models"]:checked');
        selectedModels.forEach(model => {
            formData.append('models[]', model.value);
        });
        
        const progressBar = document.getElementById('progressBar');
        const statusMessage = document.getElementById('statusMessage');

        fetch('/process', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'queued') {
                const jobId = data.job_id;
                pollStatus(jobId);
            } else {
                alert('Error: ' + data.error);
                loadingOverlay.classList.add('d-none');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An unexpected error occurred. Check console.');
            loadingOverlay.classList.add('d-none');
        });

        function pollStatus(jobId) {
            const interval = setInterval(() => {
                fetch(`/status/${jobId}`)
                .then(res => res.json())
                .then(status => {
                    if (status.status === 'completed') {
                        clearInterval(interval);
                        window.location.href = status.redirect;
                    } else if (status.status === 'failed') {
                        clearInterval(interval);
                        alert('Training Failed: ' + status.error);
                        loadingOverlay.classList.add('d-none');
                    } else {
                        // Update Progress UI
                        progressBar.style.width = status.progress + '%';
                        statusMessage.textContent = status.message;
                    }
                })
                .catch(err => {
                    clearInterval(interval);
                    console.error('Polling error:', err);
                });
            }, 1000); // Poll every second
        }
    });
});
