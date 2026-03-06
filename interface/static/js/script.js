document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('csvFile');
    const dropArea = document.getElementById('dropArea');
    const targetSection = document.getElementById('targetSection');
    const targetSelect = document.getElementById('targetColumn');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const form = document.getElementById('uploadForm');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const storyContainer = document.getElementById('storyContainer');

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
            
            // Validate file type
            if (!file.name.endsWith('.csv')) {
                alert('Please upload a valid CSV file.');
                return;
            }

            // Update UI
            fileNameDisplay.textContent = `Selected: ${file.name}`;
            fileNameDisplay.classList.remove('d-none');
            fileNameDisplay.classList.add('fade-in-up');
            
            // Parse headers
            parseHeaders(file);
        }
    }

    function parseHeaders(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const text = e.target.result;
            // Get first line
            const firstLine = text.split('\n')[0];
            
            // Robust CSV parsing handles quoted commas
            const headers = parseCSVLine(firstLine);
            
            // Populate Dropdown
            targetSelect.innerHTML = '<option value="" selected disabled>Select column to predict...</option>';
            headers.forEach(header => {
                if (header && header.trim() !== '') {
                    const option = document.createElement('option');
                    option.value = header.trim();
                    option.textContent = header.trim();
                    targetSelect.appendChild(option);
                }
            });

            // Show section with animation
            targetSection.classList.remove('d-none');
            targetSection.classList.add('fade-in-up');
        };
        // Read just the first 5kb to get headers
        reader.readAsText(file.slice(0, 5000));
    }

    // Robust CSV Line Parser
    function parseCSVLine(text) {
        const result = [];
        let cell = '';
        let quote = false;

        for (let i = 0; i < text.length; i++) {
            let char = text[i];
            
            if (char === '"') {
                quote = !quote;
            } else if (char === ',' && !quote) {
                result.push(cell.trim().replace(/^"|"$/g, '')); // Remove outer quotes
                cell = '';
            } else {
                cell += char;
            }
        }
        result.push(cell.trim().replace(/^"|"$/g, ''));
        return result;
    }

    // Handle Form Submit
    form.addEventListener('submit', function(e) {
        // Show loading overlay
        loadingOverlay.classList.remove('d-none');
        startStory();
        
        // Let form submit naturally -> Flask processes -> Redirects
    });

    // "Story" Animation
    function startStory() {
        const stories = [
            "Validating dataset structure...",
            "Detecting feature types...",
            "Imputing missing values...",
            "Scaling numerical features...",
            "Encoding categorical variables...",
            "Training Random Forest...",
            "Training SVM...",
            "Training Gradient Boosting...",
            "Evaluating model performance...",
            "Generating feature importance...",
            "Calculating ROC curves...",
            "Finalizing report..."
        ];

        let index = 0;
        const storyInterval = setInterval(() => {
            if (index < stories.length) {
                addStoryItem(stories[index]);
                index++;
            } else {
                clearInterval(storyInterval);
            }
        }, 1500); // New message every 1.5 seconds
    }

    function addStoryItem(text) {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
        
        const item = document.createElement('div');
        item.className = 'story-item fade-in-up';
        item.innerHTML = `
            <span class="story-icon text-success"><i class="fas fa-check"></i></span>
            <div>
                <div class="story-time text-white-50">${timeString}</div>
                <div class="story-text text-white">${text}</div>
            </div>
        `;
        
        storyContainer.prepend(item); // Add to top
    }
});
