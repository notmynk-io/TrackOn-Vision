const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const tolerance = document.getElementById('tolerance');
const toleranceValue = document.getElementById('toleranceValue');
const uploadStatus = document.getElementById('uploadStatus');

const cameraType = document.getElementById('cameraType');
const ipUrlGroup = document.getElementById('ipUrlGroup');
const ipUrl = document.getElementById('ipUrl');
const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const cameraStatus = document.getElementById('cameraStatus');

const toggleProcessingBtn = document.getElementById('toggleProcessingBtn');
const getStatsBtn = document.getElementById('getStatsBtn');
const resetStatsBtn = document.getElementById('resetStatsBtn');
const statsCard = document.getElementById('statsCard');
const statsContent = document.getElementById('statsContent');
const processingStatus = document.getElementById('processingStatus');

let isProcessing = false;
let isCameraRunning = false;

// Tolerance slider
tolerance.addEventListener('input', function() {
    toleranceValue.textContent = this.value;
});

// Camera type selection
cameraType.addEventListener('change', function() {
    if (this.value === 'ip') {
        ipUrlGroup.style.display = 'block';
    } else {
        ipUrlGroup.style.display = 'none';
    }
});

// Upload form submission
uploadForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('tolerance', tolerance.value);
    
    showStatus(uploadStatus, 'Uploading and processing...', 'info');
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus(uploadStatus, data.message, 'success');
            toggleProcessingBtn.disabled = false;
        } else {
            showStatus(uploadStatus, data.message, 'error');
        }
    } catch (error) {
        showStatus(uploadStatus, 'Error uploading file: ' + error.message, 'error');
    }
});

// Start camera
startCameraBtn.addEventListener('click', async function() {
    const camType = cameraType.value;
    const url = camType === 'ip' ? ipUrl.value : null;
    
    if (camType === 'ip' && !url) {
        showStatus(cameraStatus, 'Please enter IP camera URL', 'error');
        return;
    }
    
    showStatus(cameraStatus, 'Starting camera...', 'info');
    
    try {
        const response = await fetch('/start_camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                camera_type: camType,
                ip_url: url
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus(cameraStatus, data.message, 'success');
            isCameraRunning = true;
            startCameraBtn.disabled = true;
            stopCameraBtn.disabled = false;
            updateProcessingStatus('active');
        } else {
            showStatus(cameraStatus, data.message, 'error');
        }
    } catch (error) {
        showStatus(cameraStatus, 'Error starting camera: ' + error.message, 'error');
    }
});

// Stop camera
stopCameraBtn.addEventListener('click', async function() {
    try {
        const response = await fetch('/stop_camera', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus(cameraStatus, data.message, 'success');
            isCameraRunning = false;
            startCameraBtn.disabled = false;
            stopCameraBtn.disabled = true;
            toggleProcessingBtn.disabled = true;
            isProcessing = false;
            toggleProcessingBtn.textContent = 'Start Processing';
            updateProcessingStatus('idle');
        } else {
            showStatus(cameraStatus, data.message, 'error');
        }
    } catch (error) {
        showStatus(cameraStatus, 'Error stopping camera: ' + error.message, 'error');
    }
});

// Toggle processing
toggleProcessingBtn.addEventListener('click', async function() {
    try {
        const response = await fetch('/toggle_processing', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            isProcessing = data.is_processing;
            this.textContent = isProcessing ? 'Stop Processing' : 'Start Processing';
            updateProcessingStatus(isProcessing ? 'processing' : 'active');
        }
    } catch (error) {
        console.error('Error toggling processing:', error);
    }
});

// Get statistics
getStatsBtn.addEventListener('click', async function() {
    try {
        const response = await fetch('/get_stats');
        const data = await response.json();
        
        if (data.success) {
            const stats = data.stats;
            statsContent.innerHTML = `
                <p><strong>Total Frames:</strong> ${stats.total_frames}</p>
                <p><strong>Matched Frames:</strong> ${stats.matched_frames}</p>
                <p><strong>Match Rate:</strong> ${stats.match_rate.toFixed(2)}%</p>
                <p><strong>Avg Processing Time:</strong> ${(stats.avg_processing_time * 1000).toFixed(2)}ms</p>
                <p><strong>Processing FPS:</strong> ${stats.fps.toFixed(2)}</p>
            `;
            statsCard.style.display = 'block';
        } else {
            alert(data.message);
        }
    } catch (error) {
        console.error('Error getting stats:', error);
    }
});

// Reset statistics
resetStatsBtn.addEventListener('click', async function() {
    try {
        const response = await fetch('/reset_stats', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            statsCard.style.display = 'none';
            alert('Statistics reset successfully');
        }
    } catch (error) {
        console.error('Error resetting stats:', error);
    }
});

// Helper functions
function showStatus(element, message, type) {
    element.textContent = message;
    element.className = 'status-message ' + type;
    element.style.display = 'block';
}

function updateProcessingStatus(status) {
    processingStatus.className = 'status-badge ' + status;
    
    const statusText = {
        'idle': 'Idle',
        'active': 'Camera Active',
        'processing': 'Processing...'
    };
    
    processingStatus.textContent = statusText[status] || 'Unknown';
}
