document.addEventListener('DOMContentLoaded', () => {
    // Canvas setup
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    
    // Drawing settings
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000';
    
    // UI elements
    const clearBtn = document.getElementById('clear-btn');
    const predictBtn = document.getElementById('predict-btn');
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    const predictionLabel = document.getElementById('prediction-label');
    const confidenceLevel = document.getElementById('confidence-level');
    const confidenceText = document.getElementById('confidence-text');
    
    // Helper functions
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getCoordinates(e);
    }
    
    function draw(e) {
        if (!isDrawing) return;
        
        // Prevent scrolling when drawing on touch devices
        e.preventDefault();
        
        const [x, y] = getCoordinates(e);
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        
        [lastX, lastY] = [x, y];
    }
    
    function stopDrawing() {
        isDrawing = false;
    }
    
    function getCoordinates(e) {
        let x, y;
        
        // Get coordinates for both mouse and touch events
        if (e.type.includes('touch')) {
            const rect = canvas.getBoundingClientRect();
            x = e.touches[0].clientX - rect.left;
            y = e.touches[0].clientY - rect.top;
        } else {
            x = e.offsetX;
            y = e.offsetY;
        }
        
        return [x, y];
    }
    
    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        hideResult();
    }
    
    function hideResult() {
        loadingDiv.classList.add('hidden');
        resultDiv.classList.add('hidden');
    }
    
    function showLoading() {
        loadingDiv.classList.remove('hidden');
        resultDiv.classList.add('hidden');
    }
    
    function showResult(prediction) {
        loadingDiv.classList.add('hidden');
        resultDiv.classList.remove('hidden');
        
        // Update prediction display
        predictionLabel.textContent = prediction.label;
        
        const confidencePercent = Math.round(prediction.confidence * 100);
        confidenceLevel.style.width = `${confidencePercent}%`;
        confidenceText.textContent = `Confidence: ${confidencePercent}%`;
        
        // Speak the prediction if confidence is high enough
        if (prediction.speak) {
            speakPrediction(prediction.label, confidencePercent);
        }
    }
    
    function speakPrediction(label, confidence) {
        // Use the browser's Speech Synthesis API
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(`${label}, ${confidence} percent confident.`);
            window.speechSynthesis.speak(utterance);
        }
    }
    
    async function predictSketch() {
        // Check if canvas is empty
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        const isEmpty = !imageData.some(channel => channel !== 0 && channel !== 255);
        
        if (isEmpty) {
            alert('Please draw something first!');
            return;
        }
        
        showLoading();
        
        try {
            // Convert canvas to base64 image data
            const imageBase64 = canvas.toDataURL('image/png');
            
            // Send to server for prediction
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageBase64 })
            });
            
            if (!response.ok) {
                throw new Error('Prediction failed');
            }
            
            const prediction = await response.json();
            showResult(prediction);
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error predicting sketch. Please try again.');
            hideResult();
        }
    }
    
    // Event listeners for drawing
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('touchstart', startDrawing);
    
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('touchmove', draw);
    
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('touchend', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Prevent touch scrolling while drawing
    canvas.addEventListener('touchstart', e => e.preventDefault());
    
    // Button event listeners
    clearBtn.addEventListener('click', clearCanvas);
    predictBtn.addEventListener('click', predictSketch);
    
    // Initial canvas clear
    clearCanvas();
}); 