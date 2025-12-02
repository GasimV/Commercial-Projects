// Advanced Reorder Prediction System - Frontend Application

const API_BASE = 'http://localhost:8000';

class ReorderApp {
    constructor() {
        this.data = null;
        this.models = {
            trained: false,
            metrics: null
        };

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkHealth();
    }

    // ===== Setup =====

    setupEventListeners() {
        // File upload
        const fileInput = document.getElementById('fileInput');
        const uploadZone = document.getElementById('uploadZone');

        uploadZone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFileUpload(e.target.files[0]));

        // Drag and drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                this.handleFileUpload(e.dataTransfer.files[0]);
            }
        });

        // Training
        document.getElementById('trainBtn').addEventListener('click', () => this.trainModels());

        // Tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => this.switchTab(tab.dataset.tab));
        });

        // Predictions
        document.getElementById('predictCustomerBtn').addEventListener('click', () => {
            const customerId = document.getElementById('customerSearch').value;
            const model = document.getElementById('modelSelect').value;
            this.predictForCustomer(customerId, model);
        });

        document.getElementById('predictProductBtn').addEventListener('click', () => {
            const productId = document.getElementById('productSearch').value;
            const model = document.getElementById('productModelSelect').value;
            this.predictForProduct(productId, model);
        });

        document.getElementById('compareBtn').addEventListener('click', () => {
            const customerId = document.getElementById('compareCustomerSearch').value;
            this.compareModels(customerId);
        });

        // Enter key support
        document.getElementById('customerSearch').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') document.getElementById('predictCustomerBtn').click();
        });

        document.getElementById('productSearch').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') document.getElementById('predictProductBtn').click();
        });

        document.getElementById('compareCustomerSearch').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') document.getElementById('compareBtn').click();
        });
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabName);
        });

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.dataset.content === tabName);
        });
    }

    // ===== API Calls =====

    async checkHealth() {
        try {
            const response = await fetch(`${API_BASE}/health`);
            const data = await response.json();

            if (data.models_loaded) {
                this.models.trained = true;
                this.enablePredictionButtons();
            }
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateStatus('Connection Error', 'error');
        }
    }

    async handleFileUpload(file) {
        if (!file) return;

        this.updateStatus('Uploading...', 'loading');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE}/upload_data`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');

            const data = await response.json();
            this.data = data;

            this.updateStatus('Data Loaded', 'success');
            this.displayDataSummary(data);

            // Enable training button
            document.getElementById('trainBtn').disabled = false;

            this.showNotification('Data uploaded successfully!', 'success');
        } catch (error) {
            console.error('Upload error:', error);
            this.updateStatus('Upload Failed', 'error');
            this.showNotification('Failed to upload data', 'error');
        }
    }

    async trainModels() {
        const predictionHorizon = parseInt(document.getElementById('predictionHorizon').value);
        const resumeTraining = document.getElementById('resumeTraining').checked;

        // Show progress
        const progressSection = document.getElementById('trainingProgress');
        progressSection.classList.remove('hidden');

        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');

        // Disable train button
        document.getElementById('trainBtn').disabled = true;

        this.updateStatus('Training Models...', 'loading');

        if (resumeTraining) {
            progressText.textContent = 'Resuming from saved features...';
        } else {
            progressText.textContent = 'Starting fresh training pipeline...';
        }

        // Simulate progress updates
        let progress = 0;
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += Math.random() * 10;
                progressFill.style.width = `${Math.min(progress, 90)}%`;
            }
        }, 500);

        try {
            const response = await fetch(`${API_BASE}/train`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prediction_horizon: predictionHorizon,
                    test_size: 0.2,
                    resume_training: resumeTraining
                })
            });

            clearInterval(progressInterval);

            if (!response.ok) throw new Error('Training failed');

            const data = await response.json();
            this.models.trained = true;
            this.models.metrics = data;

            // Complete progress
            progressFill.style.width = '100%';
            progressText.textContent = 'Training complete!';

            setTimeout(() => {
                progressSection.classList.add('hidden');
            }, 2000);

            // Display metrics
            this.displayMetrics(data);

            // Enable prediction buttons
            this.enablePredictionButtons();

            this.updateStatus('Models Trained', 'success');
            this.showNotification('All models trained successfully!', 'success');

        } catch (error) {
            clearInterval(progressInterval);
            console.error('Training error:', error);
            progressText.textContent = 'Training failed';
            this.updateStatus('Training Failed', 'error');
            this.showNotification('Training failed. Check console for details.', 'error');
            document.getElementById('trainBtn').disabled = false;
        }
    }

    async predictForCustomer(customerId, model = 'ensemble') {
        // 1. Validation First
        if (!customerId) {
            this.showNotification('Please enter a customer ID', 'warning');
            return;
        }

        // 2. Get limit from the specific CUSTOMER input
        const limitInput = document.getElementById('customerTopK');
        const limit = limitInput ? limitInput.value : 20;

        try {
            // 3. Single Fetch Call
            const response = await fetch(
                `${API_BASE}/predict/customer/${customerId}?model=${model}&top_k=${limit}`
            );

            if (!response.ok) throw new Error('Prediction failed');

            const data = await response.json();

            // if (data.predictions && data.predictions.length > 0) {
            //     this.displayCustomerResults(data);
            // } else {
            //     this.showNotification('No predictions available for this customer', 'warning');
            // }

            // We pass the data (even if empty) to the display function
            this.displayCustomerResults(data);

        } catch (error) {
            console.error('Prediction error:', error);
            this.showNotification('Failed to get predictions', 'error');
        }
    }

    async predictForProduct(productId, model = 'ensemble') {
        // 1. Validation First
        if (!productId) {
            this.showNotification('Please enter a product ID', 'warning');
            return;
        }

        // 2. Get limit from the specific PRODUCT input
        const limitInput = document.getElementById('productTopK');
        const limit = limitInput ? limitInput.value : 20;

        try {
            // 3. Single Fetch Call
            const response = await fetch(
                `${API_BASE}/predict/product/${productId}?model=${model}&top_k=${limit}`
            );

            if (!response.ok) throw new Error('Prediction failed');

            const data = await response.json();

            // if (data.predictions && data.predictions.length > 0) {
            //     this.displayProductResults(data);
            // } else {
            //     this.showNotification('No predictions available for this product', 'warning');
            // }

            // We pass the data (even if empty) to the display function
            this.displayProductResults(data);

        } catch (error) {
            console.error('Prediction error:', error);
            this.showNotification('Failed to get predictions', 'error');
        }
    }

    async compareModels(customerId) {
        if (!customerId) {
            this.showNotification('Please enter a customer ID', 'warning');
            return;
        }

        this.updateStatus('Comparing...', 'loading');

        try {
            const response = await fetch(`${API_BASE}/compare_models`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ customer_id: customerId })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Comparison request failed');
            }

            const data = await response.json();

            // Check if data is valid before trying to display
            if (!data.comparisons || data.comparisons.length === 0) {
                throw new Error('Received empty data from server');
            }

            this.displayComparison(data);
            this.updateStatus('System Ready', 'success');

        } catch (error) {
            console.error('Comparison error:', error);
            this.updateStatus('Error', 'error');
            // Show the ACTUAL error message to help debug
            alert(`Error: ${error.message}`);
        }
    }

    // ===== Display Functions =====

    displayDataSummary(data) {
        const summarySection = document.getElementById('dataSummary');
        summarySection.classList.remove('hidden');

        document.getElementById('totalRecords').textContent =
            data.rows.toLocaleString();
        document.getElementById('totalCustomers').textContent =
            data.unique_customers.toLocaleString();
        document.getElementById('totalProducts').textContent =
            data.unique_products.toLocaleString();

        const startDate = new Date(data.date_range.start).toLocaleDateString();
        const endDate = new Date(data.date_range.end).toLocaleDateString();
        document.getElementById('dateRange').textContent =
            `${startDate} - ${endDate}`;
    }

    displayMetrics(data) {
        const metricsSection = document.getElementById('metricsDisplay');
        const metricsGrid = metricsSection.querySelector('.metrics-grid');

        metricsGrid.innerHTML = '';

        // Reorder Likelihood Metrics
        const reorderMetrics = data.reorder_likelihood_metrics;
        for (const [modelName, metrics] of Object.entries(reorderMetrics)) {
            if (typeof metrics === 'object' && metrics.roc_auc) {
                metricsGrid.innerHTML += this.createMetricCard(
                    `${modelName.toUpperCase()} - Reorder`,
                    'ROC AUC',
                    (metrics.roc_auc * 100).toFixed(2) + '%'
                );
            }
        }

        // Quantity Prediction Metrics
        const quantityMetrics = data.quantity_prediction_metrics;
        for (const [modelName, metrics] of Object.entries(quantityMetrics)) {
            if (typeof metrics === 'object' && metrics.mae) {
                metricsGrid.innerHTML += this.createMetricCard(
                    `${modelName.toUpperCase()} - Quantity`,
                    'MAE',
                    metrics.mae.toFixed(2)
                );
            }
        }

        metricsSection.classList.remove('hidden');
    }

    createMetricCard(title, label, value) {
        return `
            <div class="metric-card">
                <div class="metric-name">${title}</div>
                <div class="metric-name">${label}</div>
                <div class="metric-value">${value}</div>
            </div>
        `;
    }


    displayCustomerResults(data) {
        const resultsContainer = document.getElementById('customerResults');
        const resultsGrid = document.getElementById('resultsGrid');
        const resultsCount = document.getElementById('resultsCount');

        resultsContainer.classList.remove('hidden');

        // --- HANDLE EMPTY RESULTS ---
        if (!data.predictions || data.predictions.length === 0) {
            resultsCount.textContent = "0 products";
            resultsGrid.innerHTML = `
                <div class="no-results-card">
                    <div class="no-results-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="12" y1="8" x2="12" y2="12"></line>
                            <line x1="12" y1="16" x2="12.01" y2="16"></line>
                        </svg>
                    </div>
                    <h4>No Predictions Found</h4>
                    <p>This customer may be inactive, filtered out due to returns (quantity < 0), or has insufficient history.</p>
                </div>
            `;
            return;
        }

        resultsCount.textContent = `${data.predictions.length} products`;
        resultsGrid.innerHTML = '';

        data.predictions.forEach((pred, index) => {
            // ... (Rest of your existing card creation code) ...
            const card = document.createElement('div');
            card.className = 'result-card';
            card.style.animationDelay = `${index * 0.05}s`;

            const probabilityPercent = (pred.reorder_probability * 100).toFixed(1);
            const scorePercent = (pred.priority_score * 100).toFixed(0);

            card.innerHTML = `
                <div class="result-info">
                    <div class="result-id">${pred.product_id}</div>
                    <h4>${pred['Product Name'] || 'Unknown Product'}</h4>
                    <div class="result-meta">
                        <span>${pred['Product Manufacturer'] || ''}</span>
                        <span>•</span>
                        <span>Last: ${pred.days_since_last_order || 0} days ago</span>
                    </div>
                </div>
                <div class="result-metrics">
                    <div class="result-metric">
                        <div class="result-metric-label">Probability</div>
                        <div class="result-metric-value">${probabilityPercent}%</div>
                    </div>
                    <div class="result-metric">
                        <div class="result-metric-label">Quantity</div>
                        <div class="result-metric-value">${pred.predicted_quantity}</div>
                    </div>
                </div>
                <div class="result-score">
                    <div class="result-score-label">Priority</div>
                    <div class="result-score-value">${scorePercent}</div>
                </div>
            `;
            resultsGrid.appendChild(card);
        });
    }

    displayProductResults(data) {
        const resultsContainer = document.getElementById('productResults');
        const resultsGrid = document.getElementById('productResultsGrid');
        const resultsCount = document.getElementById('productResultsCount');

        resultsContainer.classList.remove('hidden');

        // --- HANDLE EMPTY RESULTS ---
        if (!data.predictions || data.predictions.length === 0) {
            resultsCount.textContent = "0 customers";
            resultsGrid.innerHTML = `
                <div class="no-results-card">
                    <div class="no-results-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="12" y1="8" x2="12" y2="12"></line>
                            <line x1="12" y1="16" x2="12.01" y2="16"></line>
                        </svg>
                    </div>
                    <h4>No Predictions Found</h4>
                    <p>This product may be discontinued, filtered out due to returns (quantity <= 0), or has no recent sales history.</p>
                </div>
            `;
            return;
        }

        resultsCount.textContent = `${data.predictions.length} customers`;
        resultsGrid.innerHTML = '';

        data.predictions.forEach((pred, index) => {
            // ... (Rest of your existing card creation code) ...
            const card = document.createElement('div');
            card.className = 'result-card';
            card.style.animationDelay = `${index * 0.05}s`;

            const probabilityPercent = (pred.reorder_probability * 100).toFixed(1);
            const scorePercent = (pred.priority_score * 100).toFixed(0);

            card.innerHTML = `
                <div class="result-info">
                    <div class="result-id">${pred.customer_id}</div>
                    <h4>${pred['Partner Customer Name'] || 'Unknown Customer'}</h4>
                    <div class="result-meta">
                        <span>${pred['Partner Customer District'] || 'Unknown District'}</span>
                        <span>•</span>
                        <span>Rep: ${pred['Salesman Name'] || 'N/A'}</span>
                    </div>
                </div>
                <div class="result-metrics">
                    <div class="result-metric">
                        <div class="result-metric-label">Probability</div>
                        <div class="result-metric-value">${probabilityPercent}%</div>
                    </div>
                    <div class="result-metric">
                        <div class="result-metric-label">Quantity</div>
                        <div class="result-metric-value">${pred.predicted_quantity}</div>
                    </div>
                </div>
                <div class="result-score">
                    <div class="result-score-label">Priority</div>
                    <div class="result-score-value">${scorePercent}</div>
                </div>
            `;
            resultsGrid.appendChild(card);
        });
    }

    displayComparison(data) {
        const container = document.getElementById('comparisonResults');

        // Safety check
        if (!data.comparisons || data.comparisons.length === 0) {
            alert("No comparison data available for this customer.");
            return;
        }

        container.classList.remove('hidden');

        const ctx = document.getElementById('comparisonChart').getContext('2d');

        // Prepare data for chart
        const products = data.comparisons.slice(0, 5); // Top 5 products
        const labels = products.map((p, i) => `Product ${p.product_id}`);

        const datasets = [];

        // Safely get model names from the first product
        // Use optional chaining (?.) to prevent crashes
        const firstProd = products[0];
        const models = firstProd.reorder_likelihood ? Object.keys(firstProd.reorder_likelihood) : [];

        const colors = {
            'ensemble': 'rgba(0, 217, 255, 0.8)',
            'lgbm': 'rgba(0, 255, 136, 0.8)',
            'ffnn': 'rgba(255, 184, 0, 0.8)'
        };

        models.forEach(model => {
            datasets.push({
                label: model.toUpperCase(),
                data: products.map(p => {
                    const val = p.reorder_likelihood[model];
                    return (typeof val === 'number') ? (val * 100).toFixed(2) : 0;
                }),
                backgroundColor: colors[model] || 'rgba(255, 255, 255, 0.5)',
                borderColor: colors[model] || 'rgba(255, 255, 255, 0.8)',
                borderWidth: 2
            });
        });

        // Destroy existing chart if any
        if (window.comparisonChart) {
            try {
                window.comparisonChart.destroy();
            } catch (e) {
                console.warn("Could not destroy old chart", e);
            }
        }

        // Create new chart
        window.comparisonChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Model Comparison - Customer ${data.customer_id}`,
                        color: '#e8f1ff'
                    },
                    legend: {
                        labels: { color: '#e8f1ff' }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { color: '#94a8c4' },
                        grid: { color: 'rgba(148, 168, 196, 0.1)' }
                    },
                    x: {
                        ticks: { color: '#94a8c4' },
                        grid: { color: 'rgba(148, 168, 196, 0.1)' }
                    }
                }
            }
        });
    }

    // ===== UI Helpers =====

    updateStatus(text, type = 'ready') {
        const statusElement = document.querySelector('.header-status span');
        const dotElement = document.querySelector('.status-dot');

        statusElement.textContent = text;

        // Update dot color based on type
        const colors = {
            'ready': '#00ff88',
            'loading': '#ffb800',
            'error': '#ff4757',
            'success': '#00ff88'
        };

        dotElement.style.background = colors[type] || colors.ready;
    }

    enablePredictionButtons() {
        document.getElementById('predictCustomerBtn').disabled = false;
        document.getElementById('predictProductBtn').disabled = false;
        document.getElementById('compareBtn').disabled = false;
    }

    showNotification(message, type = 'info') {
        // Simple notification (could be enhanced with a toast library)
        console.log(`[${type.toUpperCase()}] ${message}`);

        // You could add a toast notification library here for better UX
        // For now, we'll use the browser's alert
        if (type === 'error') {
            alert(`Error: ${message}`);
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ReorderApp();
});