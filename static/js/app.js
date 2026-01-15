// Weather Icons Map
const weatherIcons = {
    sun: '☀️',
    cloud: '☁️',
    rain: '🌧️',
    storm: '⛈️'
};

// DOM Elements
const generateBtn = document.getElementById('generateBtn');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const forecastContainer = document.getElementById('forecastContainer');
const daysSelector = document.getElementById('daysSelector');
const forecastTitle = document.getElementById('forecastTitle');
const alertsContainer = document.getElementById('alertsContainer');

// Update title when days change
daysSelector.addEventListener('change', () => {
    const days = daysSelector.value;
    forecastTitle.textContent = `${days}-Day Forecast`;
});

// Load alerts on page load
window.addEventListener('load', loadAlerts);

// Generate Forecast
generateBtn.addEventListener('click', async () => {
    const selectedDays = parseInt(daysSelector.value);

    generateBtn.disabled = true;
    generateBtn.querySelector('.btn-text').textContent = 'Generating...';

    // Show progress
    progressSection.style.display = 'block';
    progressFill.style.width = '0%';

    // Clear previous forecasts
    forecastContainer.innerHTML = '<div class="empty-state"><p>Loading forecast data...</p></div>';

    try {
        const response = await fetch(`/generate_forecast/?days=${selectedDays}`);
        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Animate progress
        let progress = 0;
        const progressStep = 100 / selectedDays;
        const progressInterval = setInterval(() => {
            progress += progressStep;
            progressFill.style.width = `${Math.min(progress, 100)}%`;
            if (progress >= 100) clearInterval(progressInterval);
        }, 150);

        // Clear container
        forecastContainer.innerHTML = '';

        // Add forecast cards with stagger
        data.forecasts.forEach((forecast, index) => {
            setTimeout(() => {
                const card = createForecastCard(forecast);
                forecastContainer.appendChild(card);
            }, index * 100);
        });

        setTimeout(() => {
            progressSection.style.display = 'none';
            progressFill.style.width = '0%';
        }, 2000);

    } catch (error) {
        console.error('Error generating forecast:', error);
        forecastContainer.innerHTML = `
            <div class="empty-state">
                <p style="color: #EF4444;">Error: ${error.message}</p>
            </div>
        `;
        progressSection.style.display = 'none';
    } finally {
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').textContent = 'Generate Forecast';
    }
});

// Create Forecast Card
function createForecastCard(forecast) {
    const card = document.createElement('div');
    card.className = 'forecast-card glass';

    card.innerHTML = `
        <div class="forecast-date">
            <div class="forecast-day">${forecast.day}</div>
            <div class="forecast-full-date">${forecast.full_date}</div>
        </div>
        
        <div class="forecast-icon">
            ${weatherIcons[forecast.icon]}
        </div>
        
        <div class="forecast-spacer"></div>
        
        <div class="forecast-precip">
            <div class="precip-icon">💧</div>
            <div class="precip-prob">${forecast.prob}%</div>
        </div>
        
        <div class="forecast-temp">
            ${forecast.temp.toFixed(1)}°
        </div>
    `;

    return card;
}

// Load IMD Weather Alerts
async function loadAlerts() {
    try {
        const response = await fetch('/get_alerts/');
        const data = await response.json();

        if (data.alerts && data.alerts.length > 0) {
            alertsContainer.innerHTML = '';
            data.alerts.forEach((alert, index) => {
                setTimeout(() => {
                    const alertCard = createAlertCard(alert);
                    alertsContainer.appendChild(alertCard);
                }, index * 100);
            });
        } else {
            alertsContainer.innerHTML = '<div class="no-alerts"><p>No active weather alerts for Chennai</p></div>';
        }
    } catch (error) {
        console.error('Error loading alerts:', error);
        alertsContainer.innerHTML = '<div class="no-alerts"><p>No active weather alerts for Chennai</p></div>';
    }
}

// Create Alert Card
function createAlertCard(alert) {
    const card = document.createElement('div');
    card.className = `alert-card glass severity-${alert.severity}`;

    const severityIcons = {
        red: '🚨',
        orange: '⚠️',
        yellow: '⚡',
        green: 'ℹ️'
    };

    card.innerHTML = `
        <div class="alert-icon">${severityIcons[alert.severity]}</div>
        <div class="alert-content">
            <div class="alert-title">${alert.title}</div>
            <div class="alert-description">${alert.description}</div>
            <div class="alert-meta">
                <span class="alert-badge severity-${alert.severity}">${alert.severity.toUpperCase()}</span>
                <span>Valid: ${alert.valid_until}</span>
            </div>
        </div>
    `;

    return card;
}