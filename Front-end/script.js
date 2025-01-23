// Select DOM elements
const companyDropdown = document.getElementById("company");
const recommendationText = document.getElementById("recommendation");
const ctx = document.getElementById("stockChart").getContext("2d");
const toggleDarkModeButton = document.getElementById("toggleDarkMode"); // Dark mode button
const body = document.body;

// Example stock data (to be replaced with dynamic data)
const stockData = {
    company1: {
        history: Array.from({ length: 90 }, (_, i) => 100 + Math.sin(i / 5) * 10), // Mock last 90 days of data
        forecast: [126, 128, 130, 132, 134, 136, 138],
        recommendation: "Buy"
    },
    company2: {
        history: Array.from({ length: 90 }, (_, i) => 200 + Math.cos(i / 5) * 15),
        forecast: [226, 228, 230, 235, 240, 245, 250],
        recommendation: "Sell"
    },
    company3: {
        history: Array.from({ length: 90 }, (_, i) => 300 + (i % 7) * 5),
        forecast: [341, 343, 345, 350, 355, 360, 365],
        recommendation: "Hold"
    }
};

// Initialize chart
let stockChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: Array.from({ length: 97 }, (_, i) => i < 90 ? `Day ${i - 89}` : `Day ${i - 89}`), // Dynamic labels for 90 days + 7 days
        datasets: []
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                display: true
            }
        }
    }
});

// Update chart with selected company data
function updateChart(company) {
    const data = stockData[company];
    if (data) {
        stockChart.data.datasets = [
            {
                label: "Historical Data",
                data: data.history,
                borderColor: "#007bff",
                fill: false
            },
            {
                label: "Forecast Data",
                data: [
                    ...Array(90).fill(null), // Padding for historical data
                    ...data.forecast
                ],
                borderColor: "#28a745",
                borderDash: [5, 5],
                fill: false
            }
        ];
        stockChart.update();
        recommendationText.textContent = `Recommendation: ${data.recommendation}`;
    } else {
        stockChart.data.datasets = [];
        stockChart.update();
        recommendationText.textContent = "Select a company to see recommendations.";
    }
}

// Event listener for dropdown change
companyDropdown.addEventListener("change", (e) => {
    const selectedCompany = e.target.value;
    updateChart(selectedCompany);
});
// Set default dark mode
document.addEventListener("DOMContentLoaded", () => {
    body.classList.add("dark-mode"); // Add dark-mode class to body by default

    // Update chart colors for dark mode on load
    stockChart.options.plugins.legend.labels.color = "#ffffff";
    stockChart.options.scales = {
        x: {
            ticks: { color: "#ffffff" },
            grid: { color: "#444444" }
        },
        y: {
            ticks: { color: "#ffffff" },
            grid: { color: "#444444" }
        }
    };
    stockChart.update();
});

// Toggle dark mode functionality
toggleDarkModeButton.addEventListener("click", () => {
    body.classList.toggle("dark-mode");

    const isDarkMode = body.classList.contains("dark-mode");

    // Update chart colors based on mode
    stockChart.options.plugins.legend.labels.color = isDarkMode ? "#ffffff" : "#000000";
    stockChart.options.scales = {
        x: {
            ticks: { color: isDarkMode ? "#ffffff" : "#000000" },
            grid: { color: isDarkMode ? "#444444" : "#e5e5e5" }
        },
        y: {
            ticks: { color: isDarkMode ? "#ffffff" : "#000000" },
            grid: { color: isDarkMode ? "#444444" : "#e5e5e5" }
        }
    };
    stockChart.update();
});

