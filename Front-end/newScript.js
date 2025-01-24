// Select DOM elements
const companyDropdown = document.getElementById("company");
const recommendationText = document.getElementById("recommendation");
const ctx = document.getElementById("stockChart").getContext("2d");

// Backend Flask API URL
const API_URL = "http://127.0.0.1:5000/api";

// Initialize chart
let stockChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: []
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                display: true,
                labels: { color: "#000000" }
            }
        },
        scales: {
            x: { ticks: { color: "#000000" }, grid: { color: "#e5e5e5" } },
            y: { ticks: { color: "#000000" }, grid: { color: "#e5e5e5" } }
        }
    }
});

// Fetch and populate the dropdown with companies
function populateDropdown() {
    fetch(`${API_URL}/companies`)
        .then(response => {
            if (!response.ok) {
                throw new Error("Failed to fetch companies.");
            }
            return response.json();
        })
        .then(companies => {
            // Clear existing options
            companyDropdown.innerHTML = '<option value="" disabled selected>Choose a company</option>';
            // Populate dropdown with company options
            companies.forEach(company => {
                const option = document.createElement("option");
                option.value = company.symbol;
                option.textContent = company.name;
                companyDropdown.appendChild(option);
            });
        })
        .catch(error => {
            console.error("Error fetching company data:", error);
            recommendationText.textContent = "Failed to load company data.";
        });
}

// Update chart with selected company data
function updateChart(symbol) {
    fetch(`${API_URL}/company/${symbol}`)
        .then(response => {
            if (!response.ok) {
                throw new Error("Failed to fetch stock data.");
            }
            return response.json();
        })
        .then(data => {
            // Generate labels for historical and predicted data
            const labels = data.historical_prices.dates.concat(
                Array.from({ length: data.predicted_prices.length }, (_, i) => `Future Day ${i + 1}`)
            );

            // Update chart data
            stockChart.data = {
                labels,
                datasets: [
                    {
                        label: "Historical Data",
                        data: data.historical_prices.prices,
                        borderColor: "#007bff",
                        fill: false
                    },
                    {
                        label: "Predicted Data",
                        data: [
                            ...Array(data.historical_prices.prices.length).fill(null), // Padding for historical data
                            ...data.predicted_prices
                        ],
                        borderColor: "#28a745",
                        borderDash: [5, 5],
                        fill: false
                    }
                ]
            };

            // Update recommendation text
            recommendationText.textContent = `Recommendation: ${data.recommendation || "N/A"}`;
            stockChart.update();
        })
        .catch(error => {
            console.error("Error fetching stock data:", error);
            recommendationText.textContent = "Failed to load stock data.";
        });
}

// Event listener for dropdown selection
companyDropdown.addEventListener("change", (e) => {
    const selectedSymbol = e.target.value;
    if (selectedSymbol) {
        updateChart(selectedSymbol);
    }
});

// Populate the dropdown on page load
document.addEventListener("DOMContentLoaded", () => {
    populateDropdown();
});
