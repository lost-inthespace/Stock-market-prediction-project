// Select DOM elements
const urlParams = new URLSearchParams(window.location.search);
const companyId = urlParams.get('id');

const recommendationText = document.getElementById("recommendation");
const ctx = document.getElementById("stockChart").getContext("2d");

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
            },
            tooltip: {
                titleColor: '#ffffff',
                bodyColor: '#ffffff',
                backgroundColor: 'rgb(0, 0, 0)'
            }
        },
        scales: {
            x: {
                ticks: { color: "#000000" },
                grid: { color: "rgba(0,0,0,0.1)" },
                title: { display: true, text: 'Date', color: "#333333" }
            },
            y: {
                ticks: { color: "#000000" },
                grid: { color: "rgba(0,0,0,0.1)" },
                title: { display: true, text: 'Price', color: "#333333" }
            }
        }
    }
});

// Fetch company data from backend API and populate the dropdown
function fetchCompanies() {
    fetch("http://127.0.0.1:5000/api/companies")
        .then(response => response.json())
        .then(companies => {
            const companyDropdown = $("#company");
            companyDropdown.empty(); // Clear existing options
            companyDropdown.append('<option value="">Select a Company</option>');

            companies.forEach(company => {
                companyDropdown.append(`<option value="${company.id}">${company.name} (${company.symbol})</option>`);
            });

            // Initialize Select2 (ensuring it runs after the dropdown is populated)
            companyDropdown.select2({
                placeholder: "Search for a company...",
                allowClear: true,
                width: "100%",
                dropdownAutoWidth: true
            });

            // Set dropdown value to companyId from URL if it exists
            if (companyId) {
                companyDropdown.val(companyId).trigger("change"); // Auto-select and trigger change
            }
        })
        .catch(error => {
            console.error("Error fetching companies:", error);
        });

        updateChart(companyId)
}

// Update chart with selected company data
function updateChart(companyId) {
    if (!companyId) {
        recommendationText.textContent = "Select a company to see recommendations.";
        stockChart.data.datasets = [];
        stockChart.data.labels = [];
        stockChart.update();
        return;
    }

    fetch(`http://127.0.0.1:5000/api/company/${companyId}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById("companyName").textContent = data.name;
            document.getElementById("companySymbol").textContent = data.symbol;
            document.getElementById("MarketCap").textContent = data.market_cap;
            document.getElementById("companyVolume").textContent = data.volume;
            document.getElementById("companysector").textContent = data.sector;
            document.getElementById("recommendations").textContent = data.recommendation;

            const labels = [...data.historical.dates, ...data.forecast.dates];

            stockChart.data.labels = labels;
            stockChart.data.datasets = [
                {
                    label: "Historical Data",
                    data: [
                        ...data.historical.prices,
                        ...Array(data.forecast.prices.length).fill(null)
                    ],
                    borderColor: "#007bff",
                    fill: false
                },
                {
                    label: "Forecast Data",
                    data: [
                        ...Array(data.historical.prices.length - 1).fill(null),
                        ...data.forecast.prices
                    ],
                    borderColor: "#F72C5B",
                    fill: false
                }
            ];
            stockChart.update();

            recommendationText.textContent = `Recommendation: ${data.recommendation}`;
        })
        .catch(error => {
            console.error("Error fetching company data:", error);
            recommendationText.textContent = "Failed to load company data.";
        });
}

// Event listener for Select2 dropdown change
$(document).on("select2:select", "#company", function(e) {
    const selectedCompanyId = e.params.data.id;
    updateChart(selectedCompanyId);
});

// Initialize the page
$(document).ready(() => {
    fetchCompanies();
});
