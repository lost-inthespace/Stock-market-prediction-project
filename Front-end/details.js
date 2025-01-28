document.addEventListener("DOMContentLoaded", () => {
  const urlParams = new URLSearchParams(window.location.search);
  const companyId = urlParams.get("id");
  const companyDetailsElement = document.getElementById("company-details");

  fetch(`http://127.0.0.1:5000/api/company/${companyId}`)
      .then((response) => response.json())
      .then((company) => {
          companyDetailsElement.innerHTML = `
              <h2>${company.name} (${company.symbol})</h2>
              <p>Current Price: ${company.current_price} SAR</p>
              <p>Prediction (Next Day): ${company.predicted_price} SAR</p>
              <p>Market Cap: ${company.market_cap} SAR</p>
              <p>Volume: ${company.volume} SAR</p>
              <canvas id="price-chart" width="600" height="400"></canvas>
          `;

          // Append the single predicted price to the historical prices
          const extendedPrices = [...company.historical_prices.prices, company.predicted_price];

          // Extend the labels to include one future day
          const extendedDates = [
              ...company.historical_prices.dates,
              "Next Day"
          ];

          // Draw the price chart
          const ctx = document.getElementById("price-chart").getContext("2d");
          new Chart(ctx, {
              type: "line",
              data: {
                  labels: extendedDates,
                  datasets: [
                      {
                          label: "Prices (Historical + Predicted)",
                          data: extendedPrices,
                          borderColor: "blue",
                          fill: false
                      }
                  ]
              },
              options: {
                  responsive: true,
                  plugins: {
                      legend: {
                          display: true
                      }
                  },
                  scales: {
                      x: { 
                          ticks: { color: "#000000" }, 
                          grid: { color: "#e5e5e5" } 
                      },
                      y: { 
                          ticks: { color: "#000000" }, 
                          grid: { color: "#e5e5e5" } 
                      }
                  }
              }
          });
      })
      .catch((error) => {
          console.error("Error fetching company details:", error);
          companyDetailsElement.innerHTML = `<p>Failed to load company details.</p>`;
      });
});
