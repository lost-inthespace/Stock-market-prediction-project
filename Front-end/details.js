document.addEventListener('DOMContentLoaded', () => {
    const urlParams = new URLSearchParams(window.location.search);
    const companyId = urlParams.get('id');
    const companyDetailsElement = document.getElementById('company-details');
  
    fetch(`http://127.0.0.1:5000/api/company/${companyId}`)
      .then(response => response.json())
      .then(company => {
        companyDetailsElement.innerHTML = `
          <h2>${company.name} (${company.symbol})</h2>
          <p>Current Price: ${company.current_price} SAR</p>
          <p>Prediction: ${company.predicted_price} SAR</p>
          <p>Market Cap: ${company.market_cap} SAR</p>
          <p>Volume: ${company.volume} SAR</p>
          <canvas id="price-chart" width="600" height="400"></canvas>
        `;
  
        // Draw the price chart
        const ctx = document.getElementById('price-chart').getContext('2d');
        new Chart(ctx, {
          type: 'line',
          data: {
            labels: company.historical_prices.dates,
            datasets: [
              {
                label: 'Actual Prices',
                data: company.historical_prices.prices,
                borderColor: 'blue',
                fill: false
              },
              {
                label: 'Predicted Prices',
                data: company.predicted_prices,
                borderColor: 'red',
                borderDash: [5, 5],
                fill: false
              }
            ]
          }
        });
      })
      .catch(error => {
        console.error('Error fetching company details:', error);
        companyDetailsElement.innerHTML = `<p>Failed to load company details.</p>`;
      });
  });
