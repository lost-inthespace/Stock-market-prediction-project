document.addEventListener('DOMContentLoaded', () => {
  const companyListElement = document.getElementById('company-list');
  const companyDropdown = $("#company");
  const topCompaniesContainer = document.getElementById('top-companies');

  fetch('http://127.0.0.1:5000/api/companies')
    .then(response => response.json())
    .then(data => {
      companyListElement.innerHTML = '';
      companyDropdown.empty().append('<option value="">Select a Company</option>');

      // First pass: Process all companies and calculate metrics
      data.forEach(company => {
        // Calculate metrics
        company.priceChange = company.current_price - company.last_day_price;
        company.priceChangePercent = (company.priceChange / company.last_day_price * 100);
        company.predictionChange = company.predicted_price - company.current_price;
        company.predictionChangePercent = (company.predictionChange / company.current_price * 100);
      });

      // Create main company list
      data.forEach(company => {
        // Add to dropdown
        companyDropdown.append(`<option value="${company.id}">${company.name} (${company.symbol})</option>`);

        // Create company card
        const companyCard = document.createElement('div');
        companyCard.classList.add('company-card');
        companyCard.innerHTML = `
          <h2>${company.name} (${company.symbol})</h2>
          <strong><p>Current Price: ${company.current_price ? `$${company.current_price.toFixed(2)}` : 'N/A'}
            <span class="${company.priceChange > 0 ? 'arrow-up' : 'arrow-down'}">
              ${company.priceChange > 0 ? '↑' : '↓'} $${Math.abs(company.priceChange).toFixed(2)} (${Math.abs(company.priceChangePercent).toFixed(2)}%)
            </span>
          </p></strong>
          <strong><p>Prediction: ${company.predicted_price ? `$${company.predicted_price.toFixed(2)}` : 'N/A'}
            <span class="${company.predictionChange > 0 ? 'arrow-up' : 'arrow-down'}">
              ${company.predictionChange > 0 ? '↑' : '↓'} $${Math.abs(company.predictionChange).toFixed(2)} (${Math.abs(company.predictionChangePercent).toFixed(2)}%)
            </span>
          </p></strong>
          <a href="newdetails.html?id=${company.id}" class="view-more-btn">View More</a>
        `;
        companyListElement.appendChild(companyCard);
      });

      // Create Top 5 section
      const sortedByPrediction = [...data].sort((a, b) => b.predictionChangePercent - a.predictionChangePercent);
      const top5 = sortedByPrediction.slice(0, 5);

      // In the top5.forEach loop:
      top5.forEach(company => {
        const topCard = document.createElement('div');
        topCard.classList.add('company-card', 'top-company');
        topCard.innerHTML = `
            <h3>${company.name} (${company.symbol})</h3>
            <p class="prediction-surge">
                <span class="${company.predictionChange > 0 ? 'arrow-up' : 'arrow-down'}">
                    ${company.predictionChange > 0 ? '↑' : '↓'} 
                    $${Math.abs(company.predictionChange).toFixed(2)} 
                    (${Math.abs(company.predictionChangePercent).toFixed(2)}%)
                </span>
            </p>
            <p>Current: $${company.current_price.toFixed(2)}</p>
            <p>Predicted: $${company.predicted_price.toFixed(2)}</p>
            <a href="newdetails.html?id=${company.id}" class="view-more-btn">View More</a>
        `;
        topCompaniesContainer.appendChild(topCard);
      });

      // Initialize Select2
      companyDropdown.select2({
        placeholder: "Search for a company...",
        allowClear: true,
        width: "100%",
        dropdownAutoWidth: true
      });

      companyDropdown.on("select2:select", function(e) {
          const selectedCompanyId = e.params.data.id;
          if (selectedCompanyId) {
              window.location.href = `newdetails.html?id=${selectedCompanyId}`;
          }
      });
    })
    .catch(error => {
      console.error('Error fetching companies:', error);
      companyListElement.innerHTML = `<p>Failed to load company data.</p>`;
    });
});