document.addEventListener('DOMContentLoaded', () => {
  const companyListElement = document.getElementById('company-list');
  const companyDropdown = $("#company"); // Select2-compatible selector

  // Fetch data from the backend
  fetch('http://127.0.0.1:5000/api/companies')
    .then(response => response.json())
    .then(data => {
      // Clear existing content
      companyListElement.innerHTML = '';
      companyDropdown.empty().append('<option value="">Select a Company</option>');

      // Populate dropdown and create company cards
      data.forEach(company => {
        // Add companies to dropdown
        companyDropdown.append(`<option value="${company.id}">${company.name} (${company.symbol})</option>`);

        // Create company card
        const companyCard = document.createElement('div');
        companyCard.classList.add('company-card');

        // Calculate change indicators
        const priceChange = company.current_price - company.last_day_price;
        const predictionChange = company.predicted_price - company.current_price;

        companyCard.innerHTML = `
          <h2>${company.name} (${company.symbol})</h2>
          <p>Current Price: ${company.current_price ? `$${company.current_price}` : 'N/A'}
            <span class="${priceChange > 0 ? 'arrow-up' : 'arrow-down'}">
              ${priceChange > 0 ? '↑' : '↓'} ${Math.abs(priceChange).toFixed(2)}
            </span>
          </p>
          <p>Prediction: ${company.predicted_price ? `$${company.predicted_price}` : 'N/A'}
            <span class="${predictionChange > 0 ? 'arrow-up' : 'arrow-down'}">
              ${predictionChange > 0 ? '↑' : '↓'} ${Math.abs(predictionChange).toFixed(2)}
            </span>
          </p>
          <a href="newdetails.html?id=${company.id}" class="view-more-btn">View More</a>
        `;

        companyListElement.appendChild(companyCard);
      });

      // Initialize Select2 for search
      companyDropdown.select2({
        placeholder: "Search for a company...",
        allowClear: true,
        width: "100%",
        dropdownAutoWidth: true
      });

      // Redirect on dropdown selection
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
