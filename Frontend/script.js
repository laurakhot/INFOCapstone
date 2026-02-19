window.onload = function () {

  // Check if diagnostics already exist
  const saved = sessionStorage.getItem("diagnostics");

  if (saved) {
    displayReport(JSON.parse(saved)); // Some vibe code for if we have a JSON output
  } else {
    const results = generateReport();
    sessionStorage.setItem("diagnostics", JSON.stringify(results));
    displayReport(results);
  }
};

//If no diagnostics, I threw a few basic basic ones in 
function generateReport() {
  const issues = [
    "CPU temperature too high",
    "System uptime unusually long",
    "Memory usage critically high",
    "Average battery health is low"
  ];

  //Shuffled for no repeats, but we will just directly pull these moving forward
  //All filler until we get stuff from model :p
  const shuffledIssues = [...issues].sort(() => Math.random() - 0.5); 
  let results = [];

  for (let i = 0; i < 4; i++) {
    results.push({
      diagnosis: shuffledIssues[i],
      confidence: Math.floor(Math.random() * 80) + 20
    });
  }

  //Highest percentage first - keep for diagnostics!! 
  results.sort((a, b) => b.confidence - a.confidence);

  return results;
}

// Displaying diagnostics 
function displayReport(results) {

  const main = results[0];

  // Main diagnostic asf
  document.getElementById("mainDiagnostic").innerHTML = `
    <div class="section-title">Main Diagnostic:</div>
    <div class="diagnostic">
      1. Diagnostic: ${main.diagnosis}<br>
      Confidence Score: ${main.confidence}%<br>
      <a class="support" href="article.html?issue=${encodeURIComponent(main.diagnosis)}">
        ↳ View support article
      </a>
    </div>
  `;

  // Additional diagnostics
  let additionalHTML = `<div class="section-title">Additional Diagnostics:</div>`;

  results.slice(1).forEach((item, index) => {
    additionalHTML += `
      <div class="diagnostic">
        ${index + 2}. Diagnostic: ${item.diagnosis}<br>
        Confidence Score: ${item.confidence}%<br>
        <a class="support" href="article.html?issue=${encodeURIComponent(item.diagnosis)}">
          ↳ View support article
        </a>
      </div>
    `;
  });

  document.getElementById("additionalDiagnostics").innerHTML = additionalHTML;
}



