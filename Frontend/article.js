window.onload = function () {
  const params = new URLSearchParams(window.location.search);
  const issue = params.get("issue");

  const container = document.getElementById("articleContent");

  // Cpu overheating 
  if (issue === "CPU temperature too high") {
    container.innerHTML = `
      <h2>CPU Temperature Too High</h2>

      <h3>Issue description</h3>
      <ul>
        <li>Your system is overheating</li>
        <li>Fans may be running loudly</li>
        <li>Performance may be throttled</li>
      </ul>

      <h3>Resolution path</h3>

      <h4>Step 1. Close intensive applications</h4>
      <p>High CPU load increases heat output.</p>

      <h4>Step 2. Ensure proper ventilation</h4>
      <p>Place the laptop on a hard surface. Avoid beds or blankets.</p>

      <h4>Step 3. Check for dust buildup</h4>
      <p>Dust in vents can trap heat and block airflow.</p>

      <h4>Step 4. Restart the system</h4>
      <p>Restarting clears runaway processes.</p>

      <h4>Step 5. Update system software</h4>
      <p>Updates often improve thermal management.</p>
    `;
  }

  //Memory high article
  else if (issue === "Memory usage critically high") {
    container.innerHTML = `
      <h2>Memory Usage Critically High</h2>

      <h3>Issue description</h3>
      <ul>
        <li>RAM usage exceeds safe thresholds</li>
        <li>System may feel slow or frozen</li>
        <li>Apps may crash unexpectedly</li>
      </ul>

      <h3>Resolution path</h3>

      <h4>Step 1. Close unused applications</h4>
      <p>Multiple apps consume large amounts of RAM.</p>

      <h4>Step 2. Check Activity Monitor / Task Manager</h4>
      <p>Identify processes using excessive memory.</p>

      <h4>Step 3. Restart the computer</h4>
      <p>Restarting clears memory leaks and cached data.</p>

      <h4>Step 4. Disable startup programs</h4>
      <p>Too many login items increase baseline memory usage.</p>

      <h4>Step 5. Consider adding more RAM</h4>
      <p>If usage is consistently high, hardware upgrade may be needed.</p>
    `;
  }

  //System uptime long
  else if (issue === "System uptime unusually long") {
    container.innerHTML = `
        <h2>System Uptime Too Long</h2>

        <h3>Issue description</h3>
        <ul>
        <li>Your system has been running for an extended period</li>
        <li>Performance may degrade over time</li>
        <li>Temporary glitches or slowdowns may occur</li>
        </ul>

        <h3>Resolution path</h3>

        <h4>Step 1. Save your work</h4>
        <p>Ensure all documents and applications are saved to prevent data loss.</p>

        <h4>Step 2. Restart the system</h4>
        <p>Restarting refreshes memory, clears caches, and restores performance.</p>

        <h4>Step 3. Check for system updates</h4>
        <p>Updates can resolve issues caused by long-running sessions.</p>

        <h4>Step 4. Monitor for recurring issues</h4>
        <p>If the problem persists, note patterns or error messages.</p>
    `;
  }

  //Avg battery health low
  else if (issue === "Average battery health is low") {
  container.innerHTML = `
    <h2>Battery Health is Low</h2>

    <h3>Issue description</h3>
    <ul>
      <li>Your device battery capacity has degraded below recommended levels</li>
      <li>Battery may drain quickly or shut down unexpectedly</li>
      <li>Device may not sustain normal usage unplugged</li>
    </ul>

    <h3>Resolution path</h3>

    <h4>Action Required: Contact IT Support</h4>
    <p>Battery replacement or hardware diagnostics must be performed by authorized IT personnel.</p>

    <h4>What IT Support Will Do</h4>
    <ul>
      <li>Run official battery health diagnostics</li>
      <li>Verify warranty or replacement eligibility</li>
      <li>Schedule battery replacement if necessary</li>
    </ul>

    <h4>Important</h4>
    <p>Do not attempt to replace the battery yourself. Unauthorized repairs may void warranty and cause safety risks.</p>
  `;
}


  // To let us know that an article is not populated yet
  else {
    container.innerHTML = `
      <h2>Article not found</h2>
      <p>No support article is available for this diagnostic.</p>
    `;
  }
};
