// gantt-app.js - JavaScript code to initialize and use the Gantt chart

// This script will be included in the HTML page and will use the GanttChart class
document.addEventListener('DOMContentLoaded', function() {
  // Initialize the chart when the page loads
  initGanttChart();
  
  // Set up event listeners for form submissions
  setupEventListeners();
});

// Function to initialize the Gantt chart with data
async function initGanttChart() {
  try {
    // Fetch schedule data from the server
    const response = await fetch('/get_schedule_data');
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const scheduleData = await response.json();
    
    // Create a new GanttChart instance
    const ganttChart = new GanttChart('gantt-chart-container');
    
    // Process the schedule data
    ganttChart.processScheduleData(scheduleData.schedule, scheduleData.jobs);
    
    // Render the chart
    ganttChart.render();
    
    // Display status message
    const statusElement = document.getElementById('status-message');
    if (statusElement) {
      statusElement.textContent = `Chart updated: ${new Date().toLocaleString()}`;
      statusElement.style.display = 'block';
    }
  } catch (error) {
    console.error('Error initializing Gantt chart:', error);
    
    // Display error message
    const errorElement = document.getElementById('error-message');
    if (errorElement) {
      errorElement.textContent = `Failed to load chart data: ${error.message}`;
      errorElement.style.display = 'block';
    }
  }
}

// Function to set up event listeners
function setupEventListeners() {
  // Listen for data upload form submission
  const uploadForm = document.getElementById('data-upload-form');
  if (uploadForm) {
    uploadForm.addEventListener('submit', async function(event) {
      event.preventDefault();
      
      // Show loading indicator
      const loadingElement = document.getElementById('loading-indicator');
      if (loadingElement) {
        loadingElement.style.display = 'block';
      }
      
      // Clear previous status/error messages
      const statusElement = document.getElementById('status-message');
      const errorElement = document.getElementById('error-message');
      if (statusElement) statusElement.style.display = 'none';
      if (errorElement) errorElement.style.display = 'none';
      
      try {
        // Get form data
        const formData = new FormData(uploadForm);
        
        // Submit the form data
        const response = await fetch('/upload', {
          method: 'POST',
          body: formData
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Hide loading indicator
        if (loadingElement) {
          loadingElement.style.display = 'none';
        }
        
        // Show success message
        if (statusElement) {
          statusElement.textContent = result.message || 'Data uploaded successfully';
          statusElement.style.display = 'block';
        }
        
        // Reload the chart with new data
        initGanttChart();
      } catch (error) {
        console.error('Error uploading data:', error);
        
        // Hide loading indicator
        if (loadingElement) {
          loadingElement.style.display = 'none';
        }
        
        // Show error message
        if (errorElement) {
          errorElement.textContent = `Upload failed: ${error.message}`;
          errorElement.style.display = 'block';
        }
      }
    });
  }
  
  // Listen for scheduling parameter form submission
  const scheduleForm = document.getElementById('schedule-params-form');
  if (scheduleForm) {
    scheduleForm.addEventListener('submit', async function(event) {
      event.preventDefault();
      
      // Show loading indicator
      const loadingElement = document.getElementById('loading-indicator');
      if (loadingElement) {
        loadingElement.style.display = 'block';
      }
      
      // Clear previous status/error messages
      const statusElement = document.getElementById('status-message');
      const errorElement = document.getElementById('error-message');
      if (statusElement) statusElement.style.display = 'none';
      if (errorElement) errorElement.style.display = 'none';
      
      try {
        // Get form data
        const formData = new FormData(scheduleForm);
        
        // Convert form data to JSON
        const formObject = {};
        formData.forEach((value, key) => {
          formObject[key] = value;
        });
        
        // Submit the parameters
        const response = await fetch('/schedule', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(formObject)
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Hide loading indicator
        if (loadingElement) {
          loadingElement.style.display = 'none';
        }
        
        // Show success message
        if (statusElement) {
          statusElement.textContent = result.message || 'Schedule updated successfully';
          statusElement.style.display = 'block';
        }
        
        // Reload the chart with new data
        initGanttChart();
      } catch (error) {
        console.error('Error updating schedule:', error);
        
        // Hide loading indicator
        if (loadingElement) {
          loadingElement.style.display = 'none';
        }
        
        // Show error message
        if (errorElement) {
          errorElement.textContent = `Schedule update failed: ${error.message}`;
          errorElement.style.display = 'block';
        }
      }
    });
  }
}
