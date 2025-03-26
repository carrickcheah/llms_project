// gantt-chart.ts - TypeScript implementation of Gantt chart for production planning

interface JobData {
  id: string;
  task: string;
  start: Date;
  end: Date;
  resource: string;
  priority: string;
  description: string;
  color: string;
  family?: string;
  processNumber?: number;
}

interface ScheduleData {
  [machine: string]: Array<[string, number, number, number]>; // [process_code, start, end, priority]
}

interface JobLookup {
  [processCode: string]: any;
}

class GanttChart {
  private chartElement: HTMLElement;
  private data: JobData[] = [];
  private colors: { [key: string]: string } = {
    'Priority 1 (Highest)': 'rgb(255, 0, 0)',
    'Priority 2 (High)': 'rgb(255, 165, 0)',
    'Priority 3 (Medium)': 'rgb(0, 128, 0)',
    'Priority 4 (Normal)': 'rgb(128, 0, 128)',
    'Priority 5 (Low)': 'rgb(60, 179, 113)'
  };

  constructor(elementId: string) {
    const element = document.getElementById(elementId);
    if (!element) {
      throw new Error(`Element with id ${elementId} not found`);
    }
    this.chartElement = element;
  }

  public processScheduleData(schedule: ScheduleData, jobs?: JobLookup): void {
    this.data = [];
    
    if (!schedule || Object.keys(schedule).length === 0) {
      // Create placeholder task if schedule is empty
      const currentTime = new Date();
      const oneHourLater = new Date(currentTime.getTime() + 3600 * 1000);
      
      this.data.push({
        id: "no-tasks",
        task: "No tasks scheduled",
        start: currentTime,
        end: oneHourLater,
        resource: "None",
        priority: "Priority 3 (Medium)",
        description: "No tasks were scheduled. Please check your input data.",
        color: this.colors['Priority 3 (Medium)']
      });
      
      return;
    }

    // Process and transform the schedule data
    const familyProcesses: { [family: string]: any[] } = {};
    const processDurations: { [processCode: string]: number } = {};
    
    // Extract job family from process code (e.g., "JOB123-P1" → "JOB123")
    const extractJobFamily = (processCode: string): string => {
      const match = processCode.match(/^(.+?)(?:-P\d+|$)/);
      return match ? match[1] : processCode;
    };
    
    // Extract process number from process code (e.g., "JOB123-P1" → 1)
    const extractProcessNumber = (processCode: string): number => {
      const match = processCode.match(/-P(\d+)$/);
      return match ? parseInt(match[1], 10) : 0;
    };
    
    // First pass - collect all process data and organize by family
    for (const [machine, jobsList] of Object.entries(schedule)) {
      for (const jobData of jobsList) {
        try {
          const [processCode, start, end, priority] = jobData;
          
          // Calculate duration
          const duration = end - start;
          processDurations[processCode] = duration;
          
          // Group by family
          const family = extractJobFamily(processCode);
          const seqNum = extractProcessNumber(processCode);
          
          if (!familyProcesses[family]) {
            familyProcesses[family] = [];
          }
          
          // Use original schedule times first
          familyProcesses[family].push([seqNum, processCode, machine, start, end, priority]);
        } catch (e) {
          console.error(`Error processing job data ${jobData}:`, e);
        }
      }
    }
    
    // Sort processes within each family by sequence number
    for (const family in familyProcesses) {
      familyProcesses[family].sort((a, b) => a[0] - b[0]);
    }

    // Convert timestamps to dates and create job data
    for (const [machine, jobsList] of Object.entries(schedule)) {
      for (const [processCode, start, end, priority] of jobsList) {
        const family = extractJobFamily(processCode);
        const processNumber = extractProcessNumber(processCode);
        
        let jobInfo = jobs && jobs[processCode] ? jobs[processCode] : null;
        let description = jobInfo ? `${jobInfo.JOB_ID || ''} - ${jobInfo.DESCRIPTION || ''}` : processCode;

        // Determine priority text
        let priorityText = "Priority 3 (Medium)";
        if (priority === 1) priorityText = "Priority 1 (Highest)";
        else if (priority === 2) priorityText = "Priority 2 (High)";
        else if (priority === 4) priorityText = "Priority 4 (Normal)";
        else if (priority === 5) priorityText = "Priority 5 (Low)";
        
        // Create job data object
        this.data.push({
          id: processCode,
          task: processCode,
          start: new Date(start * 1000),
          end: new Date(end * 1000),
          resource: machine,
          priority: priorityText,
          description,
          color: this.colors[priorityText],
          family,
          processNumber
        });
      }
    }
  }

  public render(): void {
    // Clear previous chart
    this.chartElement.innerHTML = '';

    if (this.data.length === 0) {
      this.chartElement.innerHTML = '<div class="error-message">No data available to display</div>';
      return;
    }

    // Group data by resource (machine)
    const resourceGroups: { [resource: string]: JobData[] } = {};
    
    for (const job of this.data) {
      if (!resourceGroups[job.resource]) {
        resourceGroups[job.resource] = [];
      }
      resourceGroups[job.resource].push(job);
    }

    // Create chart container
    const chartContainer = document.createElement('div');
    chartContainer.className = 'gantt-chart-container';
    
    // Get the earliest and latest dates from all jobs
    const allDates = this.data.flatMap(job => [job.start, job.end]);
    const minDate = new Date(Math.min(...allDates.map(d => d.getTime())));
    const maxDate = new Date(Math.max(...allDates.map(d => d.getTime())));
    
    // Create timeline header
    const timelineHeader = this.createTimelineHeader(minDate, maxDate);
    chartContainer.appendChild(timelineHeader);
    
    // Create each resource row
    for (const [resource, jobs] of Object.entries(resourceGroups)) {
      const rowContainer = document.createElement('div');
      rowContainer.className = 'gantt-row';
      
      // Resource label
      const resourceLabel = document.createElement('div');
      resourceLabel.className = 'resource-label';
      resourceLabel.textContent = resource;
      rowContainer.appendChild(resourceLabel);
      
      // Gantt bars container
      const barsContainer = document.createElement('div');
      barsContainer.className = 'gantt-bars';
      
      // Create bars for each job
      for (const job of jobs) {
        const bar = this.createJobBar(job, minDate, maxDate);
        barsContainer.appendChild(bar);
      }
      
      rowContainer.appendChild(barsContainer);
      chartContainer.appendChild(rowContainer);
    }
    
    this.chartElement.appendChild(chartContainer);
    
    // Add CSS styles for the chart
    this.addStyles();
  }

  private createTimelineHeader(minDate: Date, maxDate: Date): HTMLElement {
    const headerContainer = document.createElement('div');
    headerContainer.className = 'timeline-header';
    
    // Resource column spacer
    const spacer = document.createElement('div');
    spacer.className = 'resource-label';
    headerContainer.appendChild(spacer);
    
    // Timeline labels container
    const timelineLabels = document.createElement('div');
    timelineLabels.className = 'timeline-labels';
    
    // Calculate appropriate time intervals based on chart duration
    const totalDuration = maxDate.getTime() - minDate.getTime();
    const dayInMs = 24 * 60 * 60 * 1000;
    
    if (totalDuration <= 2 * dayInMs) {
      // For short durations, show hourly marks
      this.createHourlyMarkers(timelineLabels, minDate, maxDate);
    } else if (totalDuration <= 14 * dayInMs) {
      // For medium durations, show daily marks
      this.createDailyMarkers(timelineLabels, minDate, maxDate);
    } else {
      // For long durations, show weekly marks
      this.createWeeklyMarkers(timelineLabels, minDate, maxDate);
    }
    
    headerContainer.appendChild(timelineLabels);
    return headerContainer;
  }

  private createHourlyMarkers(container: HTMLElement, minDate: Date, maxDate: Date): void {
    const startHour = new Date(minDate);
    startHour.setMinutes(0, 0, 0);
    
    const endHour = new Date(maxDate);
    endHour.setHours(endHour.getHours() + 1);
    endHour.setMinutes(0, 0, 0);
    
    const totalMs = maxDate.getTime() - minDate.getTime();
    
    for (let time = startHour; time <= endHour; time = new Date(time.getTime() + 60 * 60 * 1000)) {
      const marker = document.createElement('div');
      marker.className = 'time-marker';
      
      const position = ((time.getTime() - minDate.getTime()) / totalMs) * 100;
      marker.style.left = `${position}%`;
      
      const label = document.createElement('span');
      label.textContent = time.getHours() + ':00';
      marker.appendChild(label);
      
      container.appendChild(marker);
    }
  }

  private createDailyMarkers(container: HTMLElement, minDate: Date, maxDate: Date): void {
    const startDay = new Date(minDate);
    startDay.setHours(0, 0, 0, 0);
    
    const totalMs = maxDate.getTime() - minDate.getTime();
    
    for (let day = startDay; day <= maxDate; day = new Date(day.getTime() + 24 * 60 * 60 * 1000)) {
      const marker = document.createElement('div');
      marker.className = 'time-marker';
      
      const position = ((day.getTime() - minDate.getTime()) / totalMs) * 100;
      marker.style.left = `${position}%`;
      
      const label = document.createElement('span');
      label.textContent = day.toLocaleDateString();
      marker.appendChild(label);
      
      container.appendChild(marker);
    }
  }

  private createWeeklyMarkers(container: HTMLElement, minDate: Date, maxDate: Date): void {
    const startDay = new Date(minDate);
    startDay.setHours(0, 0, 0, 0);
    // Go to the previous Monday
    const day = startDay.getDay();
    startDay.setDate(startDay.getDate() - (day === 0 ? 6 : day - 1));
    
    const totalMs = maxDate.getTime() - minDate.getTime();
    
    for (let week = startDay; week <= maxDate; week = new Date(week.getTime() + 7 * 24 * 60 * 60 * 1000)) {
      const marker = document.createElement('div');
      marker.className = 'time-marker';
      
      const position = ((week.getTime() - minDate.getTime()) / totalMs) * 100;
      marker.style.left = `${position}%`;
      
      const label = document.createElement('span');
      label.textContent = `Week of ${week.toLocaleDateString()}`;
      marker.appendChild(label);
      
      container.appendChild(marker);
    }
  }

  private createJobBar(job: JobData, minDate: Date, maxDate: Date): HTMLElement {
    const barContainer = document.createElement('div');
    barContainer.className = 'job-bar-container';
    
    const totalMs = maxDate.getTime() - minDate.getTime();
    const startPosition = ((job.start.getTime() - minDate.getTime()) / totalMs) * 100;
    const duration = (job.end.getTime() - job.start.getTime()) / totalMs * 100;
    
    barContainer.style.left = `${startPosition}%`;
    barContainer.style.width = `${duration}%`;
    
    const bar = document.createElement('div');
    bar.className = 'job-bar';
    bar.style.backgroundColor = job.color;
    bar.setAttribute('data-job-id', job.id);
    bar.setAttribute('title', job.description);
    
    // Add job label
    const label = document.createElement('span');
    label.className = 'job-label';
    label.textContent = job.task;
    
    bar.appendChild(label);
    barContainer.appendChild(bar);
    
    // Add click handler for details
    bar.addEventListener('click', () => {
      this.showJobDetails(job);
    });
    
    return barContainer;
  }

  private showJobDetails(job: JobData): void {
    // Create or update modal for job details
    let modal = document.getElementById('job-details-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'job-details-modal';
      modal.className = 'modal';
      document.body.appendChild(modal);
    }
    
    const startTime = job.start.toLocaleString();
    const endTime = job.end.toLocaleString();
    const duration = (job.end.getTime() - job.start.getTime()) / (60 * 60 * 1000);
    
    modal.innerHTML = `
      <div class="modal-content">
        <span class="close-button">&times;</span>
        <h2>${job.task}</h2>
        <div class="job-info">
          <p><strong>Description:</strong> ${job.description}</p>
          <p><strong>Machine:</strong> ${job.resource}</p>
          <p><strong>Priority:</strong> ${job.priority}</p>
          <p><strong>Start Time:</strong> ${startTime}</p>
          <p><strong>End Time:</strong> ${endTime}</p>
          <p><strong>Duration:</strong> ${duration.toFixed(1)} hours</p>
          ${job.family ? `<p><strong>Job Family:</strong> ${job.family}</p>` : ''}
          ${job.processNumber ? `<p><strong>Process Number:</strong> ${job.processNumber}</p>` : ''}
        </div>
      </div>
    `;
    
    // Show modal
    modal.style.display = 'block';
    
    // Add close handler
    const closeButton = modal.querySelector('.close-button');
    if (closeButton) {
      closeButton.addEventListener('click', () => {
        modal!.style.display = 'none';
      });
    }
    
    // Close when clicking outside
    window.addEventListener('click', (event) => {
      if (event.target === modal) {
        modal!.style.display = 'none';
      }
    });
  }

  private addStyles(): void {
    // Add CSS styles for the chart
    const styleElement = document.createElement('style');
    styleElement.textContent = `
      .gantt-chart-container {
        display: flex;
        flex-direction: column;
        width: 100%;
        overflow-x: auto;
        font-family: Arial, sans-serif;
      }
      
      .timeline-header {
        display: flex;
        border-bottom: 1px solid #ccc;
        position: sticky;
        top: 0;
        background: white;
        z-index: 10;
      }
      
      .timeline-labels {
        position: relative;
        flex: 1;
        height: 40px;
      }
      
      .time-marker {
        position: absolute;
        height: 100%;
        border-left: 1px dashed #ccc;
      }
      
      .time-marker span {
        position: absolute;
        top: 5px;
        left: 5px;
        font-size: 12px;
        white-space: nowrap;
      }
      
      .gantt-row {
        display: flex;
        height: 40px;
        margin-bottom: 10px;
      }
      
      .resource-label {
        width: 120px;
        padding: 5px;
        background-color: #f0f0f0;
        font-weight: bold;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      
      .gantt-bars {
        position: relative;
        flex: 1;
        border-bottom: 1px solid #eee;
      }
      
      .job-bar-container {
        position: absolute;
        top: 5px;
        height: 30px;
      }
      
      .job-bar {
        width: 100%;
        height: 100%;
        border-radius: 4px;
        cursor: pointer;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
      }
      
      .job-bar:hover {
        opacity: 0.8;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
      }
      
      .job-label {
        padding: 0 5px;
        font-size: 12px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: white;
        line-height: 30px;
        text-shadow: 0 1px 1px rgba(0,0,0,0.5);
      }
      
      .modal {
        display: none;
        position: fixed;
        z-index: 1000;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.5);
      }
      
      .modal-content {
        background-color: white;
        margin: 10% auto;
        padding: 20px;
        border-radius: 8px;
        width: 60%;
        max-width: 600px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
      }
      
      .close-button {
        float: right;
        font-size: 28px;
        font-weight: bold;
        cursor: pointer;
      }
      
      .job-info {
        margin-top: 15px;
      }
      
      .error-message {
        padding: 20px;
        text-align: center;
        color: #d32f2f;
        font-style: italic;
      }
    `;
    
    document.head.appendChild(styleElement);
  }
}

// Export the class
export default GanttChart;
