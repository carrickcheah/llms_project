# ui.py | dont edit this line

import sys
import os
import pandas as pd
from datetime import datetime
import webbrowser
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QFileDialog, QProgressBar, QComboBox,
                           QCheckBox, QTabWidget, QGridLayout, QGroupBox, QSpinBox,
                           QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                           QSplitter, QFrame, QStatusBar, QDateTimeEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDateTime
from PyQt5.QtGui import QIcon, QFont, QColor, QPalette

# Import scheduler modules
from ingest_data import load_jobs_planning_data
from greedy import greedy_schedule
from chart import create_interactive_gantt
from chart_two import export_schedule_html


class SchedulerWorker(QThread):
    """Worker thread to run scheduling calculations"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, file_path, params):
        super().__init__()
        self.file_path = file_path
        self.params = params
        
    def run(self):
        try:
            # Simulate progress
            self.progress.emit(10)
            
            # Load data
            jobs, machines, setup_times = load_jobs_planning_data(self.file_path)
            self.progress.emit(40)
            
            # Run scheduler
            enforce_sequence = self.params.get('enforce_sequence', True)
            optimize_operators = self.params.get('optimize_operators', False)
            consider_setup_times = self.params.get('consider_setup_times', True)
            
            schedule = greedy_schedule(
                jobs, 
                machines, 
                setup_times,
                enforce_sequence=enforce_sequence,
                optimize_operators=optimize_operators,
                consider_setup_times=consider_setup_times
            )
            self.progress.emit(80)
            
            # Generate report data
            result = {
                'schedule': schedule,
                'jobs': jobs,
                'machines': machines,
                'setup_times': setup_times
            }
            
            self.progress.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class ProductionSchedulerUI(QMainWindow):
    """Main UI for Production Scheduler application"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.schedule_data = None
        self.gantt_file = os.path.join(os.getcwd(), "interactive_schedule.html")
        
    def init_ui(self):
        # Set window properties
        self.setWindowTitle("Production Scheduler")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.setup_tab = QWidget()
        self.report_tab = QWidget()
        self.analysis_tab = QWidget()
        
        self.tabs.addTab(self.setup_tab, "Setup & Run")
        self.tabs.addTab(self.report_tab, "Schedule Results")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        
        # Set up each tab
        self.setup_setup_tab()
        self.setup_report_tab()
        self.setup_analysis_tab()
        
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to schedule production")
        
        # Set initial state
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        
    def setup_setup_tab(self):
        """Setup the first tab with file selection and run options"""
        layout = QVBoxLayout(self.setup_tab)
        
        # File selection section
        file_group = QGroupBox("Data Source")
        file_layout = QGridLayout()
        
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setWordWrap(True)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_file)
        
        file_layout.addWidget(QLabel("Production Data File:"), 0, 0)
        file_layout.addWidget(self.file_path_label, 0, 1)
        file_layout.addWidget(browse_button, 0, 2)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Scheduling options
        options_group = QGroupBox("Scheduling Options")
        options_layout = QGridLayout()
        
        self.enforce_sequence_cb = QCheckBox("Enforce Process Sequence")
        self.enforce_sequence_cb.setChecked(True)
        self.enforce_sequence_cb.setToolTip("Ensure job processes are scheduled in the correct sequence")
        
        self.optimize_operators_cb = QCheckBox("Optimize Operator Assignments")
        self.optimize_operators_cb.setToolTip("Consider operator requirements when scheduling jobs")
        
        self.consider_setup_cb = QCheckBox("Consider Setup Times")
        self.consider_setup_cb.setChecked(True)
        self.consider_setup_cb.setToolTip("Include machine setup times between different job types")
        
        options_layout.addWidget(self.enforce_sequence_cb, 0, 0)
        options_layout.addWidget(self.optimize_operators_cb, 0, 1)
        options_layout.addWidget(self.consider_setup_cb, 1, 0)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Date range selection
        date_group = QGroupBox("Planning Horizon")
        date_layout = QGridLayout()
        
        self.start_date = QDateTimeEdit(QDateTime.currentDateTime())
        self.start_date.setCalendarPopup(True)
        self.start_date.setDisplayFormat("yyyy-MM-dd HH:mm")
        
        end_date = QDateTime.currentDateTime().addDays(14)  # Default to 2 weeks
        self.end_date = QDateTimeEdit(end_date)
        self.end_date.setCalendarPopup(True)
        self.end_date.setDisplayFormat("yyyy-MM-dd HH:mm")
        
        date_layout.addWidget(QLabel("Start Date:"), 0, 0)
        date_layout.addWidget(self.start_date, 0, 1)
        date_layout.addWidget(QLabel("End Date:"), 1, 0)
        date_layout.addWidget(self.end_date, 1, 1)
        
        date_group.setLayout(date_layout)
        layout.addWidget(date_group)
        
        # Run button and progress bar
        run_layout = QHBoxLayout()
        
        self.run_button = QPushButton("Run Scheduler")
        self.run_button.clicked.connect(self.run_scheduler)
        self.run_button.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.progress_bar)
        
        layout.addLayout(run_layout)
        layout.addStretch()
        
    def setup_report_tab(self):
        """Setup the second tab with schedule results"""
        layout = QVBoxLayout(self.report_tab)
        
        # Summary section
        summary_group = QGroupBox("Schedule Summary")
        summary_layout = QGridLayout()
        
        self.total_jobs_label = QLabel("Total Jobs: 0")
        self.total_machines_label = QLabel("Total Machines: 0")
        self.makespan_label = QLabel("Makespan: 0 hours")
        self.utilization_label = QLabel("Average Machine Utilization: 0%")
        
        summary_layout.addWidget(self.total_jobs_label, 0, 0)
        summary_layout.addWidget(self.total_machines_label, 0, 1)
        summary_layout.addWidget(self.makespan_label, 1, 0)
        summary_layout.addWidget(self.utilization_label, 1, 1)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Schedule table
        table_group = QGroupBox("Schedule Details")
        table_layout = QVBoxLayout()
        
        self.schedule_table = QTableWidget()
        self.schedule_table.setColumnCount(6)
        self.schedule_table.setHorizontalHeaderLabels(["Process Code", "Machine", "Start", "End", "Duration (hrs)", "Priority"])
        self.schedule_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        table_layout.addWidget(self.schedule_table)
        
        # Buttons for Gantt chart and export
        buttons_layout = QHBoxLayout()
        
        self.gantt_button = QPushButton("View Gantt Chart")
        self.gantt_button.clicked.connect(self.view_gantt)
        
        self.export_button = QPushButton("Export Schedule")
        self.export_button.clicked.connect(self.export_schedule)
        
        buttons_layout.addWidget(self.gantt_button)
        buttons_layout.addWidget(self.export_button)
        
        table_layout.addLayout(buttons_layout)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
    def setup_analysis_tab(self):
        """Setup the third tab with analysis features"""
        layout = QVBoxLayout(self.analysis_tab)
        
        # Machine utilization section
        utilization_group = QGroupBox("Machine Utilization")
        utilization_layout = QVBoxLayout()
        
        self.machine_combo = QComboBox()
        self.machine_combo.addItem("All Machines")
        
        self.utilization_table = QTableWidget()
        self.utilization_table.setColumnCount(3)
        self.utilization_table.setHorizontalHeaderLabels(["Machine", "Utilization (%)", "Total Hours"])
        self.utilization_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        utilization_layout.addWidget(QLabel("Filter by Machine:"))
        utilization_layout.addWidget(self.machine_combo)
        utilization_layout.addWidget(self.utilization_table)
        
        utilization_group.setLayout(utilization_layout)
        layout.addWidget(utilization_group)
        
        # Due date performance
        due_date_group = QGroupBox("Due Date Performance")
        due_date_layout = QVBoxLayout()
        
        self.due_date_table = QTableWidget()
        self.due_date_table.setColumnCount(4)
        self.due_date_table.setHorizontalHeaderLabels(["Process Code", "Due Date", "Completion", "Buffer (hrs)"])
        self.due_date_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        due_date_layout.addWidget(self.due_date_table)
        
        due_date_group.setLayout(due_date_layout)
        layout.addWidget(due_date_group)
        
    def browse_file(self):
        """Open file dialog to select input data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Production Data File",
            "",
            "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            self.file_path_label.setText(file_path)
            self.run_button.setEnabled(True)
            self.status_bar.showMessage(f"Selected file: {os.path.basename(file_path)}")
    
    def run_scheduler(self):
        """Run the scheduling algorithm"""
        file_path = self.file_path_label.text()
        
        if file_path == "No file selected" or not os.path.exists(file_path):
            QMessageBox.warning(self, "Error", "Please select a valid data file first.")
            return
        
        # Prepare parameters
        params = {
            'enforce_sequence': self.enforce_sequence_cb.isChecked(),
            'optimize_operators': self.optimize_operators_cb.isChecked(),
            'consider_setup_times': self.consider_setup_cb.isChecked(),
            'start_date': self.start_date.dateTime().toSecsSinceEpoch(),
            'end_date': self.end_date.dateTime().toSecsSinceEpoch()
        }
        
        # Disable UI elements during calculation
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Running scheduler...")
        
        # Create and start worker thread
        self.scheduler_worker = SchedulerWorker(file_path, params)
        self.scheduler_worker.progress.connect(self.update_progress)
        self.scheduler_worker.finished.connect(self.scheduler_finished)
        self.scheduler_worker.error.connect(self.scheduler_error)
        self.scheduler_worker.start()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def scheduler_error(self, error_msg):
        """Handle scheduler errors"""
        self.run_button.setEnabled(True)
        self.status_bar.showMessage("Error during scheduling")
        QMessageBox.critical(self, "Scheduling Error", f"An error occurred:\n{error_msg}")
    
    def scheduler_finished(self, result):
        """Process scheduling results"""
        self.schedule_data = result
        self.status_bar.showMessage("Scheduling completed successfully")
        self.run_button.setEnabled(True)
        
        # Enable result tabs
        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(2, True)
        self.tabs.setCurrentIndex(1)  # Switch to results tab
        
        # Update summary information
        schedule = result['schedule']
        jobs = result['jobs']
        
        total_jobs = sum(len(jobs_list) for jobs_list in schedule.values())
        total_machines = len(schedule)
        
        # Calculate makespan and utilization
        makespan = 0
        total_job_time = 0
        
        for machine, jobs_list in schedule.items():
            for job in jobs_list:
                process_code, start, end, priority = job
                job_duration = end - start
                total_job_time += job_duration
                makespan = max(makespan, end)
        
        start_time = min([job[1] for machine_jobs in schedule.values() for job in machine_jobs]) if total_jobs > 0 else 0
        makespan_hours = (makespan - start_time) / 3600
        
        if makespan > 0 and total_machines > 0:
            utilization = (total_job_time / (makespan * total_machines)) * 100
        else:
            utilization = 0
        
        # Update summary labels
        self.total_jobs_label.setText(f"Total Jobs: {total_jobs}")
        self.total_machines_label.setText(f"Total Machines: {total_machines}")
        self.makespan_label.setText(f"Makespan: {makespan_hours:.1f} hours")
        self.utilization_label.setText(f"Average Machine Utilization: {utilization:.1f}%")
        
        # Populate schedule table
        self.populate_schedule_table(schedule)
        
        # Populate analysis tab
        self.populate_analysis_tab(schedule, jobs)
        
        # Generate Gantt chart
        self.generate_gantt_chart(schedule, jobs)
    
    def populate_schedule_table(self, schedule):
        """Populate the schedule results table"""
        # Flatten the schedule for table display
        flat_schedule = []
        for machine, jobs_list in schedule.items():
            for job in jobs_list:
                process_code, start, end, priority = job
                start_dt = datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M')
                end_dt = datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')
                duration = (end - start) / 3600  # hours
                flat_schedule.append((process_code, machine, start_dt, end_dt, duration, priority))
        
        # Sort by start time
        flat_schedule.sort(key=lambda x: x[2])
        
        # Populate table
        self.schedule_table.setRowCount(len(flat_schedule))
        for i, (process, machine, start, end, duration, priority) in enumerate(flat_schedule):
            self.schedule_table.setItem(i, 0, QTableWidgetItem(process))
            self.schedule_table.setItem(i, 1, QTableWidgetItem(machine))
            self.schedule_table.setItem(i, 2, QTableWidgetItem(start))
            self.schedule_table.setItem(i, 3, QTableWidgetItem(end))
            self.schedule_table.setItem(i, 4, QTableWidgetItem(f"{duration:.1f}"))
            
            priority_item = QTableWidgetItem(str(priority))
            
            # Color code by priority
            if priority == 1:
                priority_item.setBackground(QColor(255, 200, 200))  # Light red
            elif priority == 2:
                priority_item.setBackground(QColor(255, 235, 156))  # Light orange
            elif priority == 3:
                priority_item.setBackground(QColor(198, 239, 206))  # Light green
            
            self.schedule_table.setItem(i, 5, priority_item)
    
    def populate_analysis_tab(self, schedule, jobs):
        """Populate the analysis tab with utilization and due date data"""
        # Machine utilization table
        machine_data = {}
        
        for machine, jobs_list in schedule.items():
            total_hours = sum((end - start) / 3600 for _, start, end, _ in jobs_list)
            machine_data[machine] = total_hours
        
        # Update machine combo box
        self.machine_combo.clear()
        self.machine_combo.addItem("All Machines")
        self.machine_combo.addItems(sorted(machine_data.keys()))
        
        # Calculate utilization (approximate)
        makespan_hours = 0
        if schedule:
            earliest_start = float('inf')
            latest_end = 0
            for jobs_list in schedule.values():
                for _, start, end, _ in jobs_list:
                    earliest_start = min(earliest_start, start)
                    latest_end = max(latest_end, end)
            makespan_hours = (latest_end - earliest_start) / 3600 if earliest_start < float('inf') else 0
        
        # Populate utilization table
        self.utilization_table.setRowCount(len(machine_data))
        
        for i, (machine, hours) in enumerate(sorted(machine_data.items())):
            utilization = (hours / makespan_hours * 100) if makespan_hours > 0 else 0
            
            self.utilization_table.setItem(i, 0, QTableWidgetItem(machine))
            self.utilization_table.setItem(i, 1, QTableWidgetItem(f"{utilization:.1f}"))
            self.utilization_table.setItem(i, 2, QTableWidgetItem(f"{hours:.1f}"))
        
        # Due date performance
        due_date_data = []
        
        # Create a lookup for completion times
        completion_times = {}
        for machine, jobs_list in schedule.items():
            for process_code, _, end, _ in jobs_list:
                completion_times[process_code] = end
        
        # Check due dates
        job_lookup = {job['PROCESS_CODE']: job for job in jobs}
        
        for process_code, completion_time in completion_times.items():
            if process_code in job_lookup:
                job = job_lookup[process_code]
                due_date_field = next((f for f in ['LCD_DATE_EPOCH', 'DUE_DATE_TIME'] if f in job), None)
                
                if due_date_field and job[due_date_field]:
                    due_date = job[due_date_field]
                    buffer_hours = (due_date - completion_time) / 3600
                    
                    due_date_str = datetime.fromtimestamp(due_date).strftime('%Y-%m-%d %H:%M')
                    completion_str = datetime.fromtimestamp(completion_time).strftime('%Y-%m-%d %H:%M')
                    
                    due_date_data.append((process_code, due_date_str, completion_str, buffer_hours))
        
        # Sort by buffer (ascending)
        due_date_data.sort(key=lambda x: x[3])
        
        # Populate due date table
        self.due_date_table.setRowCount(len(due_date_data))
        
        for i, (process_code, due_date, completion, buffer) in enumerate(due_date_data):
            self.due_date_table.setItem(i, 0, QTableWidgetItem(process_code))
            self.due_date_table.setItem(i, 1, QTableWidgetItem(due_date))
            self.due_date_table.setItem(i, 2, QTableWidgetItem(completion))
            
            buffer_item = QTableWidgetItem(f"{buffer:.1f}")
            
            # Color code by buffer
            if buffer < 8:
                buffer_item.setBackground(QColor(255, 150, 150))  # Red
            elif buffer < 24:
                buffer_item.setBackground(QColor(255, 200, 100))  # Orange
            elif buffer < 72:
                buffer_item.setBackground(QColor(255, 255, 150))  # Yellow
            else:
                buffer_item.setBackground(QColor(150, 255, 150))  # Green
            
            self.due_date_table.setItem(i, 3, buffer_item)
    
    def generate_gantt_chart(self, schedule, jobs):
        """Generate the interactive Gantt chart"""
        try:
            success = create_interactive_gantt(schedule, jobs, self.gantt_file)
            if success:
                self.status_bar.showMessage(f"Gantt chart generated: {self.gantt_file}")
            else:
                self.status_bar.showMessage("Failed to generate Gantt chart")
        except Exception as e:
            self.status_bar.showMessage(f"Error generating Gantt chart: {str(e)}")
    
    def view_gantt(self):
        """View the generated Gantt chart"""
        if os.path.exists(self.gantt_file):
            webbrowser.open(f"file://{os.path.abspath(self.gantt_file)}")
        else:
            QMessageBox.warning(self, "Error", "Gantt chart file not found.")
    
    def export_schedule(self):
        """Export the schedule to Excel"""
        if not self.schedule_data:
            QMessageBox.warning(self, "Error", "No schedule data available to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Schedule",
            "production_schedule.xlsx",
            "Excel Files (*.xlsx)"
        )
        
        if not file_path:
            return
        
        try:
            # Prepare data for export
            schedule = self.schedule_data['schedule']
            jobs = self.schedule_data['jobs']
            
            # Create flattened schedule dataframe
            schedule_rows = []
            for machine, jobs_list in schedule.items():
                for job in jobs_list:
                    process_code, start, end, priority = job
                    start_dt = datetime.fromtimestamp(start)
                    end_dt = datetime.fromtimestamp(end)
                    duration = (end - start) / 3600  # hours
                    
                    # Get job details if available
                    job_info = next((j for j in jobs if j.get('PROCESS_CODE') == process_code), {})
                    
                    row = {
                        'Process Code': process_code,
                        'Machine': machine,
                        'Start Date': start_dt,
                        'End Date': end_dt,
                        'Duration (hrs)': duration,
                        'Priority': priority
                    }
                    
                    # Add additional job details
                    for key in ['JOB_ID', 'NUMBER_OPERATOR', 'PRODUCTION_RATE']:
                        if key in job_info:
                            row[key] = job_info[key]
                    
                    schedule_rows.append(row)
            
            # Convert to dataframe
            df = pd.DataFrame(schedule_rows)
            
            # Create Excel writer
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Schedule', index=False)
                
                # Add machine utilization sheet
                utilization_data = []
                machine_data = {}
                
                for machine, jobs_list in schedule.items():
                    total_hours = sum((end - start) / 3600 for _, start, end, _ in jobs_list)
                    machine_data[machine] = total_hours
                
                for machine, hours in sorted(machine_data.items()):
                    utilization_data.append({
                        'Machine': machine,
                        'Total Hours': hours
                    })
                
                pd.DataFrame(utilization_data).to_excel(writer, sheet_name='Machine Utilization', index=False)
            
            QMessageBox.information(self, "Export Successful", f"Schedule exported to {file_path}")
            self.status_bar.showMessage(f"Schedule exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting schedule: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look across platforms
    
    window = ProductionSchedulerUI()
    window.show()
    
    sys.exit(app.exec_())