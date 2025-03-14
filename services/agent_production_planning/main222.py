file_path = "/Users/carrickcheah/llms_project/services/agent_production_planning/mydata2.xlsx"
jobs, machines, setup_times = load_jobs_planning_data(file_path)
schedule = schedule_jobs(jobs, machines, setup_times)
plot_gantt_chart(schedule)