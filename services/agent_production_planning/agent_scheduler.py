# agent_scheduler.py

import os
import json
import time
import copy
from typing import Dict, List, Any, Tuple, Optional, Set
from loguru import logger
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Import model initializer from model.py
from model import initialize_open_deep

# OR-Tools scheduler and greedy scheduler (to be imported)
try:
    import sch_jobs
    import greedy
    HAS_SCHEDULERS = True
except ImportError:
    logger.warning("sch_jobs.py or greedy.py not found. Will use internal scheduling only.")
    HAS_SCHEDULERS = False

# Load environment variables
load_dotenv()

class AgentScheduler:
    def __init__(self, jobs_data, machines, setup_times, max_operators=None, max_iterations=5):
        self.jobs_data = jobs_data
        self.machines = machines
        self.setup_times = setup_times
        self.max_operators = max_operators  # New constraint
        self.max_iterations = max_iterations  # Maximum optimization iterations
        self.bottleneck_machines = self._identify_bottleneck_machines()
        
        # Initialize OpenAI LLM
        try:
            self.llm = initialize_open_deep(os.getenv("DEEPSEEK_API_KEY"))
            self.use_llm = True
            logger.info("OpenAI LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}. Using rule-based scheduling instead.")
            self.use_llm = False
        
    def _identify_bottleneck_machines(self):
        """Identify machines with highest total processing time."""
        machine_load = {}
        for job in self.jobs_data:
            machine = job.get('RSC_CODE', job.get('MACHINE_ID', 'Unknown'))
            if machine not in machine_load:
                machine_load[machine] = 0
            # Use HOURS_NEED if available, otherwise HOURS_NEEDED
            hours = job.get('HOURS_NEED', job.get('HOURS_NEEDED', 0))
            machine_load[machine] += hours
            
        # Sort machines by load (descending)
        sorted_machines = sorted(machine_load.items(), key=lambda x: x[1], reverse=True)
        # Return top 20% as bottleneck machines
        return [m[0] for m in sorted_machines[:int(len(sorted_machines) * 0.2)]]
    
    def schedule(self):
        """Main scheduling method with iterative optimization."""
        if not HAS_SCHEDULERS:
            logger.warning("OR-Tools scheduler not available. Using internal scheduling only.")
            return self._internal_scheduling()
        
        # Iterative optimization process
        return self._iterative_optimization()
    
    def _iterative_optimization(self):
        """Implement iterative optimization with OR-Tools and AI agent."""
        logger.info("Starting iterative optimization process")
        
        # Initialize optimization variables
        current_iterations = 0
        feasible_solution = False
        best_schedule = None
        jobs = copy.deepcopy(self.jobs_data)  # Work with a copy to preserve originals
        
        # Pre-process jobs to ensure START_DATE constraints are properly set
        for job in jobs:
            # Ensure START_DATE is properly set if it exists
            if 'START_DATE' in job and job['START_DATE']:
                logger.info(f"Job {job.get('UNIQUE_JOB_ID', job.get('JOB', 'Unknown'))} has START_DATE constraint: {job['START_DATE']}")
        
        # Keep track of jobs modified by the AI agent
        modified_job_ids = set()
        
        while not feasible_solution and current_iterations < self.max_iterations:
            logger.info(f"Optimization iteration {current_iterations + 1}/{self.max_iterations}")
            
            # Try OR-Tools scheduling
            try:
                # Use the correct function from sch_jobs.py (schedule_jobs, not create_schedule)
                schedule = sch_jobs.schedule_jobs(jobs, self.machines, self.setup_times)
                logger.info(f"OR-Tools scheduling complete for iteration {current_iterations + 1}")
                
                # Store the best schedule we have so far
                best_schedule = schedule
                
                # Check if we have infeasible/late jobs
                infeasible_jobs = self._get_infeasible_jobs(schedule)
                
                if not infeasible_jobs:
                    logger.info("Feasible solution found! No infeasible jobs.")
                    feasible_solution = True
                else:
                    logger.info(f"Found {len(infeasible_jobs)} infeasible jobs. Using AI agent to resolve.")
                    
                    # Use AI agent to modify constraints or priorities based on infeasible jobs
                    constraint_changes = self._resolve_constraints(infeasible_jobs, modified_job_ids)
                    
                    # Update job constraints in our working copy
                    self._update_job_constraints(jobs, constraint_changes)
                    
                    # Track which jobs were modified
                    for job_id in constraint_changes.keys():
                        modified_job_ids.add(job_id)
                    
                    logger.info(f"Updated constraints for {len(constraint_changes)} jobs")
                    
            except Exception as e:
                logger.error(f"Error in OR-Tools scheduling: {e}")
                # If OR-Tools fails, try our internal scheduling
                best_schedule = self._internal_scheduling()
                break
                
            current_iterations += 1
            
        if not feasible_solution:
            logger.warning("Could not find feasible solution with OR-Tools + AI. Trying greedy approach.")
            
            try:
                # Fall back to greedy approach for remaining infeasible jobs
                remaining_infeasible = self._get_infeasible_jobs(best_schedule) if best_schedule else jobs
                
                if remaining_infeasible:
                    # The correct function in greedy.py is greedy_schedule, not schedule_remaining_jobs
                    greedy_schedule = greedy.greedy_schedule(remaining_infeasible, self.machines, self.setup_times)
                    
                    # Combine the best OR-Tools schedule with the greedy schedule
                    if best_schedule:
                        final_schedule = self._merge_schedules(best_schedule, greedy_schedule)
                    else:
                        final_schedule = greedy_schedule
                else:
                    final_schedule = best_schedule
            except Exception as e:
                logger.error(f"Error in greedy scheduling: {e}")
                # If everything else fails, use our internal scheduling
                final_schedule = self._internal_scheduling()
        else:
            final_schedule = best_schedule
            
        logger.info(f"Scheduling complete after {current_iterations} iterations")
        return final_schedule
    
    def _internal_scheduling(self):
        """Internal scheduling method when external schedulers are not available."""
        logger.info("Using internal scheduling logic")
        
        # Check if LLM is available
        if self.use_llm:
            return self._llm_guided_scheduling()
        else:
            # Fall back to rule-based scheduling
            # 1. Prioritize jobs on bottleneck machines
            bottleneck_jobs = self._prioritize_bottleneck_jobs()
            
            # 2. Schedule remaining jobs considering operator constraints
            remaining_jobs = self._schedule_with_operator_constraints()
            
            # 3. Combine schedules
            full_schedule = self._combine_schedules(bottleneck_jobs, remaining_jobs)
            
            return full_schedule
            
    def _llm_guided_scheduling(self):
        """Use LLM to guide the scheduling process."""
        # 1. Analyze bottlenecks using LLM
        bottleneck_analysis = self._llm_analyze_bottlenecks()
        
        # 2. Get job priorities using LLM reasoning
        prioritized_jobs = self._llm_prioritize_jobs(bottleneck_analysis)
        
        # 3. Schedule jobs based on priorities and operator constraints
        schedule = self._create_schedule_from_priorities(prioritized_jobs)
        
        return schedule
    
    def _prioritize_bottleneck_jobs(self):
        """Schedule jobs on bottleneck machines first."""
        # Group jobs by machine
        machine_jobs = {}
        for job in self.jobs_data:
            machine = job.get('RSC_CODE', job.get('MACHINE_ID', 'Unknown'))
            if machine not in machine_jobs:
                machine_jobs[machine] = []
            machine_jobs[machine].append(job)
        
        # For bottleneck machines, sort jobs by due date and priority
        bottleneck_schedule = {}
        for machine in self.bottleneck_machines:
            if machine in machine_jobs:
                # Multi-level sorting: priority first (higher is more important),
                # then by due date (earlier is more important),
                # then by processing time (shorter first) as a tiebreaker
                sorted_jobs = sorted(machine_jobs[machine], 
                                     key=lambda x: (
                                         -int(x.get('PRIORITY', 0)),  # Higher priority first
                                         x.get('LCD_DATE', float('inf')),  # Earlier due date first
                                         float(x.get('HOURS_NEED', x.get('HOURS_NEEDED', 0)))  # Shorter jobs first
                                     ))
                bottleneck_schedule[machine] = sorted_jobs
        
        return bottleneck_schedule
    
    def _schedule_with_operator_constraints(self):
        """Schedule remaining jobs with operator constraints."""
        # Get all jobs for non-bottleneck machines
        non_bottleneck_jobs = []
        for job in self.jobs_data:
            machine = job.get('RSC_CODE', job.get('MACHINE_ID', 'Unknown'))
            if machine not in self.bottleneck_machines:
                non_bottleneck_jobs.append(job)
        
        # Log some stats about the non-bottleneck jobs
        logger.info(f"Scheduling {len(non_bottleneck_jobs)} jobs on {len(set(job.get('RSC_CODE', job.get('MACHINE_ID', 'Unknown')) for job in non_bottleneck_jobs))} non-bottleneck machines")
        
        # Sort by priority (higher first), due date, and processing time
        sorted_jobs = sorted(non_bottleneck_jobs, 
                             key=lambda x: (
                                 -int(x.get('PRIORITY', 0)),  # Higher priority first
                                 x.get('LCD_DATE', float('inf')),  # Earlier due date first
                                 float(x.get('HOURS_NEED', x.get('HOURS_NEEDED', 0)))  # Shorter jobs first
                             ))
        
        # Schedule jobs considering operator constraints
        schedule = {}
        current_time = {}
        operator_count = 0
        
        for job in sorted_jobs:
            # Skip if max operators reached
            if self.max_operators and operator_count >= self.max_operators:
                continue
                
            machine = job.get('RSC_CODE', job.get('MACHINE_ID', 'Unknown'))
            processing_time = job.get('HOURS_NEED', job.get('HOURS_NEEDED', 0))
            
            # Initialize machine time if not already done
            if machine not in current_time:
                current_time[machine] = 0
                schedule[machine] = []
            
            # Calculate job start and end times
            start_time = current_time[machine]
            end_time = start_time + processing_time
            
            # Add job to schedule
            schedule[machine].append({
                **job,
                'START_TIME': start_time,
                'END_TIME': end_time
            })
            
            # Update machine time
            current_time[machine] = end_time
            
            # Update operator count
            operator_count += job.get('NUMBER_OPERATOR', 1)
        
        return schedule
    
    def _combine_schedules(self, bottleneck_schedule, remaining_schedule):
        """Combine and finalize schedules."""
        combined_schedule = {}
        
        # Add bottleneck jobs first
        for machine, jobs in bottleneck_schedule.items():
            combined_schedule[machine] = []
            current_time = 0
            
            for job in jobs:
                # Respect START_DATE constraint if present
                if 'START_DATE' in job and job['START_DATE']:
                    start_time = max(current_time, job['START_DATE'])
                else:
                    start_time = current_time
                    
                processing_time = job.get('HOURS_NEED', job.get('HOURS_NEEDED', 0))
                end_time = start_time + processing_time
                
                # Check due date and log warning if job will be late
                if 'LCD_DATE' in job and job['LCD_DATE'] and end_time > job['LCD_DATE']:
                    lateness = end_time - job['LCD_DATE']
                    job_id = job.get('UNIQUE_JOB_ID', job.get('JOB', 'Unknown'))
                    logger.warning(f"Job {job_id} on machine {machine} will be late by {lateness:.1f} hours")
                    # Store lateness value for later reporting
                    job['BAL_HR'] = -lateness
                
                # Add job to schedule
                combined_schedule[machine].append({
                    **job,
                    'START_TIME': start_time,
                    'END_TIME': end_time
                })
                
                # Update current time
                current_time = end_time
        
        # Add remaining jobs
        for machine, jobs in remaining_schedule.items():
            if machine not in combined_schedule:
                combined_schedule[machine] = []
                current_time = 0
            else:
                # Get the end time of the last job
                if combined_schedule[machine]:
                    current_time = combined_schedule[machine][-1]['END_TIME']
                else:
                    current_time = 0
            
            for job in jobs:
                # Respect START_DATE constraint if present
                if 'START_DATE' in job and job['START_DATE']:
                    start_time = max(current_time, job['START_DATE'])
                else:
                    start_time = current_time
                    
                processing_time = job.get('HOURS_NEED', job.get('HOURS_NEEDED', 0))
                end_time = start_time + processing_time
                
                # Check due date and log warning if job will be late
                if 'LCD_DATE' in job and job['LCD_DATE'] and end_time > job['LCD_DATE']:
                    lateness = end_time - job['LCD_DATE']
                    job_id = job.get('UNIQUE_JOB_ID', job.get('JOB', 'Unknown'))
                    logger.warning(f"Job {job_id} on machine {machine} will be late by {lateness:.1f} hours")
                    # Store lateness value for later reporting
                    job['BAL_HR'] = -lateness
                
                # Add job to schedule
                combined_schedule[machine].append({
                    **job,
                    'START_TIME': start_time,
                    'END_TIME': end_time
                })
                
                # Update current time
                current_time = end_time
        
        return combined_schedule
    
    def _llm_analyze_bottlenecks(self) -> Dict:
        """Use LLM to analyze bottlenecks in the current job set."""
        # Prepare data for LLM
        machine_stats = self._get_machine_statistics()
        job_stats = self._get_job_statistics()
        
        # Create prompt for LLM
        prompt = f"""
        You are an expert production scheduler tasked with identifying and resolving bottlenecks.
        
        ## Machine Statistics
        ```json
        {json.dumps(machine_stats, indent=2)}
        ```
        
        ## Job Statistics 
        ```json
        {json.dumps(job_stats, indent=2)}
        ```
        
        ## Current Constraints
        - Maximum operators available: {self.max_operators if self.max_operators else 'No limit'}
        
        Analyze the data and identify:
        1. The primary bottleneck machines and why they are bottlenecks
        2. Job patterns that contribute to bottlenecks
        3. Suggested strategies to resolve these bottlenecks
        
        Provide your analysis as a JSON object with the following structure:
        {{"bottleneck_machines": [...], "bottleneck_factors": [...], "resolution_strategies": [...]}}
        """
        
        try:
            # Get LLM response
            response = self.llm.invoke(prompt).content
            # Extract JSON from response
            analysis = self._extract_json_from_response(response)
            logger.info(f"LLM bottleneck analysis: {analysis}")
            return analysis
        except Exception as e:
            logger.error(f"Error in LLM bottleneck analysis: {e}")
            # Return default analysis
            return {
                "bottleneck_machines": self.bottleneck_machines,
                "bottleneck_factors": ["High processing time"],
                "resolution_strategies": ["Prioritize short jobs on bottleneck machines"]
            }
    
    def _llm_prioritize_jobs(self, bottleneck_analysis: Dict) -> List[Dict]:
        """Use LLM to prioritize jobs based on bottleneck analysis."""
        # Extract strategies from bottleneck analysis
        strategies = bottleneck_analysis.get("resolution_strategies", 
                                              ["Prioritize short jobs on bottleneck machines"])
        bottleneck_machines = bottleneck_analysis.get("bottleneck_machines", self.bottleneck_machines)
        
        # Select a subset of jobs for the LLM to analyze (to avoid context length issues)
        sample_jobs = self.jobs_data[:min(20, len(self.jobs_data))]
        
        # Create prompt for LLM
        prompt = f"""
        You are an expert production scheduler. Based on the bottleneck analysis:
        
        ## Bottleneck Machines
        {bottleneck_machines}
        
        ## Resolution Strategies
        {strategies}
        
        ## Jobs to Schedule
        ```json
        {json.dumps(sample_jobs, indent=2)}
        ```
        
        ## Operator Constraints
        Maximum operators available: {self.max_operators if self.max_operators else 'No limit'}
        
        PRIORITY GUIDELINES FOR SCHEDULING:
        1. HIGHEST PRIORITY: Ensure all START_DATE constraints are respected (jobs cannot start before their START_DATE)
        2. Jobs with earlier due dates (LCD_DATE) must be scheduled first, especially if they are at risk of being late
        3. Jobs with higher priority values should be given precedence
        4. For bottleneck machines, prioritize shorter jobs to reduce overall machine idle time
        5. Consider setup time transitions and try to group similar jobs
        6. Respect operator availability constraints
        
        CRITICAL ISSUE: Many jobs are currently scheduled to be late. You must optimize to reduce late jobs.
        
        Return a list of job IDs in order of suggested processing priority as a JSON array: ["job1", "job2", ...]
        """
        
        try:
            # Get LLM response
            response = self.llm.invoke(prompt).content
            # Extract JSON from response
            prioritized_job_ids = self._extract_json_from_response(response)
            
            # Map back to full job data
            id_to_job = {job.get('JOB', '') + job.get('PROCESS_CODE', ''): job for job in self.jobs_data}
            prioritized_jobs = []
            
            # Add prioritized jobs first
            for job_id in prioritized_job_ids:
                if job_id in id_to_job:
                    prioritized_jobs.append(id_to_job[job_id])
            
            # Add remaining jobs
            for job in self.jobs_data:
                job_id = job.get('JOB', '') + job.get('PROCESS_CODE', '')
                if job_id not in prioritized_job_ids:
                    prioritized_jobs.append(job)
                    
            logger.info(f"LLM prioritized {len(prioritized_job_ids)} jobs")
            return prioritized_jobs
        except Exception as e:
            logger.error(f"Error in LLM job prioritization: {e}")
            # Fall back to rule-based prioritization
            return self._rule_based_prioritize_jobs()
    
    def _create_schedule_from_priorities(self, prioritized_jobs: List[Dict]) -> Dict:
        """Create a schedule based on prioritized jobs, considering operator constraints."""
        schedule = {}
        current_time = {}
        operator_count = 0
        
        for job in prioritized_jobs:
            # Skip if max operators reached
            if self.max_operators and operator_count >= self.max_operators:
                continue
                
            machine = job.get('RSC_CODE', job.get('MACHINE_ID', 'Unknown'))
            processing_time = job.get('HOURS_NEED', job.get('HOURS_NEEDED', 0))
            
            # Initialize machine time if not already done
            if machine not in current_time:
                current_time[machine] = 0
                schedule[machine] = []
            
            # Calculate job start and end times
            start_time = current_time[machine]
            end_time = start_time + processing_time
            
            # Add job to schedule
            schedule[machine].append({
                **job,
                'START_TIME': start_time,
                'END_TIME': end_time
            })
            
            # Update machine time
            current_time[machine] = end_time
            
            # Update operator count
            operator_count += job.get('NUMBER_OPERATOR', 1)
        
        return schedule
    
    def _rule_based_prioritize_jobs(self) -> List[Dict]:
        """Prioritize jobs using rule-based approach."""
        # First, create a copy of jobs
        jobs = self.jobs_data.copy()
        
        # Define sorting key function
        def job_priority_key(job):
            machine = job['MACHINE_ID']
            is_bottleneck = machine in self.bottleneck_machines
            process_time = job['HOURS_NEEDED']
            priority = job.get('PRIORITY', 3)  # Default priority 3 if not specified
            
            # Create a tuple for sorting (lower values = higher priority)
            return (
                priority,  # Lower priority number = higher priority
                0 if is_bottleneck else 1,  # Bottleneck machines first
                process_time if is_bottleneck else 0  # For bottleneck machines: shorter jobs first
            )
        
        # Sort jobs by priority key
        prioritized_jobs = sorted(jobs, key=job_priority_key)
        return prioritized_jobs
    
    def _get_machine_statistics(self) -> Dict:
        """Generate statistics for each machine."""
        machine_stats = {}
        
        for job in self.jobs_data:
            machine = job['MACHINE_ID']
            if machine not in machine_stats:
                machine_stats[machine] = {
                    'total_jobs': 0,
                    'total_hours': 0,
                    'avg_time_per_job': 0,
                    'max_time_job': 0,
                    'min_time_job': float('inf'),
                }
            
            # Update stats
            machine_stats[machine]['total_jobs'] += 1
            machine_stats[machine]['total_hours'] += job['HOURS_NEEDED']
            machine_stats[machine]['max_time_job'] = max(machine_stats[machine]['max_time_job'], job['HOURS_NEEDED'])
            machine_stats[machine]['min_time_job'] = min(machine_stats[machine]['min_time_job'], job['HOURS_NEEDED'])
        
        # Calculate averages
        for machine, stats in machine_stats.items():
            if stats['total_jobs'] > 0:
                stats['avg_time_per_job'] = stats['total_hours'] / stats['total_jobs']
            if stats['min_time_job'] == float('inf'):
                stats['min_time_job'] = 0
                
        return machine_stats
    
    def _get_job_statistics(self) -> Dict:
        """Generate statistics for jobs."""
        return {
            'total_jobs': len(self.jobs_data),
            'priority_counts': self._count_priorities(),
            'avg_processing_time': sum(job['HOURS_NEEDED'] for job in self.jobs_data) / max(1, len(self.jobs_data)),
            'total_operators_needed': sum(job.get('NUMBER_OPERATOR', 1) for job in self.jobs_data),
        }
        
    def _get_infeasible_jobs(self, schedule) -> List[Dict]:
        """Identify infeasible or late jobs from a schedule."""
        infeasible_jobs = []
        
        # Check for jobs with lateness (beyond due date)
        for machine, jobs in schedule.items():
            for job in jobs:
                # Check if this job is late (beyond due date)
                if 'DUE_DATE' in job and 'END_TIME' in job:
                    if job['END_TIME'] > job['DUE_DATE']:
                        infeasible_jobs.append(job)
                        continue
                
                # Check for operator constraint violations
                if self.max_operators and job.get('NUMBER_OPERATOR', 1) > self.max_operators:
                    infeasible_jobs.append(job)
                    continue
                    
                # Add other feasibility checks as needed...
        
        return infeasible_jobs
    
    def _resolve_constraints(self, infeasible_jobs, previously_modified: Set[str]):
        """Use LLM to suggest constraint changes to make jobs feasible."""
        if not self.use_llm:
            # Fallback to rule-based constraint resolution
            return self._rule_based_resolve_constraints(infeasible_jobs)
            
        # Prepare data for LLM analysis
        constraint_context = {
            "infeasible_jobs": infeasible_jobs[:min(10, len(infeasible_jobs))],  # Limit to 10 jobs to avoid token limits
            "bottleneck_machines": self.bottleneck_machines,
            "operators_available": self.max_operators,
            "previously_modified": list(previously_modified),
            "all_jobs_count": len(self.jobs_data)
        }
        
        # Create prompt for LLM
        prompt = f"""
        You are an expert production scheduler trying to resolve scheduling conflicts.
        
        ## Infeasible Jobs
        ```json
        {json.dumps(constraint_context['infeasible_jobs'], indent=2)}
        ```
        
        ## Context
        - Bottleneck Machines: {constraint_context['bottleneck_machines']}
        - Operators Available: {constraint_context['operators_available']}
        - Total Jobs: {constraint_context['all_jobs_count']}
        - Previously Modified Jobs: {constraint_context['previously_modified']}
        
        Suggest changes to these job constraints to make them feasible:
        1. Adjust priorities (lower number = higher priority)
        2. Relax due dates where possible
        3. Change operator requirements if needed
        4. Suggest machine reassignments
        
        Return a JSON object with job IDs as keys and suggested changes as values:
        {{
          "job_id1": {{
            "PRIORITY": new_priority,
            "DUE_DATE": new_due_date,
            "NUMBER_OPERATOR": new_operator_count,
            "MACHINE_ID": new_machine
          }},
          "job_id2": {{
            "PRIORITY": new_priority
          }}
        }}
        
        Only include the fields that need to be changed.
        """
        
        try:
            # Get LLM response
            response = self.llm.invoke(prompt).content
            # Extract JSON from response
            constraint_changes = self._extract_json_from_response(response)
            logger.info(f"LLM suggested constraint changes: {constraint_changes}")
            return constraint_changes
        except Exception as e:
            logger.error(f"Error in LLM constraint resolution: {e}")
            # Fall back to rule-based resolution
            return self._rule_based_resolve_constraints(infeasible_jobs)
    
    def _rule_based_resolve_constraints(self, infeasible_jobs):
        """Rule-based approach to resolve constraints for infeasible jobs."""
        constraint_changes = {}
        
        for job in infeasible_jobs:
            job_id = job.get('JOB', '') + job.get('PROCESS_CODE', '')
            changes = {}
            
            # Priority adjustments (reduce priority of late jobs)
            if 'PRIORITY' in job:
                # Higher number = lower priority
                changes['PRIORITY'] = job['PRIORITY'] + 1
            
            # Operator reductions if possible
            if 'NUMBER_OPERATOR' in job and job['NUMBER_OPERATOR'] > 1:
                changes['NUMBER_OPERATOR'] = job['NUMBER_OPERATOR'] - 1
            
            # Machine reassignment (if job is on a bottleneck machine)
            if job['MACHINE_ID'] in self.bottleneck_machines and len(self.machines) > 1:
                # Find a non-bottleneck machine
                for machine in self.machines:
                    if machine not in self.bottleneck_machines:
                        changes['MACHINE_ID'] = machine
                        break
            
            # Add the changes if we made any
            if changes:
                constraint_changes[job_id] = changes
        
        return constraint_changes
    
    def _update_job_constraints(self, jobs, constraint_changes):
        """Apply constraint changes to the jobs data."""
        for i, job in enumerate(jobs):
            job_id = job.get('JOB', '') + job.get('PROCESS_CODE', '')
            
            if job_id in constraint_changes:
                changes = constraint_changes[job_id]
                
                # Apply each change
                for key, value in changes.items():
                    if key in job:
                        # Update the job in-place
                        jobs[i][key] = value
                        logger.debug(f"Updated {key} for job {job_id} to {value}")
    
    def _merge_schedules(self, primary_schedule, secondary_schedule):
        """Merge two schedules, giving priority to the primary schedule."""
        merged = copy.deepcopy(primary_schedule)
        
        # Collect all job IDs in the primary schedule
        primary_job_ids = set()
        for machine, jobs in primary_schedule.items():
            for job in jobs:
                job_id = job.get('JOB', '') + job.get('PROCESS_CODE', '')
                primary_job_ids.add(job_id)
        
        # Add jobs from secondary schedule that aren't in the primary schedule
        for machine, jobs in secondary_schedule.items():
            if machine not in merged:
                merged[machine] = []
                
            for job in jobs:
                job_id = job.get('JOB', '') + job.get('PROCESS_CODE', '')
                
                if job_id not in primary_job_ids:
                    # Calculate new start time to avoid conflicts
                    if merged[machine]:
                        last_end_time = max(j['END_TIME'] for j in merged[machine])
                        job['START_TIME'] = last_end_time
                        job['END_TIME'] = last_end_time + job['HOURS_NEEDED']
                    else:
                        job['START_TIME'] = 0
                        job['END_TIME'] = job['HOURS_NEEDED']
                        
                    merged[machine].append(job)
                    primary_job_ids.add(job_id)  # Mark as processed
        
        return merged
    
    def _count_priorities(self) -> Dict:
        """Count jobs by priority level."""
        priority_counts = {}
        for job in self.jobs_data:
            priority = job.get('PRIORITY', 3)  # Default to 3 if not specified
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        return priority_counts
    
    def _extract_json_from_response(self, response: str) -> Any:
        """Extract JSON from an LLM response string."""
        try:
            # Try to parse directly
            return json.loads(response)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from text
            try:
                # Look for JSON between triple backticks
                if "```json" in response and "```" in response.split("```json", 1)[1]:
                    json_str = response.split("```json", 1)[1].split("```", 1)[0].strip()
                    return json.loads(json_str)
                # Look for JSON between curly braces
                elif "{" in response and "}" in response:
                    json_str = response[response.find("{"):response.rfind("}")+1]
                    return json.loads(json_str)
                # Look for JSON between square brackets
                elif "[" in response and "]" in response:
                    json_str = response[response.find("["):response.rfind("]")+1]
                    return json.loads(json_str)
                else:
                    logger.error(f"Could not extract JSON from response: {response}")
                    return {}
            except Exception as e:
                logger.error(f"Error extracting JSON from response: {e}")
                return {}


# Execute this code when running the script directly
if __name__ == "__main__":
    print("\n===== Testing Agent Scheduler =====\n")
    
    # Sample test data
    test_jobs = [
        {"JOB": "J001", "PROCESS_CODE": "P1", "MACHINE_ID": "M1", "HOURS_NEEDED": 5, "PRIORITY": 1, "NUMBER_OPERATOR": 1},
        {"JOB": "J002", "PROCESS_CODE": "P1", "MACHINE_ID": "M1", "HOURS_NEEDED": 3, "PRIORITY": 2, "NUMBER_OPERATOR": 1},
        {"JOB": "J003", "PROCESS_CODE": "P1", "MACHINE_ID": "M2", "HOURS_NEEDED": 7, "PRIORITY": 1, "NUMBER_OPERATOR": 2},
        {"JOB": "J004", "PROCESS_CODE": "P1", "MACHINE_ID": "M2", "HOURS_NEEDED": 2, "PRIORITY": 3, "NUMBER_OPERATOR": 1},
        {"JOB": "J005", "PROCESS_CODE": "P1", "MACHINE_ID": "M3", "HOURS_NEEDED": 4, "PRIORITY": 2, "NUMBER_OPERATOR": 1},
    ]
    
    test_machines = ["M1", "M2", "M3"]
    test_setup_times = {}
    
    # Create scheduler
    print("Initializing Agent Scheduler...")
    scheduler = AgentScheduler(test_jobs, test_machines, test_setup_times, max_operators=3)
    
    # Identify bottlenecks
    print(f"\nBottleneck machines identified: {scheduler.bottleneck_machines}")
    
    # Create schedule
    print("\nGenerating schedule...")
    schedule = scheduler.schedule()
    
    # Define a helper function to print the schedule in a human-readable format
    def print_schedule_human_readable(schedule, bottleneck_machines, max_operators):
        print("\n" + "="*60)
        print("PRODUCTION SCHEDULE SUMMARY".center(60))
        print("="*60)
        
        # Print bottleneck information
        print("\n📊 BOTTLENECK ANALYSIS")
        if bottleneck_machines:
            print(f"Bottleneck Machines: {', '.join(bottleneck_machines)}")
            print("These machines are critical paths in the production process.")
        else:
            print("No significant bottlenecks detected in the current job set.")
        
        print(f"\nOperator Constraint: Maximum {max_operators if max_operators else 'Unlimited'} operators")
        
        # Calculate some statistics
        total_jobs = 0
        total_hours = 0
        machine_hours = {}
        earliest_start = float('inf')
        latest_end = 0
        
        for machine, jobs in schedule.items():
            total_jobs += len(jobs)
            machine_hours[machine] = 0
            
            for job in jobs:
                job_hours = job['HOURS_NEEDED']
                total_hours += job_hours
                machine_hours[machine] += job_hours
                
                if job['START_TIME'] < earliest_start:
                    earliest_start = job['START_TIME']
                if job['END_TIME'] > latest_end:
                    latest_end = job['END_TIME']
        
        makespan = latest_end - earliest_start
        
        # Print schedule statistics
        print(f"\n📈 SCHEDULE STATISTICS")
        print(f"Total Jobs: {total_jobs}")
        print(f"Total Processing Hours: {total_hours}")
        print(f"Makespan (Total Schedule Duration): {makespan} hours")
        print(f"Machine Utilization:")
        for machine, hours in machine_hours.items():
            utilization = hours / makespan * 100 if makespan > 0 else 0
            print(f"  - {machine}: {hours} hours ({utilization:.1f}% utilization)")
        
        # Print the schedule for each machine
        print("\n📅 DETAILED SCHEDULE BY MACHINE")
        for machine, jobs in schedule.items():
            bottleneck_flag = "⚠️ BOTTLENECK" if machine in bottleneck_machines else ""
            print(f"\n{machine} {bottleneck_flag}")
            print("-" * 60)
            print(f"{'Job':10} {'Priority':10} {'Start':10} {'End':10} {'Duration':10} {'Operators':10}")
            print("-" * 60)
            
            for job in jobs:
                job_id = f"{job['JOB']}-{job['PROCESS_CODE']}"
                priority = job.get('PRIORITY', '-')
                start = job['START_TIME']
                end = job['END_TIME']
                duration = job['HOURS_NEEDED']
                operators = job.get('NUMBER_OPERATOR', 1)
                
                print(f"{job_id:10} {priority:10} {start:10} {end:10} {duration:10} {operators:10}")
        
        print("\n" + "="*60)
        print("END OF SCHEDULE".center(60))
        print("="*60)
    
    # Print the schedule in human-readable format
    print_schedule_human_readable(schedule, scheduler.bottleneck_machines, scheduler.max_operators)