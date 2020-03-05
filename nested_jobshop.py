from __future__ import print_function

import collections

# Import Python wrapper for or-tools CP-SAT solver.
from ortools.sat.python import cp_model


def MinimalJobshopSat():
    """Minimal jobshop problem."""
    # Create the model.
    model = cp_model.CpModel()

    jobs_data = [  # task = (machine_ids, processing_time, delay_time, deliver_time).
        [((0, 1), 3, 1, 1),
         ((2,), 1, 1, 1)],  # Job0

        [((0, 2), 1, 0, 0), 
         ((1,), 4, 1, 0)],  # Job1

        [((1,), 4, 0, 26), 
         ((2,), 3, 1, 0)]  # Job2
    ]

    machines_count = 1 + max(max(task[0]) for job in jobs_data for task in job)
    all_machines = range(machines_count)

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(sum(task[1:]) for job in jobs_data for task in job)

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start process_end delay_end deliver_end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index process_end delay_end deliver_end')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            occup_machines = task[0]
            duration = task[1] + task[2]    # processing + delay
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            process_var = model.NewIntVar(0, horizon, 'process_end' + suffix)
            delay_var = model.NewIntVar(0, horizon, 'delay_end' + suffix)
            deliver_var = model.NewIntVar(0, horizon, 'deliver_end' + suffix)

            # Each interval includes its processing + delay
            interval_var = model.NewIntervalVar(start_var, duration, delay_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(
                start=start_var, process_end=process_var, delay_end=delay_var, deliver_end=deliver_var, interval=interval_var)
            
            for machine in occup_machines:
                machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Set delay constraints
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job)):
            # A task must wait for its preceeding task to finish and deliver
            delay_time = job[task_id][2]
            model.Add(all_tasks[job_id, task_id].delay_end == all_tasks[job_id, task_id].process_end + delay_time)

    # Set deliver constraints
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job)):
            # A task must wait for its preceeding task to finish and deliver
            deliver_time = job[task_id][3]
            model.Add(all_tasks[job_id, task_id].deliver_end == all_tasks[job_id, task_id].delay_end + deliver_time)

    # Precedences inside a job
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            # A task must wait for its preceeding task to finish and deliver
            deliver_time = job[task_id][3]
            model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].deliver_end)

    # Makespan objective.
    # The Makespan includes deliver time
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].deliver_end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                occup_machines = task[0]
                for machine in occup_machines:
                    assigned_jobs[machine].append(
                        assigned_task_type(
                            start=solver.Value(all_tasks[job_id, task_id].start),
                            job=job_id,
                            index=task_id,
                            process_end=solver.Value(all_tasks[job_id, task_id].process_end),
                            delay_end=solver.Value(all_tasks[job_id, task_id].delay_end),
                            deliver_end=solver.Value(all_tasks[job_id, task_id].deliver_end))                            )

        # Create per machine output lines.
        output = ''
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line = '           '

            for assigned_task in assigned_jobs[machine]:
                name = 'job_%i_%i' % (assigned_task.job, assigned_task.index)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-10s' % name

                start = assigned_task.start
                process_end = assigned_task.process_end
                delay_end = assigned_task.delay_end
                deliver_end = assigned_task.deliver_end

                # Format for each task: [start, process_end, delay_end, deliver_end]
                sol_tmp = '[%i,%i,%i,%i]' % (start, process_end, delay_end, deliver_end)
                # Add spaces to output to align columns.
                sol_line += '%-10s' % sol_tmp

            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print('Optimal Schedule Length: %i' % solver.ObjectiveValue())
        print('Task Format: [start, process_end, delay_end, deliver_end]')
        print(output)


MinimalJobshopSat()
