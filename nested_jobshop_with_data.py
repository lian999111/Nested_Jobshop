from __future__ import print_function

import collections
from convert import gen_taskset

# Import Python wrapper for or-tools CP-SAT solver.
from ortools.sat.python import cp_model


def MinimalJobshopSat():
    """Minimal jobshop problem."""
    # Create the model.
    model = cp_model.CpModel()

    # The flag subsequent indicate this task should stick together with the preceeding one
    consecutive = True

    # task = (machine_ids, local_time, remote_time, consecutive).
    # machine_ids:
    #   Tuple of machine Ids the task occupies
    #   Special ids:
    #       0: DON'T USE IT! 0 is reserved for a special purpose to make finish times easy to show.
    #      -1: Use it for tasks that don't run on a shared machine.
    #          When id is set to -1, remote_time is ignored.
    #          It is recommended to explicitly set remote_time to 0 to avoid confusion.
    # local_time:
    #   Time that a job must run on its local machine before occupying a shared machine
    # remote_time:
    #   Time that a job occupies a remote (shared) machine. Equivalent to the duration.
    # consecutive:
    #   Indicate if the the task should continue immediately after its preceeding task
    #   When setting a task to consecutive, local_time is ignored.
    #   It is recommended to explicitly set local_tim to 0 to avoid confusion.
    # jobs_data = [
    #     # Job0
    #     [((1, 2), 3, 1, not consecutive),
    #      ((3,), 1, 1, not consecutive),
    #      ((-1,), 10, 0, not consecutive)],   # final task doesn't occupy shared machine

    #     # Job1
    #     [((1, 3), 1, 2, not consecutive),
    #      ((2,), 4, 1, not consecutive),
    #      ((2, 3), 0, 1, consecutive),
    #      ((3,), 0, 1, consecutive)],         # 2 consecutive tasks

    #     # Job2
    #     [((1,), 4, 2, not consecutive),
    #      ((2,), 3, 1, not consecutive)]      # last task runs on machine 2
    # ]

    time_conversion = 1e14
    tasksets = gen_taskset('tasksets_n80_m100_p8_u0.3_r8_s0.05_l0.1_c5_d2_b20.npy', time_conversion)
    jobs_data = tasksets[0]

    print('Taskset ready.\n')

    # A dummy task is appended to each of the jobs.
    # The 0-th machine is a special reserved one.
    # This dummy task will happen on the 0-th machine 
    # reserved for the special purpose to show the 
    # finish time of each job.
    dummy_task = ((0,), 0, 0, not consecutive)
    for job in jobs_data:
        job.append(dummy_task)

    machines_count = 1 + max(max(task[0]) for job in jobs_data for task in job)
    all_machines = range(machines_count)

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(sum(task[1:-1]) for job in jobs_data for task in job)

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            occup_machines = task[0]
            
            assert min(occup_machines) >= -1, 'Machine id must be >= -1'

            if occup_machines[0] > 0:
                duration = task[2]    # remote time
            else:
                # If the machine id is -1, any set remote time is ignore
                duration = 0

            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)

            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var)

            for machine in occup_machines:
                # Only add intervals to real machines
                # Not necessary to take special care here, just for logical correctness
                if machine > 0:
                    machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        if machine > 0:
            model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job (plus a local_time that it must wait for)
    for job_id, job in enumerate(jobs_data):
        # The first task should wait for its preceeding local time first
        first_local_time = job[0][1]
        model.Add(all_tasks[job_id, 0].start >= first_local_time)
        # For the rest of the tasks
        for task_id in range(len(job)-1):
            if job[task_id + 1][3]: # check the subsequent flag
                # A subsequent task continues after immediately its preceeding task to finish and deliver
                model.Add(all_tasks[job_id, task_id + 1].start ==
                          all_tasks[job_id, task_id].end)
            else:
                # A non-subsequent task must at least wait for its preceeding task plus its preceeding local_time
                local_time = job[task_id + 1][1]
                model.Add(all_tasks[job_id, task_id + 1].start >=
                          all_tasks[job_id, task_id].end + local_time)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    # solver.parameters.max_time_in_seconds = 5
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                occup_machines = task[0]
                for machine in occup_machines:
                    assigned_jobs[machine].append(
                        assigned_task_type(
                            start=solver.Value(
                                all_tasks[job_id, task_id].start),
                            job=job_id,
                            index=task_id,
                            duration=task[2]))

        # Create per machine output lines.
        output = ''

        # Show finish times using the special 0-th machine
        assigned_jobs[0].sort()
        sol_line_tasks = 'Finish Time: '
        sol_line = '             '

        for assigned_task in assigned_jobs[0]:
            name = 'job_%i' % (assigned_task.job)
            # Add spaces to output to align columns.
            sol_line_tasks += '%-10s' % name

            finish_time = assigned_task.start

            # Format for each job's finish time: 
            sol_tmp = '%i' % finish_time
            # Add spaces to output to align columns.
            sol_line += '%-10s' % sol_tmp
        
        sol_line += '\n'
        sol_line_tasks += '\n'
        output += sol_line_tasks
        output += sol_line

        # Show info of real machines
        for machine in all_machines[1:]:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line = '           '

            for assigned_task in assigned_jobs[machine]:
                name = 'job_%i_%i' % (assigned_task.job, assigned_task.index)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-10s' % name

                start = assigned_task.start
                duration = assigned_task.duration

                # Format for each task: [start, end]
                sol_tmp = '[%i,%i]' % (start, start+duration)
                # Add spaces to output to align columns.
                sol_line += '%-10s' % sol_tmp

            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print('Optimal Schedule Length: %i' % solver.ObjectiveValue())
        # print('Task Format: [start, end]')
        # print(output)


MinimalJobshopSat()
