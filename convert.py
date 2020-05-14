import numpy as np 

def gen_taskset(path, time_conversion):
    raw_tasksets = np.load(path)
    tasksets = []

    time_conversion
    for raw_taskset in raw_tasksets:
        # For the pairs
        taskset = []
        for raw_task in raw_taskset:
            raw_task = raw_task[:-1]
            task = []
            for segment1, segment2 in zip(raw_task[::2], raw_task[1::2]):
                machine_ids = tuple([id+1 for id in segment2[1]])

                local_time = int(segment1[0]*time_conversion)
                remote_time = int(segment2[0]*time_conversion)
                task.append([machine_ids, local_time, remote_time, False])   # False for non-consecutive
        
            # For the last segment
            last_segment = raw_task[-1]
            task.append([(-1,), int(last_segment[0]*time_conversion), 0, False])
            taskset.append(task)
        tasksets.append(taskset)

    return tasksets
