from options import task_launcher

def run(ps_tasks):
    print(ps_tasks)

if __name__ == '__main__':
    task_launcher(run)()