from examples import bAbI15, degree

if __name__ == '__main__':
    import sys
    if len(sys.argv) <= 1:
        print("#Usage main.py <task> [<arguments>...]")
    task = sys.argv[1]
    sys.argv.pop(1)
    
    if task == 'degree':
        degree.train()
    elif task == 'bAbI15':
        bAbI15.train()
    else:
        print("Invalid task: {}".format(task))
    