
def write_loss(iteration, loss, file_path):
    # write loss to file, at the end of the file
    with open(file_path, 'a+') as f:
        f.write(f"{iteration},{loss['train_loss']},{loss['test_loss']}\n")


def get_last_loss(file_path):
    # return last iteration and loss from file
    with open(file_path, 'w+') as f:
        lines = f.readlines()
        if len(lines) == 0:
            return None, None, None
        last = lines[-1].split(',')
        return int(last[0]), float(last[1]), float(last[2])