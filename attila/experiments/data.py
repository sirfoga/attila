import pickle


def save_stuff(stuff, f_path):
    with open(f_path, 'wb') as fp:
        pickle.dump(stuff, fp)


def load_stuff(f_path):
    with open(f_path, 'rb') as fp:
        return pickle.load(fp)
