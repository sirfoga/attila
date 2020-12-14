import json
import pickle


def stuff2text(stuff, out_file):
    with open(out_file, 'w') as w:
        w.write(stuff)


def append2text(stuff, out_file):
    with open(out_file, 'a') as w:
        w.write('\n')
        w.write(stuff)


def append_rows2text(rows, out_file):
    stuff = '\n'.join(rows)
    stuff = '\n' + stuff + '\n'
    append2text(stuff, out_file)


def load_json(inp_file):
    with open(inp_file, 'r') as fp:
        return json.load(fp)


def stuff2json(stuff, f_path):
    with open(f_path, 'w') as fp:
        return json.dump(stuff, fp)


def stuff2pickle(stuff, f_path):
    with open(f_path, 'wb') as fp:
        pickle.dump(stuff, fp)


def load_pickle(f_path):
    with open(f_path, 'rb') as fp:
        return pickle.load(fp)


def get_summary(folder, config):
    summary_file = folder / config.get('experiments', 'output file')
    return load_pickle(summary_file)


def dirs(folder):
    return sorted(filter(
        lambda p: p.is_dir(),
        folder.iterdir()
    ))
