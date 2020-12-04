from pathlib import Path

from attila.util.config import get_env
from attila.util.plots import plot_history
from attila.util.io import load_pickle, append_rows2text
from attila.experiments.tools import experiment2tex, runs2tex

_here = Path('.').resolve()


def dirs(folder):
    return filter(
        lambda p: p.is_dir(),
        folder.iterdir()
    )


def get_summary(folder):
    summary_file = folder / config.get('experiments', 'output file')
    return load_pickle(summary_file)


def make_history(config, out_path):
    last_epochs = int(config.getint('training', 'epochs') * 0.8)

    for run_folder in dirs(out_path):
        for model_folder in dirs(run_folder):
            summary = get_summary(model_folder)
            plot_history(
                summary['history'],
                last=last_epochs,
                out_folder=model_folder
            )
            print('img saved in {}'.format(model_folder))


def make_tex(config, out_path):
    all_runs = []

    for run_folder in dirs(out_path):
        for model_folder in dirs(run_folder):
            summary = get_summary(model_folder)
            name, summary = experiment2tex(summary)
            run_results[name] = summary

        all_runs.append(run_results)

    rows, _ = runs2tex(all_runs)
    out_file = out_path / config.get('experiments', 'output tables')
    append_rows2text(rows, out_file)



def main():
    config, _, out_path, _ = get_env(_here)

    make_tex(config, out_path)
    make_history(config, out_path)



if __name__ == '__main__':
  main()
