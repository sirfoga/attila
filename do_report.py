from pathlib import Path

from attila.util.config import get_env
from attila.util.plots import plot_history
from attila.util.io import load_pickle, append_rows2text
from attila.experiments.tools import run2tex, runs2tex

_here = Path('.').resolve()


def make_history(config, out_path):
    nruns = config.getint('experiments', 'nruns')
    last_epochs = int(config.getint('training', 'epochs') * 0.8)

    for nrun in range(nruns):
        folder = out_path / 'run-{}'.format(nrun)

        for model_folder in filter(lambda p: p.is_dir(), folder.iterdir()):
            summary_file = model_folder / config.get('experiments', 'output file')
            summary = load_pickle(summary_file)

            plot_history(
                summary['history'],
                last=last_epochs,
                out_folder=model_folder
            )
            print('img saved in {}'.format(model_folder))


def make_tex(config, out_path):
    out_file = out_path / config.get('experiments', 'output tables')

    nruns = config.getint('experiments', 'nruns')
    all_runs = []

    for nrun in range(nruns):
        folder = out_path / 'run-{}'.format(nrun)
        summary_file = folder / config.get('experiments', 'output file')
        _out_file = folder / config.get('experiments', 'output tables')  # todo use each model result file

        run_results = run2tex(summary_file, config, _out_file)

        all_runs.append(run_results)

    rows, _ = runs2tex(all_runs)
    append_rows2text(rows, out_file)



def main():
    config, _, out_path, _ = get_env(_here)

    # make_tex(config, out_path)
    make_history(config, out_path)



if __name__ == '__main__':
  main()
