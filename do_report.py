from pathlib import Path

from attila.util.config import get_env
from attila.util.plots import plot_history, plot_preds
from attila.util.io import load_pickle, append_rows2text, load_json, get_summary, dirs
from attila.experiments.tools import experiment2tex, runs2tex

_here = Path('.').resolve()


def make_history(config, out_path):
    last_epochs = int(config.getint('training', 'epochs') * 0.8)

    for run_folder in dirs(out_path):
        for model_folder in dirs(run_folder):
            summary = get_summary(model_folder, config)
            plot_history(
                summary['history'],
                last=last_epochs,
                out_folder=model_folder
            )
            print('history img saved in {}'.format(model_folder))


def make_plots(config, out_path):
    for run_folder in dirs(out_path):
        for model_folder in dirs(run_folder):
            summary = get_summary(model_folder, config)
            


def make_tex(models_config, config, out_path):
    all_runs = []

    for run_folder in dirs(out_path):
        run_results = {}

        for model_folder in dirs(run_folder):
            summary = get_summary(model_folder, config)
            name, summary = experiment2tex(summary)
            run_results[name] = summary

        all_runs.append(run_results)

    rows, _ = runs2tex(all_runs, models_config)
    out_file = out_path / config.get('experiments', 'output tables')
    append_rows2text(rows, out_file)



def main():
    config, _, out_path, models_config_path = get_env(_here)
    models_config = load_json(models_config_path)

    make_tex(models_config, config, out_path)
    make_history(config, out_path)
    make_plots(config, out_path)


if __name__ == '__main__':
  main()
