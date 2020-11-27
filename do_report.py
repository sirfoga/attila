from pathlib import Path

from attila.util.config import get_env
from attila.experiments.tools import out2tex

_here = Path('.').resolve()


def main():
    config, _, out_path, _ = get_env(_here)
    out_f = out_path / config.get('experiments', 'output tables')
    out2tex(config, out_path, out_f)


if __name__ == '__main__':
  main()
