from pathlib import Path
from util.config import get_config

here = Path('.').resolve()


def main():
  config = get_config(here / './config.ini')
  print(here)


if __name__ == '__main__':
  main()
