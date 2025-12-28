from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_collection import main


if __name__ == "__main__":
    main()
