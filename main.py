import sys

from streamlit import cli as stcli

if __name__ == "__main__":
  filename = 'GUI.py'
  sys.argv = ["streamlit", "run", filename]
  sys.exit(stcli.main())


