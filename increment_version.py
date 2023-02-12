import fileinput
import sys
from verstack.version import __version__ as current_version

files = ['verstack/version.py', 'README.rst', 'docs/source/index.rst']
         
def update_version(file, new_version):
    for line in fileinput.input(file, inplace=1):
        if current_version in line:
            line = line.replace(current_version, new_version)
        sys.stdout.write(line)
    print(f'{new_version} is set in {file}')

if __name__ == '__main__':
    args = sys.argv[1:]
    for file in files:
        update_version(file, args[0])


