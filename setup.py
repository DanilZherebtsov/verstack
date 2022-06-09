from setuptools import setup, find_packages
import os

# parse __version__ from version.py
exec(open('verstack/version.py').read())

# parse long_description from readme.md
with open("README.md", "r") as fh:
    long_description = fh.read()

# we conditionally add python-snappy based on the presence of an env var
dependencies = ['pandas', 'numpy']
rtd_build_env = os.environ.get('READTHEDOCS', False)
if not rtd_build_env:
    dependencies.append('xgboost')
    dependencies.append('scikit-learn==1.0.1')
    dependencies.append('lightgbm==3.3.0')
    dependencies.append('optuna==2.10.0')
    dependencies.append('plotly==5.3.1')
    dependencies.append('matplotlib')
    dependencies.append('python-dateutil==2.8.1')
    dependencies.append('holidays==0.11.3.1')
    dependencies.append('mlxtend')
    dependencies.append('tensorflow==2.7.0')
    dependencies.append('keras==2.7.0')
    dependencies.append('category_encoders==2.4.0')

setup(
  name = 'verstack',
  packages = find_packages(),
  version = __version__,
  license='MIT',
  description = "Machine learning tools to make a Data Scientist's work more efficient",
  long_description=long_description,
  long_description_content_type="text/markdown",
  author = 'Danil Zherebtsov',
  author_email = 'danil.com@me.com',
  url = 'https://github.com/DanilZherebtsov/verstack',
  download_url = 'https://github.com/DanilZherebtsov/verstack/archive/refs/tags/3.1.9.zip',
  keywords = ['impute', 'missing', 'values', 'stratify', 'nan', 'continuous', 'multiprocessing', 'concurrent', 'timer'],
  install_requires=dependencies,
  classifiers=[
  'Development Status :: 4 - Beta',
  'Intended Audience :: Developers',
  'Topic :: Software Development :: Build Tools',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3',
  'Programming Language :: Python :: 3.4',
  'Programming Language :: Python :: 3.5',
  'Programming Language :: Python :: 3.6',
  'Programming Language :: Python :: 3.7'
  ]
)
