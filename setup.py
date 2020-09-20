
from distutils.core import setup
import setuptools

# parse __version__ from version.py
exec(open('verstack/version.py').read())

# parse long_description from readme.md
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name = 'verstack',
  packages = ['verstack'],
  version = __version__,
  license='MIT',
  description = "Machine learning tools to make a Data Scientist's work more efficient",
  long_description=long_description,
  long_description_content_type="text/markdown",
  author = 'Danil Zherebtsov',
  author_email = 'danil.com@me.com',
  url = 'https://github.com/DanilZherebtsov/verstack',
  download_url = 'https://github.com/DanilZherebtsov/verstack/archive/0.1.1.tar.gz',
  keywords = ['impute', 'missing', 'values', 'stratify', 'nan', 'continuous'],
  install_requires=[
          'pandas',
          'numpy',
          'xgboost'
      ],
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
