
from distutils.core import setup
setup(
  name = 'verstack',
  packages = ['verstack'],
  version = '0.1.0',
  license='MIT',
  description = "Machine learning tools to make a Data Scientist's work more efficient",
  author = 'Danil Zherebtsov',
  author_email = 'danil.com@me.com',
  url = 'https://github.com/DanilZherebtsov/verstack',
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['impute', 'missing', 'values', 'stratify', 'nan'],
  install_requires=[            # I get to this in a second
          'validators',
          'beautifulsoup4',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6'
    'Programming Language :: Python :: 3.7',
  ],
)
