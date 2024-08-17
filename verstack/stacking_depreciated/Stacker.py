'''
Stacker has been removed from the package in August 2024. Latest verstack version with stacker was 3.9.8.
'''

import warnings

class Stacker(object):
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning('Stacker has been removed from verstack in August 2024. If you need to reinstate the Stacker class, please raise an issue at https://github.com/DanilZherebtsov/verstack/issues.')