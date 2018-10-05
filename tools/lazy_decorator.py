"""
Useful tool in order to access cached version of properties and functions which have to be executed just once
(especially useful for building up the tensorflow computation graph)
adapted from: https://stevenloria.com/lazy-properties/ and
https://danijar.com/structuring-your-tensorflow-models/
"""
import functools


def lazy_property(function):
    """
    caches the output of the property and just returns the value for next calls
    :param function: property to be cached
    :return: cached output of property
    """
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def lazy_function(function):
    """
    caches the output of the function and just returns the value for next calls
    :param function: function to be cached
    :return: cached output of function
    """
    attribute = '_cache_' + function.__name__

    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator
