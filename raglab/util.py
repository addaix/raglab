"""Utils for raglab"""


from config2py import get_configs_folder_for_app
from i2 import Pipe 
from msword.base import bytes_to_doc, get_text_from_docx
import os
from pdfdol.base import bytes_to_pdf_text_pages
import re
from types import SimpleNamespace

app_dir = os.environ.get('RAGLAB_APP_FOLDER', None) or get_configs_folder_for_app(
    'raglab'
)


configs_dir = os.path.join(app_dir, 'configs')
data_dir = os.path.join(app_dir, 'data')
local_stores_dir = os.path.join(app_dir, 'local_stores')
users_dir_name = 'u'
local_space_stores_dir = os.path.join(local_stores_dir, users_dir_name)


for path in [configs_dir, data_dir, local_stores_dir]:
    os.makedirs(path, exist_ok=True)


class IterableNamespace(SimpleNamespace):
    def __iter__(self):
        yield from self.__dict__

    # make it read-only
    def __setattr__(self, name, value):
        raise AttributeError("can't set attribute")


from typing import MutableMapping


class StoreAccess:
    """
    Delegator for MutableMapping, providing list, read, write, and delete methods.

    This is intended to be used in web services, offering nicer method names than
    the MutableMapping interface, and an actual list instead of a generator in
    the case of list.
    """

    store: MutableMapping

    def list(self):
        return list(self.store.keys())

    def read(self, key):
        return self.store[key]

    def write(self, key, value):
        self.store[key] = value

    def delete(self, key):
        del self.store[key]


from i2 import call_forgivingly, Namespace
from typing import Mapping, Callable


class LazyAccessor(Namespace):
    """A namespace of factories, with lazy instantiation.

    Give it a `Mapping` of factory functions, and it will call these factories to make
    object instances only if and when they're requested
    (via `.attribute` or `[key]` access), keeping the instances created in a cache.

    Note that if the factories have parameters, they will be passed to the factories
    when the factories are called, via the `**named_parameters` argument.

    Note that if there are any parameters, any of them whose name matches the name of
    a factory will be used to instantiate the factory.

    >>> def plus_2(x):
    ...     return x + 2
    >>> def times(x, y):
    ...     return x * y
    >>> f = LazyAccessor(dict(plus_2=plus_2, times=times), x=10, y=2)
    >>> list(f)
    ['plus_2', 'times']
    >>> f.plus_2
    12
    >>> f.times
    20

    If you need to make a LazyAccessor factory with a fixed set of object factories,
    you can use `functools.partial` to do so:

    >>> from functools import partial
    >>> object_factories = dict(plus_2=plus_2, times=times)
    >>> mk_accessor = partial(LazyAccessor, object_factories)
    >>> g = mk_accessor(x=10, y=2)
    >>> g.plus_2, g.times
    (12, 20)
    >>> h = mk_accessor(x=3, y=10)
    >>> h.plus_2, h.times
    (5, 30)

    """

    def __init__(self, factories: Mapping[str, Callable], **named_parameters):
        self._factories = factories
        self._named_parameters = named_parameters
        self._cached_objs = {}

    def __getattr__(self, name):
        # Note: Could use a specialized "method" lru_cache instead.
        # See for example: https://stackoverflow.com/questions/33672412/python-functools-lru-cache-with-instance-methods-release-object/68052994#68052994
        if name not in self._cached_objs:
            obj = call_forgivingly(self._factories[name], **self._named_parameters)
            self._cached_objs[name] = obj
        return self._cached_objs[name]

    def __getitem__(self, k):
        return self._factories[k]

    def __iter__(self):
        return iter(self._factories)

    def __dir__(self):
        return list(self)


def clog(condition, *args, log_func=print, **kwargs):
    """Conditional log

    >>> clog(False, "logging this")
    >>> clog(True, "logging this")
    logging this

    One common usage is when there's a verbose flag that allows the user to specify
    whether they want to log or not. Instead of having to litter your code with
    `if verbose:` statements you can just do this:

    >>> verbose = True  # say versbose is True
    >>> _clog = clog(verbose)  # makes a clog with a fixed condition
    >>> _clog("logging this")
    logging this

    You can also choose a different log function.
    Usually you'd want to use a logger object from the logging module,
    but for this example we'll just use `print` with some modification:

    >>> _clog = clog(verbose, log_func=lambda x: print(f"hello {x}"))
    >>> _clog("logging this")
    hello logging this

    """
    if not args and not kwargs:
        import functools

        return functools.partial(clog, condition, log_func=log_func)
    if condition:
        return log_func(*args, **kwargs)

msword_to_string = Pipe(bytes_to_doc, get_text_from_docx)

def pdf_to_string(x) :
    """Converts a PDF to a string."""
    return ''.join(bytes_to_pdf_text_pages(x))

DFLT_FORMAT_EGRESS = {
    'text': lambda x: x,
    'pdf': pdf_to_string,
    'msword': msword_to_string,
}
# note: see from pdfdol.base import bytes_to_pdf_text_pages

def extract_string_from_data_url(data_url, *, format_egress=DFLT_FORMAT_EGRESS):
    """Extract a string from a data URL."""
    import base64

    # Split the URL at the first comma to separate the metadata from the base64 content
    metadata, encoded = data_url.split(',', 1)
    header, *_ = metadata.split('/')
    _, data_format = header.split(':')
    
    egress = format_egress.get(data_format, lambda x: x)


    # Ensure the base64 string is a multiple of 4 in length by padding with '='
    padding = 4 - len(encoded) % 4
    if padding and padding != 4:
        encoded += '=' * padding

    # Decode the base64 string
    original_string = base64.b64decode(encoded).decode('utf-8')

    return egress(original_string)

def is_file_param(string: str) -> bool:
    """Check if a string is a data URL."""
    regex = r"data:(.*?);name=(.*?);base64,(.*)"
    return re.match(regex, string) is not None

def prompt_func_ingress(kwargs: dict) -> dict:
    """Ingress function for prompt functions."""
    # Get the prompt from the kwargs
    def conditional_trans(x):
        if is_file_param(x):
            return extract_string_from_data_url(x)
        else:
            return x
        
    return {k: conditional_trans(v) for k, v in kwargs.items()}