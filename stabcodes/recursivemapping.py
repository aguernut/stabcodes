"""
Implements a custom data structure midway between a :obj:`MutableMapping` and a :obj:`MutableSequence`.
"""


from typing import Optional, Union, Any, Self
from itertools import chain
from copy import copy
from collections.abc import MutableMapping, Mapping, Sequence, MutableSequence, Hashable


class RecursiveMapping(MutableMapping, MutableSequence):
    """
    Implements the behaviour of a :obj:`list`, :obj:`dict[Hashable, list]`, :obj:`dict[Hashable, dict[Hashable, list]]`...

    The structure must be uniform, the key at each level are enforced to be of the same type
    and should be (totally) comparable. This class is similar to a recursive SortedDict, with
    some little tweaks (:meth:`SortedDict.peekitem` is the default behaviour of __get__(int)
    if the keys are not integers, iter() iterates over the base level values and not the top
    level keys...).

    Examples
    --------
    >>> c = RecursiveMapping({1: {"a": [1], "b": [2]}, 2: {"e": [3], "f": [4]}, 3: {}})
    >>> c
    RecursiveMapping({1: RecursiveMapping({'a': RecursiveMapping([1]), 'b': RecursiveMapping([2])}), 2: RecursiveMapping({'e': RecursiveMapping([3]), 'f': RecursiveMapping([4])})})
    >>> c[1]["c"] = [2.5]
    >>> c
    RecursiveMapping({1: RecursiveMapping({'a': RecursiveMapping([1]), 'b': RecursiveMapping([2]), 'c': RecursiveMapping([2.5])}), 2: RecursiveMapping({'e': RecursiveMapping([3]), 'f': RecursiveMapping([4])})})
    >>> del c[2]
    >>> c
    RecursiveMapping({1: RecursiveMapping({'a': RecursiveMapping([1]), 'b': RecursiveMapping([2]), 'c': RecursiveMapping([2.5])})})
    >>> del c[1]["c"][0]
    >>> c
    RecursiveMapping({1: RecursiveMapping({'a': RecursiveMapping([1]), 'b': RecursiveMapping([2]), 'c': RecursiveMapping([])})})

    """

    def __init__(self, value: Optional[Union[Sequence, Mapping]] = None):
        """Builds a RecursiveMapping from a sequence or a possibly recursive uniform mapping, ending with sequences.

        Parameters
        ----------
        value: Union[Sequence, Mapping], optional
            When :obj:`None`, builds the empty :obj:`RecursiveMapping`.
            When given a sequence, builds a base level :obj:`RecursiveMapping` containing its elements.
            When given a :obj:`RecursiveMapping`, performs a shallow copy.
            When given a mapping, recursively calls itself to build the :obj:`RecursiveMapping`.

        Raises
        ------
        TypeError
            Raised when non-uniform keys would be used or an input of the wrong type is given.

        Examples
        --------
        >>> a = RecursiveMapping()
        >>> b = RecursiveMapping([1, 2, 3])
        >>> c = RecursiveMapping({"a": [1, 2], "b": range(3)})
        >>> c
        RecursiveMapping({'a': RecursiveMapping([1, 2]), 'b': RecursiveMapping([0, 1, 2])})

        """
        if value is None:
            self._container = {}
            self._type = ()

        elif isinstance(value, type(self)):
            self._container = value._container
            self._type = value._type

        elif isinstance(value, Sequence):
            self._container: Union[list[Any], dict[Hashable, Self]] = list(value)
            self._type = (int,)

        elif isinstance(value, Mapping):
            self._container = {}
            self._type = ()
            for (key, v) in value.items():
                v_as_recmap = type(self)(v)
                if v_as_recmap._type == ():
                    continue
                self._container[key] = v_as_recmap

                if self._type == ():
                    self._type = (type(key),) + self._container[key]._type
                else:
                    if ((type(key),) + self._container[key]._type) != self._type:
                        raise TypeError("Unconsistent type found during RecursiveMapping creation")

        else:
            raise TypeError(f"Expected a sequence or mapping, got {value} ({type(value).__name__}).")

    def __getitem__(self, index: Union[Hashable, int], /, *, _bykey=True): # type: ignore
        """
        Custom special __getitem__ method, with a flag controlling the by key or by index access.

        Parameters
        ----------
        index:
            Key to look for if `_bykey` is :obj:`True`, otherwise index of the element to look for.
        _bykey:
            Internal flag to toggle the get by index behaviour. When :obj:`False`, will not try to look for integer keys but directly goes to the base level.

        Raises
        ------
        TypeError
            Raised when an index/key of the wrong type as been used.
        IndexError
            Raised when the element with given index/key does not exist

        Examples
        --------
        >>> c = RecursiveMapping({"a": [1, 2], "b": range(3)})
        >>> c["a"]
        RecursiveMapping([1, 2])
        >>> c[2]
        0
        >>> d = RecursiveMapping({1: [1, 2], 2: range(3)})
        >>> d[2]
        RecursiveMapping([0, 1, 2])
        >>> d.__getitem__(2, _bykey=False) # Use peekitem() instead
        0
        """
        if len(self._type) == 0:
            raise IndexError(f"{type(self).__name__} is empty")

        if len(self._type) == 1:
            if not isinstance(index, int):
                raise IndexError(f"Final index {index} must be of type int, not {type(index).__name__}.")
            return self._container[index]

        if _bykey and isinstance(index, self._type[0]):
            if not isinstance(self._container, dict):
                raise TypeError
            return self._container.__getitem__(index)

        if not isinstance(index, int):
            raise TypeError(f"Expected type {self._type[0].__name__}{' or int' if self._type[0] != int else ''}, got {index} ({type(index).__name__}).")

        cum_len = 0
        for (key, v) in self.items():
            cum_len += (last := len(v))
            if cum_len > index:
                return v.__getitem__(index - cum_len + last, _bykey=False)

        raise IndexError(f"{type(self).__name__} index out of range ({index} >= {len(self)}).")

    def peekitem(self, index: int):
        """
        Implements the peekitem method from :obj:`SortedDict`, that is accesses element by index.
        Parameters
        ----------
        index:
            Key to look for if `_bykey` is :obj:`True`, otherwise index of the element to look for.

        Raises
        ------
        TypeError
            Raised when an index of the wrong type as been used.
        IndexError
            Raised when the element with given index does not exist

        Examples
        --------
        >>> d = RecursiveMapping({1: [1, 2], 2: range(3)})
        >>> d[2]
        RecursiveMapping([0, 1, 2])
        >>> d.peekitem(2)
        0
        """
        return self.__getitem__(index, _bykey=False)

    def __setitem__(self, index, value, /, *, _bykey=True): #type: ignore
        """
        Custom special __setitem__ method, with a flag controlling the by key or by index access.

        Parameters
        ----------
        index:
            Key whose value to set if `_bykey` is :obj:`True`, otherwise index of the element to modify.
        value:
            Value to set for the given key/index.

        _bykey:
            Internal flag to toggle the set by index behaviour. When :obj:`False`, will not try to look for integer keys but directly goes to the base level.

        Examples
        --------
        >>> c = RecursiveMapping({"a": [1, 2], "b": range(3)})
        >>> c["a"][0] = 0
        >>> c
        RecursiveMapping({'a': RecursiveMapping([0, 2]), 'b': RecursiveMapping([0, 1, 2])})
        >>> c["c"] = [3, 4]
        >>> c
        RecursiveMapping({'a': RecursiveMapping([0, 2]), 'b': RecursiveMapping([0, 1, 2]), 'c': RecursiveMapping([3, 4])})
        >>> c[2] = 15
        >>> c
        RecursiveMapping({'a': RecursiveMapping([0, 2]), 'b': RecursiveMapping([15, 1, 2]), 'c': RecursiveMapping([3, 4])})
        """
        if len(self._type) == 0:
            value_as_recmap = type(self)(value)
            self._container = {index: value_as_recmap}
            self._type = (type(index),) + value_as_recmap._type

        if len(self._type) == 1:
            self._container[index] = value
            return

        if _bykey and isinstance(index, self._type[0]):
            value_as_recmap = type(self)(value)
            if value_as_recmap._type != self._type[1:]:
                raise TypeError(f"Unconsistent type {value_as_recmap._type}, expected {self._type[1:]}")
            self._container[index] = value_as_recmap
            return

        if not isinstance(index, int):
            raise TypeError(f"Expected type {self._type[0].__name__}{' or int' if self._type[0] != int else ''}, got {index} ({type(index).__name__}).")

        cum_len = 0
        for (key, v) in self.items():
            cum_len += (last := len(v))
            if cum_len > index:
                v.__setitem__(index - cum_len + last, value, _bykey=False)
                return

        raise IndexError(f"{type(self).__name__} index out of range ({index} >= {len(self)}).")

    def __delitem__(self, index, /, *, _bykey=True): #type: ignore
        """
        Custom special __delitem__ method, with a flag controlling the by key or by index access.

        Parameters
        ----------
        index:
            Key to delete if `_bykey` is :obj:`True`, otherwise index of the element to delete.
        _bykey:
            Internal flag to toggle the get by index behaviour. When :obj:`False`, will not try to look for integer keys but directly goes to the base level.

        Raises
        ------
        TypeError
            Raised when an index/key of the wrong type as been used.
        IndexError
            Raised when the element with given index/key does not exist

        Examples
        --------
        >>> c = RecursiveMapping({"a": [1, 2], "b": range(3), "c": [3]})
        >>> del c[3]
        >>> c
        RecursiveMapping({'a': RecursiveMapping([1, 2]), 'b': RecursiveMapping([0, 2]), 'c': RecursiveMapping([3])})
        >>> del c["b"]
        >>> c
        RecursiveMapping({'a': RecursiveMapping([1, 2]), 'c': RecursiveMapping([3])})
        >>> d = RecursiveMapping({"a": c})
        >>> d
        RecursiveMapping({'a': RecursiveMapping({'a': RecursiveMapping([1, 2]), 'c': RecursiveMapping([3])})})
        >>> del d["a"]["c"][0]
        >>> d
        RecursiveMapping({'a': RecursiveMapping({'a': RecursiveMapping([1, 2]), 'c': RecursiveMapping([])})})

        """
        if len(self._type) == 0:
            raise IndexError(f"{type(self).__name__} is empty")

        if len(self._type) == 1:
            del self._container[index]

        elif _bykey and isinstance(index, self._type[0]):
            del self._container[index]

        elif not isinstance(index, int):
            raise TypeError(f"Expected type {self._type[0].__name__}{' or int' if self._type[0] != int else ''}, got {index} ({type(index).__name__}).")

        else:
            cum_len = 0
            for (key, v) in self.items():
                cum_len += (last := len(v))
                if cum_len > index:
                    v.__delitem__(index - cum_len + last, _bykey=False)
                    # if not v:
                    #     del self._container[key]
                    break
            else:
                raise IndexError(f"{type(self).__name__} index out of range ({index} >= {len(self)}).")

        if not self._container and len(self._type) != 1:
            self._type = ()
            self._container = {}

    def __iter__(self):
        """
        Iterates through the base-level values in the data structure.

        The order is given by the key ordering and then by the order is the base-level sequence.

        Examples
        --------
        >>> d = RecursiveMapping({"a": [1, 2], "z": range(3), "Z": [3]})
        >>> list(d)  # "Z" < "a" in ASCII
        [3, 1, 2, 0, 1, 2]
        >>> binary = RecursiveMapping({1: {0: ["right-left"], 1: ["right-right"]}, 0: {0: ["left-left"], 1: ["left-right"]}})
        >>> list(binary)
        ['left-left', 'left-right', 'right-left', 'right-right']

        """
        if len(self._type) == 1:
            return iter(self._container)

        return chain.from_iterable(self.values())

    def __repr__(self):
        return type(self).__name__ + "(" + (repr(self._container) if len(self._type) == 1 else repr(dict(self.items()))) + ")"

    def __len__(self) -> int:
        """
        Number of base-level elements.

        Examples
        --------
        >>> d = RecursiveMapping({"a": [1, 2], "z": range(3), "Z": [3]})
        >>> len(d)
        6
        >>> len(RecursiveMapping())
        0
        >>> len(RecursiveMapping([1, 2]))
        2

        """
        if len(self._type) == 1:
            return len(self._container)

        else:
            return sum(len(subobj) for subobj in self.values())

    def keys(self): #type: ignore
        """
        Iterator over the ordered top-level keys.

        Examples
        --------
        >>> binary = RecursiveMapping({1: {0: ["right-left"], 1: ["right-right"]}, 0: {0: ["left-left"], 1: ["left-right"]}})
        >>> list(binary.keys())
        [0, 1]

        """
        if len(self._type) == 1:
            raise AttributeError(f"Base level of {type(self).__name__} has no keys.")
        return iter(k for (k, _) in self.items())

    def values(self): #type: ignore
        """
        Iterator over the top-level values given in the same order as the ordered keys.

        Examples
        --------
        >>> binary = RecursiveMapping({1: {0: ["right-left"], 1: ["right-right"]}, 0: {0: ["left-left"], 1: ["left-right"]}})
        >>> list(binary.values())
        [RecursiveMapping({0: RecursiveMapping(['left-left']), 1: RecursiveMapping(['left-right'])}), RecursiveMapping({0: RecursiveMapping(['right-left']), 1: RecursiveMapping(['right-right'])})]

        """
        if len(self._type) == 1:
            raise AttributeError(f"Base level of {type(self).__name__} has no values.")
        return iter(v for (_, v) in self.items())

    def __copy__(self):
        if len(self._type) == 0:
            return type(self)()

        if len(self._type) == 1:
            return type(self)(list(self))

        # self._container should be dict
        return type(self)({copy(k): copy(v) for k, v in self._container.items()}) # type: ignore

    def copy(self):
        """
        Deepcopy of this object.

        Examples
        --------
        >>> d = RecursiveMapping({"a": [1, 2], "z": range(3), "Z": [3]})
        >>> e = d.copy()
        >>> e[3] = 4
        >>> d
        RecursiveMapping({'Z': RecursiveMapping([3]), 'a': RecursiveMapping([1, 2]), 'z': RecursiveMapping([0, 1, 2])})
        >>> e
        RecursiveMapping({'Z': RecursiveMapping([3]), 'a': RecursiveMapping([1, 2]), 'z': RecursiveMapping([4, 1, 2])})

        """
        return self.__copy__()

    def items(self): #type: ignore
        """
        Iterator over the top-level ordered given in the same order as the ordered keys.

        Examples
        --------
        >>> d = RecursiveMapping({"a": [1, 2], "z": range(3), "Z": [3]})
        >>> list(d.items())  # "Z" < "a" in ASCII
        [('Z', RecursiveMapping([3])), ('a', RecursiveMapping([1, 2])), ('z', RecursiveMapping([0, 1, 2]))]

        """
        if len(self._type) == 1:
            raise AttributeError(f"Base level of {type(self).__name__} has no keys.")
        
        # self._container should be dict
        return iter(sorted(self._container.items())) # type: ignore

    def popitem(self):
        """
        Removes and returns the first pair (key, value) in the top level.

        Raises
        ------
        AttributeError
            Raised when called on a base-level :obj:`RecursiveMapping`

        KeyError
            Raised when called on an empty :obj:`RecursiveMapping`

        Examples
        --------
        >>> d = RecursiveMapping({"a": [1, 2], "z": range(3), "Z": [3]})
        >>> d.popitem()
        ('Z', RecursiveMapping([3]))
        >>> d["a"].popitem()
        Traceback (most recent call last):
            ...
        AttributeError: Base level of RecursiveMapping has no keys.

        """
        try:
            key = next(iter(self.keys()))
        except StopIteration:
            raise KeyError from None
        value = self[key]
        del self[key]
        return key, value

    def pop(self, *key, _bykey=True): #type: ignore
        """
        Removes and returns the element of given key / index if provided, the last element if no argument is given.

        Parameters
        ----------
        key: optional
            The key or index pointing towards the element to remove.

        Raises
        ------
        TypeError
            Raised if an improper number of parameter or a parameter of the wrong type
            was given
        IndexError
            Raised when the key/index given does not exist.

        Examples
        --------
        >>> d = RecursiveMapping({"a": [1, 2], "z": range(3), "Z": [3]})
        >>> d.pop()
        2
        >>> d
        RecursiveMapping({'Z': RecursiveMapping([3]), 'a': RecursiveMapping([1, 2]), 'z': RecursiveMapping([0, 1])})
        >>> d.pop(0)
        3
        >>> d.pop("a")
        RecursiveMapping([1, 2])
        >>> d
        RecursiveMapping({'Z': RecursiveMapping([]), 'z': RecursiveMapping([0, 1])})

        """
        if len(key) == 0:
            return self.pop(len(self) - 1, _bykey=False)

        if len(key) > 1:
            raise TypeError(f"pop() takes at most one argument ({len(key)} given).")

        value = self.__getitem__(key[0], _bykey=_bykey)
        self.__delitem__(key[0], _bykey=_bykey)
        return value

    def __contains__(self, value) -> bool:
        """
        Special method to check if a value is among the base-level elements.

        Returns
        -------
        bool
            :obj:`True` if `value` is among the base-level elements, otherwise :obj:`False`.

        Examples
        --------
        >>> d = RecursiveMapping({"a": [1, 2], "z": range(3), "Z": [3]})
        >>> "a" in d
        False
        >>> 0 in d
        True

        """
        return any(value == v for v in self)

    def update(self, other: Self): #type: ignore
        """
        Recursively update the :obj:`RecursiveMapping` until the base level is reached.
        As the base-level elements are sequences, they are not updated by replaced when
        the level just above them is reached.

        Parameters
        ----------
        other: RecursiveMapping
            :obj:`RecursiveMapping` to copy conflicting values from.

        Raises
        ------
        TypeError
            Raised if key types mismatch.
        AttributeError
            Raised if called on a base-level object
        NotImplementedError
            Raised when given a non-:obj:`RecursiveMapping` object.

        Examples
        --------
        >>> d = RecursiveMapping({"a": [1, 2], "z": range(3), "Z": [3]})
        >>> e = RecursiveMapping({"b": [1, 2], "Z": [4]})
        >>> d.update(e)
        >>> d
        RecursiveMapping({'Z': RecursiveMapping([4]), 'a': RecursiveMapping([1, 2]), 'b': RecursiveMapping([1, 2]), 'z': RecursiveMapping([0, 1, 2])})

        """
        if not isinstance(other, type(self)):
            raise NotImplementedError

        if self._type == ():
            new = type(self)(other)
            self._container = new._container
            self._type = new._type
            return

        if self._type != other._type:
            raise TypeError("Mismatching types between {type(self).__name__} objects.")

        if not isinstance(self._container, dict) or len(self._type) == 1:
            raise AttributeError(f"Base level of {type(self).__name__} cannot be updated.")

        if len(self._type) > 2:
            for (k, v) in other.items():
                if k in self.keys():
                    self._container[k].update(v)
                else:
                    self._container[k] = v
        else:
            self._container.update(other._container)

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return self._type == other._type and self._container == other._container

    def insert(self, *index): #type: ignore
        """
        Inserts a base-level value.

        If more than on index is given, use them as key before defaulting to integer indexing.

        Parameters
        ----------
        index:
            Non-empty list of keys/indices ending with the value to insert.

        Raises
        ------
        IndexError
            Raised when an improper number of indices are given, or given keys/index do not point towards valid location.
        TypeError
            Keys of improper type given.

        Examples
        --------
        >>> d = RecursiveMapping({"a": [1, 2], "z": range(3), "Z": [3]})
        >>> d.insert(0, 1)
        >>> d
        RecursiveMapping({'Z': RecursiveMapping([1, 3]), 'a': RecursiveMapping([1, 2]), 'z': RecursiveMapping([0, 1, 2])})
        >>> d.insert("a", 2, 3)
        >>> d
        RecursiveMapping({'Z': RecursiveMapping([1, 3]), 'a': RecursiveMapping([1, 2, 3]), 'z': RecursiveMapping([0, 1, 2])})

        """
        index, value = index[:-1], index[-1]
        if len(index) == 0:
            raise IndexError("Expected at least one index, got none.")

        if len(self._type) == 0:
            raise TypeError(f"Cannot insert to an empty {type(self).__name__}.")

        if isinstance(self._container, list):
            if len(index) == 1:
                self._container.insert(index[0], value)
            else:
                IndexError(f"Too much indices given ({len(index)}), expected one.")

        else:
            if len(index) > 1:
                self._container[index[0]].insert(*index[1:], value)
                return

            if not isinstance(index[0], int):
                raise TypeError(f"Expected type {self._type[0].__name__}{' or int' if self._type[0] != int else ''}, got {index[0]} ({type(index[0]).__name__}).")

            cum_len = 0
            for (key, v) in self.items():
                cum_len += (last := len(v))
                if cum_len >= index[0]:
                    v.insert(index[0] - cum_len + last, value)
                    break
            else:
                raise IndexError(f"{type(self).__name__} index out of range ({index} >= {len(self)}).")

    # def index(self):
    #     pass

    def index(self, value, start: int = 0, stop: Optional[int] = None) -> int:
        '''Returns the first index of value in the base level.

        The search only occurs between start and stop indices if provided.

        Parameters
        ----------
        value:
            Value to look for.
        start: int,  optional
            Index at which the search should start.
        stop:
            Index at which to stop the search.

        Raises
        ------
        ValueError
            Raised when the value is not present as a base-level element.

        Examples
        --------
        >>> d = RecursiveMapping({"a": [1, 2], "z": range(3), "Z": [3]})
        >>> d.index(3)
        0
        >>> d.index(2, start=3)
        5
        >>> d.index(2, start=3, stop=5)
        Traceback (most recent call last):
            ...
        ValueError

        '''
        if start is not None and start < 0:
            start = max(len(self) + start, 0)
        if stop is not None and stop < 0:
            stop += len(self)

        i = start
        while stop is None or i < stop:
            try:
                v = self.__getitem__(i, _bykey=False)
                if v is value or v == value:
                    return i
            except IndexError:
                break
            i += 1
        raise ValueError

    def append(self, *index): #type: ignore
        """
        Append a value as last element of the sequence.
        If keys are provided, the value will inserted as the last element of the
        pointed :obj:`RecursiveMapping`.

        Parameters
        ----------
        index:
            Possibly empty list of keys to select the sequence that should host the appended value. This value should be last in the list of parameters.

        Raises
        ------
        TypeError
            Raised when trying to append to an empty :obj:`RecursiveMapping`
        IndexError
            Raised when too much keys are given.
        KeyError
            Raised when a non-existing key is used.

        Examples
        --------
        >>> d = RecursiveMapping({"a": [1, 2], "z": range(3), "Z": [3]})
        >>> d.append(3)
        >>> d
        RecursiveMapping({'Z': RecursiveMapping([3]), 'a': RecursiveMapping([1, 2]), 'z': RecursiveMapping([0, 1, 2, 3])})
        >>> d.append("a", 3)
        >>> d
        RecursiveMapping({'Z': RecursiveMapping([3]), 'a': RecursiveMapping([1, 2, 3]), 'z': RecursiveMapping([0, 1, 2, 3])})

        """
        index, value = index[:-1], index[-1]
        if len(index) == 0:
            self.insert(len(self), value)
            return

        if len(self._type) == 0:
            raise TypeError(f"Cannot insert to an empty {type(self).__name__}.")

        if isinstance(self._container, list) and len(self._type) == 1:
            if len(index) == 0:
                self._container.append(value)
            else:
                raise IndexError(f"Too much indices given ({len(index)}), expected none.")

        else:
            if len(index) > 1:
                self._container[index[0]].append(*index[1:], value)
            else:
                self._container[index[0]].append(value)

    def remove(self, value):
        """
        Removes the first occurence of the base-level element `value`.

        Parameters
        ----------
        value:
            Element to remove

        Raises
        ------
        ValueError
            Raised when `value` is not among the base-level elements

        Examples
        --------
        >>> d = RecursiveMapping({"a": [1, 2], "z": range(3), "Z": [3]})
        >>> d.remove(2)
        >>> d
        RecursiveMapping({'Z': RecursiveMapping([3]), 'a': RecursiveMapping([1]), 'z': RecursiveMapping([0, 1, 2])})
        >>> d.remove(2)
        >>> d
        RecursiveMapping({'Z': RecursiveMapping([3]), 'a': RecursiveMapping([1]), 'z': RecursiveMapping([0, 1])})
        >>> d.remove(2)
        Traceback (most recent call last):
            ...
        ValueError

        """
        self.__delitem__(self.index(value), _bykey=False)

    def extend(self, *index): #type: ignore
        """
        Extends the subsequence pointed by optional indices by the last-given iterable.

        Parameters
        ----------
        index:
            Possibly empty list of keys to select the sequence that should be extended. The values that forms the extension should be an iterable that is given last in the list of parameters.

        Raises
        ------
        TypeError
            Raised when trying to append to an empty :obj:`RecursiveMapping`
        IndexError
            Raised when too much keys are given.
        KeyError
            Raised when a non-existing key is used.

        Examples
        --------
        >>> d = RecursiveMapping({"a": [1, 2], "z": range(3), "Z": [3]})
        >>> d.extend([3, 4])
        >>> d
        RecursiveMapping({'Z': RecursiveMapping([3]), 'a': RecursiveMapping([1, 2]), 'z': RecursiveMapping([0, 1, 2, 3, 4])})
        >>> d.extend("a", [3, 5])
        >>> d
        RecursiveMapping({'Z': RecursiveMapping([3]), 'a': RecursiveMapping([1, 2, 3, 5]), 'z': RecursiveMapping([0, 1, 2, 3, 4])})

        """
        index, value = index[:-1], index[-1]
        if len(index) == 0:
            for v in value:
                self.insert(len(self), v)
            return

        if len(self._type) == 0:
            raise TypeError(f"Cannot insert to an empty {type(self).__name__}.")

        if isinstance(self._container, list) and len(self._type) == 1:
            if len(index) == 0:
                self._container.extend(value)
            else:
                raise IndexError(f"Too much indices given ({len(index)}), expected none.")

        else:
            if len(index) > 1:
                self._container[index[0]].extend(*index[1:], value)
            else:
                self._container[index[0]].extend(value)

    def __iadd__(self, values):
        if len(self._type) == 0:
            raise TypeError(f"Cannot insert to an empty {type(self).__name__}.")
        
        if isinstance(self._container, list):
            self._container.extend(values)

        if len(self._type) > 1:
            raise AttributeError(f"Non base level of {type(self).__name__} cannot be summed. Use update() instead.")

        

        return self

    def get(self, key, default=None):
        """
        Returns the value/base-level element pointed by the given `key` or `default` if the key/index does not exist.

        Parameters
        ----------
        key:
            Key or index that could be pointing towards some data.

        default: optional
            Default value to return if nothing was found.

        Examples
        --------
        >>> d = RecursiveMapping({"a": [1, 2], "z": range(3), "Z": [3]})
        >>> d.get(3)
        0
        >>> d.get("a")
        RecursiveMapping([1, 2])
        >>> d.get(None, 'Something')
        'Something'

        """
        try:
            return self[key]
        except KeyError:
            return default
        except IndexError:
            return default
        except TypeError:
            return default

    setdefault = None #type: ignore

    __reversed__ = None #type: ignore

    reverse = None #type: ignore


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
