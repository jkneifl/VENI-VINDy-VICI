import abc


class BaseLibrary(abc.ABC):
    """
    Abstract class for feature libraries
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, x):
        """
        construct features for the input x
        :param x: input
        :return: feature
        """
        pass

    @abc.abstractmethod
    def get_names(self, x):
        """
        construct the names of the features for the input x
        :param x: input
        :return: feature
        """
        pass
