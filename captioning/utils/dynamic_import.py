import importlib


def get_class_by_name(name):
    """
    Gets a class by its fully qualified name
    :param name: fully qualified class name eg "mypackage.mymodule.MyClass"
    :return: the requested class
    """
    splits = name.split('.')
    module_name = '.'.join(splits[:-1])
    class_name = splits[-1]
    loaded_module = importlib.import_module(module_name)
    loaded_class = getattr(loaded_module, class_name)

    return loaded_class
