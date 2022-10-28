import pkgutil
from pathlib import Path


HEADING = """API
===

.. autosummary::
    :toctree: _autosummary
    :recursive:

"""

def import_submodules_contents(module_path, base_name: str = ''):
    """ Import contents of all submodules.

    Source:

    Args:
        module: Module.

    Returns:
        Dictionary of imported submodules contents.

    """
    results = []
    for _loader, name, is_package in pkgutil.walk_packages([module_path]):
        full_name = base_name + name
        print(full_name)
        results.append(full_name)
        if is_package:
            results.extend(import_submodules_contents(str(Path(module_path) / name), full_name + '.'))
    return results

def make_module_list(package_dir, output_file):
    with open(output_file, 'w') as outf:
        outf.write(HEADING)
        DISCARD = {'setup'}
        modules = filter(lambda x: x not in DISCARD, import_submodules_contents(package_dir, ''))
        outf.write('\n'.join(map(lambda x: ' ' * 4 + x, modules)))


if __name__ == "__main__":
    make_module_list("..", "api.rst")
