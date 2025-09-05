import os
import json
import ast

try:
    # For Python 3.8+
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from pkg_resources import get_distribution, DistributionNotFound as PackageNotFoundError
    def version(pkg):
        try:
            return get_distribution(pkg).version
        except PackageNotFoundError:
            raise

def find_imports_in_code(code):
    """
    Parse Python code to extract imported module/package names.
    """
    tree = ast.parse(code)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            # Only consider absolute imports
            if node.module and node.level == 0:
                imports.add(node.module.split('.')[0])
    return imports

def collect_imports_from_notebooks(root_dir):
    """
    Walk through root_dir, open each .ipynb, and collect imports.
    """
    libs = set()
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.ipynb'):
                filepath = os.path.join(dirpath, fname)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        notebook = json.load(f)
                    for cell in notebook.get('cells', []):
                        if cell.get('cell_type') == 'code':
                            source = ''.join(cell.get('source', []))
                            libs |= find_imports_in_code(source)
                except Exception as e:
                    print(f"Skipping {filepath}: {e}")
    return libs

def write_requirements(libs, output_file='requirements.txt'):
    """
    Write detected libraries with versions if installed, into requirements.txt
    """
    lines = []
    for pkg in sorted(libs):
        try:
            ver = version(pkg)
            lines.append(f"{pkg}=={ver}")
        except PackageNotFoundError:
            lines.append(pkg)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Written {len(lines)} packages to {output_file}")

if __name__ == '__main__':
    root = os.getcwd()
    print(f"Scanning for imports in Jupyter notebooks under {root}...")
    libs = collect_imports_from_notebooks(root)
    if libs:
        write_requirements(libs)
    else:
        print("No imports found in notebooks.")
