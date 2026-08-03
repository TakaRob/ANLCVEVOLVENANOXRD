"""Discover, create, configure, and validate an xrd-app project in VS Code."""

# %% [markdown]
# # Project setup
# Edit the form below, then run cells from top to bottom. Existing projects are
# never overwritten. An xrd-app project is identified by `config.yaml`; its scan
# registry is `Raw/scans.json`.

# %% Configuration
from pathlib import Path

CREATE_NEW = False
SEARCH_DIRECTORY = Path("/home/takaji")
PROJECT_NAME = "rocking_203_214"
SELECT_PROJECT_NAME = "rocking_203_214"
SCAN_NUMBER = 203

# Optional project-wide sources. Use None to leave the current value unchanged.
RAW_ROOT = None
POSITION_ROOT = None
TTH_MAP = None
REFLECTIONS = None
DETECTOR_SCRIPT = None

# Scan discovery reads HDF5 headers. Enable only when adding/updating scans.
REGISTER_SCANS = False
DEEP_SCAN_CHECK = False

# %% Imports
from xrd_app.config import DataManager, ProjectConfig
from xrd_app.core import io
from xrd_app.workspace import create_project

# %% Find projects
# Resolve the directory once so all displayed paths are absolute.
search_root = SEARCH_DIRECTORY.expanduser().resolve()
if not search_root.is_dir():
    raise FileNotFoundError(f"Search directory does not exist: {search_root}")

# Search recursively for the canonical xrd-app project marker.
project_roots = sorted({path.parent for path in search_root.rglob("config.yaml")})
print(f"Found {len(project_roots)} project(s) below {search_root}:")
for index, root in enumerate(project_roots, 1):
    print(f"  {index:>2}. {root}")

# %% Select or safely create
# Resolve the requested new-project path under the search directory.
target_root = search_root / PROJECT_NAME
if CREATE_NEW:
    if (target_root / "config.yaml").exists():
        print(f"Project already exists; opening without modification: {target_root}")
        project_root = target_root
    else:
        # Create the standard project tree and seed default reflections.
        project_root = create_project(
            PROJECT_NAME, workspace=search_root, scan_number=SCAN_NUMBER
        )
        print(f"Created project: {project_root}")
else:
    matches = [
        root for root in project_roots
        if root.name == SELECT_PROJECT_NAME or str(root) == SELECT_PROJECT_NAME
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one project named {SELECT_PROJECT_NAME!r}; found {len(matches)}. "
            "Set SELECT_PROJECT_NAME to a unique directory name or full path."
        )
    project_root = matches[0]

# Load and validate the selected project's YAML configuration.
config = ProjectConfig.load(project_root)
if not config.exists() or config.root != project_root.resolve():
    raise FileNotFoundError(f"Not an xrd-app project: {project_root}")
print(f"Selected project: {config.data.get('name', config.root.name)} at {config.root}")

# %% Apply the optional hardcoded source form
source_form = {
    "raw_root": RAW_ROOT,
    "position_root": POSITION_ROOT,
    "tth_map": TTH_MAP,
    "reflections": REFLECTIONS,
    "detector_script": DETECTOR_SCRIPT,
}
configured_sources = config.data.setdefault("data_sources", {})
changed = False
for key, value in source_form.items():
    if value is not None:
        # Normalize configured inputs before persisting them.
        configured_sources[key] = str(Path(value).expanduser().resolve())
        changed = True

if changed:
    # Save only when the user supplied at least one source value.
    config.save()
    print(f"Updated sources in {config.config_path}")
else:
    print("No source values supplied; configuration was not changed.")

# %% Optionally discover and register raw scans
# Construct the project path resolver used by every later notebook.
dm = DataManager(config=config, scan=SCAN_NUMBER)
if REGISTER_SCANS:
    raw_root = dm.config.get("data_sources", "raw_root")
    if not raw_root:
        raise ValueError("Set RAW_ROOT before enabling REGISTER_SCANS.")

    # Discover scan directories and detector metadata from the configured raw root.
    discovered = io.discover_scans(raw_root, deep=DEEP_SCAN_CHECK)
    expected_shape = config.get("detector", "shape")
    if not expected_shape:
        expected_shape = next(
            (item["shape"] for item in discovered if item.get("shape")), None
        )

    valid = []
    for item in discovered:
        # Validate frame counts and detector shape before registration.
        problems = io.validate_scan(item, expected_shape=expected_shape)
        print(f"{item['name']}: {'OK' if not problems else '; '.join(problems)}")
        if not problems:
            valid.append(item)

    registry = dm.scans_registry()
    for item in valid:
        registry[item["name"]] = {
            key: item[key]
            for key in ("dir", "frames_dir", "n_files", "n_frames", "shape")
        }

    # Persist the canonical scan registry consumed by DataManager.
    dm.write_scans_registry(registry)
    config.data.setdefault("detector", {})["shape"] = expected_shape
    config.data["scans"] = registry
    if not config.get("scan", "name") and len(valid) == 1:
        name = valid[0]["name"]
        config.data["scan"] = {
            "number": DataManager.scan_number_of(name), "name": name
        }
    # Persist detector shape and mirrored scan metadata.
    config.save()
    print(f"Registered {len(valid)} valid scan(s) in {dm.scans_registry_path()}")

# %% Validate the ready project
# Recreate DataManager after any configuration updates.
dm = DataManager(config.root, scan=SCAN_NUMBER)
scans = dm.discover_scans()
print(f"Project: {dm.root}")
print(f"Scans: {', '.join(scans) if scans else '(none registered)'}")
print(f"Selected scan: {dm.scan_name}")

resolved = {
    "raw scan": dm.raw_scan_dir(),
    "positions": dm.position_csv(),
    "2-theta map": dm.tth_map(),
    "reflections": dm.reflections(),
}
for label, path in resolved.items():
    print(f"{label:>12}: {path}  [{'OK' if Path(path).exists() else 'MISSING'}]")
