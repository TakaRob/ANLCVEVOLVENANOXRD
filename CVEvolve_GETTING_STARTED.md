# CVEvolve Getting Started

This guide explains how to set up CVEvolve in a way that keeps the framework,
your data, and your experiment outputs separate.

The mental model is:

- CVEvolve is the framework code.
- Your experiment folder contains `config.yaml`, `task.md`, data, and run outputs.
- `uv` manages the Python environment for CVEvolve.
- CVEvolve creates a separate session workspace for each run.
- MLflow tracks run-level metrics and artifacts.
- Hutch tracks candidate lineage, operators, fitness, and provenance.

## 1. Recommended Directory Layout

Use separate folders for code, data, and experiments. Do not copy your data into
the CVEvolve repository, and do not clone CVEvolve inside your data folder.

Example:

```text
~/projects/
  CVEvolve/                    # framework clone

~/experiments/
  my-cv-task/
    config.yaml                # experiment config
    task.md                    # task prompt for the agent
    holdout_test.md            # optional holdout prompt
    data/                      # development data used by the main agent
    holdout/                   # optional holdout data
    runs/                      # CVEvolve session outputs
    mlruns/                    # local MLflow store
    hutch/                     # local Hutch DB
```

With this layout, you run commands from `~/experiments/my-cv-task`, while using
the CVEvolve environment from `~/projects/CVEvolve`.

## 2. Clone CVEvolve

If you already have a clone, skip this step.

```bash
mkdir -p ~/projects
cd ~/projects
git clone <CVEvolve repo URL> CVEvolve
```

For the local checkout used while writing this guide, the path is:

```text
/home/beams0/XYIN/projects/CVEvolve
```

Replace that path with your own clone path in the commands below.

## 3. Understand `uv` commands

`uv sync` reads CVEvolve's `pyproject.toml` and `uv.lock`, then creates or
updates the framework environment. For this project, that environment lives at:

```text
/home/beams0/XYIN/projects/CVEvolve/.venv
```

`uv run --project /path/to/CVEvolve ...` runs a command using that environment,
even if your current directory is somewhere else.

## 4. Create The CVEvolve Environment

From the CVEvolve clone:

```bash
cd /home/beams0/XYIN/projects/CVEvolve
uv sync
```

If you want native Hutch tracking, include the optional Hutch dependency:

```bash
uv sync --extra hutch
```

You can verify the CLI:

```bash
uv run cvevolve --help
```

## 5. Create An Experiment Folder

Create a folder for one task:

```bash
mkdir -p ~/experiments/my-cv-task
cd ~/experiments/my-cv-task

mkdir -p data holdout runs mlruns hutch
```

Put your development data under `data/`. Put optional holdout data under
`holdout/`.

Example:

```text
~/experiments/my-cv-task/data/
  images/
  labels/
  metadata.csv

~/experiments/my-cv-task/holdout/
  images/
  labels/
  metadata.csv
```

The exact layout is up to your task. CVEvolve does not require a fixed dataset
schema. You explain the layout in `task.md`.

## 6. Data Rules

`workspace.data_dir` is the development set. CVEvolve copies it into the session
workspace before the agent starts.

If your config says:

```yaml
workspace:
  root_dir: ./runs
  data_dir: ./data
```

then CVEvolve copies:

```text
~/experiments/my-cv-task/data/
```

into:

```text
~/experiments/my-cv-task/runs/<session-name>/workspace/data/
```

The agent works on that copied snapshot, not your original data folder.

Holdout data is different. `workspace.holdout_data_dir` is not copied into the
main development workspace. It is copied only into temporary per-candidate
holdout test workspaces after a candidate is submitted.

This is intentional: the main search agent should not tune directly on holdout
data.

## 7. Write `task.md`

Create `task.md` in your experiment folder.

Use relative paths under `data/`. Avoid absolute host paths in the task prompt.

Example:

```markdown
1. Introduction

Find a robust translation-only image registration method for the provided image
pairs.

2. Data

The development data is under `data/`.

- Moving images are under `data/images/moving/`.
- Fixed images are under `data/images/fixed/`.
- Labels are under `data/labels/`.
- File stems match across folders.

3. Evaluation

The candidate should predict x/y translations for each pair. Evaluate against
the labels using mean Euclidean translation error. Lower is better.

4. Output

Candidate scripts should expose command-line arguments for input data and output
locations. Do not hardcode paths.
```

Avoid markdown `#` section headers because CVEvolve injects this prompt into
larger templates. Numbered section names are safer.

## 8. Write `config.yaml`

Create `config.yaml` in the experiment folder.

Example with MLflow and embedded Hutch:

```yaml
name: my-cv-task

model:
  model_name: gpt-4.1
  api_key_env_var: OPENAI_API_KEY
  api_base: null
  temperature: 0.2
  max_retries: 5
  rate_limit_resend_attempts: 5
  rate_limit_sleep_seconds: 60

workspace:
  root_dir: ./runs
  data_dir: ./data
  holdout_data_dir: null
  additional_skill_dirs: []
  require_dangerous_command_approval: true

metric:
  name_hint: registration_error
  direction_hint: minimize
  target_value: null
  description_hint: Mean Euclidean translation error. Lower is better.

branching:
  warmup_rounds: 3
  force_generate_every: 0
  generate_batch_size: 1
  tune_every: 3
  evolve_every: 2
  min_excellent_for_tune: 1
  min_successful_for_evolve: 2
  lineage_selection_temperature: 1.0
  crossover_candidates_per_lineage: 2
  crossover_same_lineage_penalty: 0.3
  exclude_poor_lineages: true
  exclude_lineages_with_failure_streak: 0

tracking:
  enabled: true
  mlflow_tracking_uri: file:///home/beams0/XYIN/experiments/my-cv-task/mlruns
  mlflow_experiment_name: CVEvolve

hutch:
  enabled: true
  project: cvevolve
  run_id: null
  daemon_url: null
  db_path: /home/beams0/XYIN/experiments/my-cv-task/hutch/hutch.duckdb
  strict: false

num_workers_generate: 1
num_workers_tune: 1
cap_num_requests: null

stopping:
  max_rounds: 12
  patience_rounds: 4
  min_improvement: 0.0
```

Use absolute paths for MLflow and Hutch URIs because they may be read by other
tools later. Relative paths such as `./runs` and `./data` are fine when you run
commands from the experiment folder.

### What `file://` Means

`file://` marks a value as a local filesystem URI. In this guide, it is used for
MLflow because `tracking.mlflow_tracking_uri` expects a URI-like tracking store
location.

For an absolute Linux path, the local file URI has three slashes:

```text
file:// + /home/beams0/XYIN/experiments/my-cv-task/mlruns
= file:///home/beams0/XYIN/experiments/my-cv-task/mlruns
```

Use `file://` here:

```yaml
tracking:
  mlflow_tracking_uri: file:///home/beams0/XYIN/experiments/my-cv-task/mlruns
```

Do not use `file://` for normal filesystem paths:

```yaml
workspace:
  root_dir: ./runs
  data_dir: ./data

hutch:
  db_path: /home/beams0/XYIN/experiments/my-cv-task/hutch/hutch.duckdb
```

Short rule:

```text
MLflow tracking URI: file:///absolute/path/mlruns
Hutch DB path:       /absolute/path/hutch.duckdb
Data path:           ./data or /absolute/path/data
Run output path:     ./runs or /absolute/path/runs
API base URL:        https://...
```

## 9. Export API Keys

At minimum, set the model API key:

```bash
export OPENAI_API_KEY=...
```

Optional:

```bash
export OPENAI_BASE_URL=...              # only for OpenAI-compatible providers
export SEMANTIC_SCHOLAR_API_KEY=...     # optional paper search key
export TAVILY_API_KEY=...               # required for Tavily web search
```

If `model.api_key_env_var` in `config.yaml` is not `OPENAI_API_KEY`, export that
variable instead.

## 10. Use Argo Instead Of OpenAI

If you are using Argonne's Argo Gateway API, you do not need an OpenAI API key.
CVEvolve can use Argo through its OpenAI-compatible endpoint.

Relevant Argo details:

- Production OpenAI-compatible base URL:

  ```text
  https://apps.inside.anl.gov/argoapi/v1
  ```

- CVEvolve/Argo calls must be made from the Argonne internal network, or through
  VPN on an Argonne-managed computer.
- Use your Argonne domain username as the API key value.
- Do not use your full email address.
- For application usage, consider requesting an Argo service account.
- Tool/function calling is supported on the non-streaming `/chat` endpoint.

For CVEvolve, set the model block like this:

```yaml
model:
  model_name: gpt55
  api_key_env_var: ARGO_USERNAME
  api_base: https://apps.inside.anl.gov/argoapi/v1
  max_retries: 3
```

Then export your Argonne username before running:

```bash
export ARGO_USERNAME=your_anl_domain_username
```

Do not include quotes in the value, and do not use `your.name@anl.gov`.

You can verify model access with:

```bash
curl https://apps.inside.anl.gov/argoapi/v1/models \
  -H "Authorization: Bearer $ARGO_USERNAME"
```

The Argo docs list model examples such as:

```text
gpt5
gpt55
claudeopus46
claudeopus48
```

For CVEvolve, prefer a model that supports OpenAI chat completions and tool
calling.

The rest of the CVEvolve setup is unchanged. MLflow and Hutch do not care
whether the model backend is OpenAI or Argo.

## 11. Run From The Experiment Folder

From `~/experiments/my-cv-task`:

```bash
cd ~/experiments/my-cv-task

uv run --project /home/beams0/XYIN/projects/CVEvolve --extra hutch \
  cvevolve run --config ./config.yaml --prompt ./task.md
```

This command does three things:

- Uses CVEvolve's environment from `/home/beams0/XYIN/projects/CVEvolve/.venv`.
- Keeps your current working directory as `~/experiments/my-cv-task`.
- Resolves relative config paths such as `./data` and `./runs` from the
  experiment folder.

## 12. What Happens During A Run

CVEvolve creates:

```text
./runs/my-cv-task/
  config.snapshot.yaml
  mlflow_run.json
  workspace/
    data/
    prompt/task_prompt.md
    candidates/
    skills/user/
    cvevolve_git_commit.txt
  history/
    search_history.sqlite
  exports/
    candidates.csv
    metrics.csv
    evaluation_metrics.csv
    holdout_test_metrics.csv
    evolution_tree.csv
    rounds.csv
  reports/
    final_report.md
    final_summary.json
    best_candidate.py
    best_metric_by_round.png
    candidate_lineage.png
```

There are two Python environments to understand:

1. CVEvolve controller environment:

   ```text
   /home/beams0/XYIN/projects/CVEvolve/.venv
   ```

   This runs the CVEvolve CLI, agent controller, MLflow tracking, and Hutch
   tracking.

2. Session workspace environment:

   ```text
   ./runs/my-cv-task/workspace/.venv
   ```

   Candidate scripts and task-specific dependencies are created inside the
   session workspace when the agent uses its `uv` tool.

This separation is useful: CVEvolve's own dependencies do not get mixed with
candidate experiment dependencies.

## 13. MLflow Tracking

MLflow is enabled by:

```yaml
tracking:
  enabled: true
  mlflow_tracking_uri: file:///home/beams0/XYIN/experiments/my-cv-task/mlruns
  mlflow_experiment_name: CVEvolve
```

Start the local MLflow UI:

```bash
cd ~/experiments/my-cv-task

uv run --project /home/beams0/XYIN/projects/CVEvolve \
  mlflow ui --backend-store-uri ./mlruns --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

CVEvolve logs:

- flattened config parameters,
- `config.snapshot.yaml`,
- `metric` at each round,
- `best_metric_so_far`,
- `round_type`,
- `holdout_test_metric` when available,
- `history/search_history.sqlite`,
- final reports and plots,
- best candidate source file,
- CVEvolve git commit tag.

Round type codes:

```text
0 = baseline
1 = generate
2 = tune
3 = evolve
```

## 14. Hutch Tracking

Hutch is optional. It records provenance and lineage, not just scalar metrics.

CVEvolve emits:

- run start and run end,
- candidate individuals,
- operators such as propose, refine, mutate, and crossover,
- primary fitness,
- holdout fitness,
- candidate failures.

### Option A: Embedded Hutch

Embedded mode writes directly to a local DuckDB file. No separate Hutch server
is needed during the run.

Config:

```yaml
hutch:
  enabled: true
  project: cvevolve
  db_path: /home/beams0/XYIN/experiments/my-cv-task/hutch/hutch.duckdb
  daemon_url: null
  strict: false
```

Use this when you want local provenance with minimal setup.

### Option B: Hutch Daemon

Daemon mode runs a Hutch server and CVEvolve sends events to it. Use this if you
want live dashboard behavior during a long run.

Terminal 1:

```bash
cd ~/experiments/my-cv-task

uv run --project /home/beams0/XYIN/projects/CVEvolve --extra hutch \
  hutch serve --db ./hutch/hutch.duckdb --port 7777
```

Terminal 2:

```bash
cd ~/experiments/my-cv-task

export HUTCH_DAEMON_URL=http://127.0.0.1:7777

uv run --project /home/beams0/XYIN/projects/CVEvolve --extra hutch \
  cvevolve run --config ./config.yaml --prompt ./task.md
```

Config for daemon mode:

```yaml
hutch:
  enabled: true
  project: cvevolve
  daemon_url: http://127.0.0.1:7777
  db_path: null
  strict: false
```

`strict: false` means Hutch problems produce warnings but do not stop the
CVEvolve run. `strict: true` means Hutch tracking errors fail the run.

## 15. Optional Holdout Testing

Holdout testing requires both:

1. `workspace.holdout_data_dir` in `config.yaml`.
2. `--holdout-test-prompt ./holdout_test.md` on the run command.

Example config:

```yaml
workspace:
  root_dir: ./runs
  data_dir: ./data
  holdout_data_dir: ./holdout
```

Example command:

```bash
uv run --project /home/beams0/XYIN/projects/CVEvolve --extra hutch \
  cvevolve run \
  --config ./config.yaml \
  --prompt ./task.md \
  --holdout-test-prompt ./holdout_test.md
```

Example `holdout_test.md`:

```markdown
The holdout folder has the same structure as the development data:

- `images/moving/`
- `images/fixed/`
- `labels/`
- `metadata.csv`

Use the same metric as the main task. Report the holdout metric after running
the submitted candidate on this folder.
```

During normal development, the main agent sees only the folder description, not
the holdout files themselves.

## 16. Resume A Run

If a run is interrupted, resume from the session root:

```bash
cd ~/experiments/my-cv-task

uv run --project /home/beams0/XYIN/projects/CVEvolve --extra hutch \
  cvevolve resume --session ./runs/my-cv-task
```

Resume is round-level. If interruption happened midway through a round, CVEvolve
reruns that round rather than continuing from the exact tool call.

MLflow resumes the same run using:

```text
./runs/my-cv-task/mlflow_run.json
```

Hutch uses the configured `hutch.run_id`, or defaults to:

```text
cvevolve-<session-name>
```

## 17. Upload An Existing Session To MLflow

If MLflow was not enabled during the run, or you want to upload DB-backed
history later:

```bash
cd ~/experiments/my-cv-task

uv run --project /home/beams0/XYIN/projects/CVEvolve \
  cvevolve upload ./runs/my-cv-task \
  --mlflow-tracking-uri file:///home/beams0/XYIN/experiments/my-cv-task/mlruns \
  --mlflow-experiment-name CVEvolve
```

This reads:

```text
./runs/my-cv-task/config.snapshot.yaml
./runs/my-cv-task/history/search_history.sqlite
```

and logs metrics/artifacts to MLflow.

## 18. Common Questions

### Should `config.yaml` and `task.md` be inside the CVEvolve repo?

No. Put them in your experiment folder.

### Should I copy my data into the CVEvolve repo?

No. Keep data in the experiment folder or a dataset folder. Point
`workspace.data_dir` at it.

### Should I clone CVEvolve inside my data folder?

No. Keep the framework clone separate.

### Where does the CVEvolve environment live?

With `uv sync`, it lives inside the CVEvolve clone:

```text
/home/beams0/XYIN/projects/CVEvolve/.venv
```

### Where do candidate dependencies live?

Inside the session workspace, usually:

```text
./runs/<session-name>/workspace/.venv
```

### Which paths should be absolute?

Use absolute paths for tracking stores:

- `tracking.mlflow_tracking_uri`
- `hutch.db_path`

Relative paths are fine for `workspace.root_dir` and `workspace.data_dir` if you
always run from the experiment folder.

### What should I back up?

For a finished experiment, keep:

```text
config.yaml
task.md
holdout_test.md
runs/<session-name>/
mlruns/
hutch/
```

Also keep the raw data or a stable pointer to it.

## 19. Quick Start Checklist

```bash
# 1. Clone and prepare CVEvolve
cd ~/projects
git clone <CVEvolve repo URL> CVEvolve
cd ~/projects/CVEvolve
uv sync --extra hutch

# 2. Create experiment folder
mkdir -p ~/experiments/my-cv-task/{data,holdout,runs,mlruns,hutch}
cd ~/experiments/my-cv-task

# 3. Add your files
# - put development data in ./data
# - write ./task.md
# - write ./config.yaml

# 4. Set model credential
export OPENAI_API_KEY=...
# or, if using Argo:
# export ARGO_USERNAME=your_anl_domain_username

# 5. Run
uv run --project ~/projects/CVEvolve --extra hutch \
  cvevolve run --config ./config.yaml --prompt ./task.md

# 6. View MLflow
uv run --project ~/projects/CVEvolve \
  mlflow ui --backend-store-uri ./mlruns --port 5000
```
