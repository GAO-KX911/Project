# Repository Guidelines

## Project Structure & Module Organization
`Model/` contains training and research code. The main image classification workflow lives in `Model/Classification/Shuffle_PSA/`, and object detection experiments live under `Model/ObjectDetection/Improved_DFFT/`. `Deploy/` contains runnable services: Flask backends for classification, detection, and camera feeds, plus the frontend in `Deploy/flame-detection/`. Large datasets, weights, logs, and captures are intentionally ignored by Git; keep source code and lightweight configs under version control.

## Build, Test, and Development Commands
- `python Model/Classification/Shuffle_PSA/main_train.py` trains the binary classifier.
- `python Model/Classification/Shuffle_PSA/main_eval.py` evaluates a trained classifier on `Test/`.
- `python Model/Classification/Shuffle_PSA/main_predict.py` runs test-set inference and writes CSV summaries.
- `bash Deploy/start.sh` starts the deployed stack: classifier, detector, camera service, and frontend container.
- `bash Deploy/stop.sh` stops the deployed stack.
- `python -m py_compile path/to/file.py` is the minimum syntax check before committing Python changes.

## Coding Style & Naming Conventions
Use 4-space indentation for Python and keep imports grouped by standard library, third-party, then local modules. Follow existing naming: `snake_case` for functions and files, `PascalCase` for classes, and concise experiment directories such as `Train_01` or `lr_0.006`. Prefer small, targeted scripts over large utility modules. Avoid committing generated files under `Our/`, `Predict/`, `log_file/`, or dataset directories.

## Testing Guidelines
There is no single global test suite yet, so validate changes at the script level. For training or inference updates, run `py_compile` first, then execute the affected entry point with a small known dataset slice. For deployment changes, verify the relevant Flask service starts cleanly and check the corresponding log file in `Deploy/*/*.log`. Keep evaluation outputs reproducible by fixing seeds where random splits are introduced.

## Commit & Pull Request Guidelines
Recent history uses short, direct commit subjects, often in Chinese, such as `三分类改二分类`, `增加checkpoint`, and `修改main_predict`. Follow that style: one focused change per commit, imperative wording, no mixed refactors. PRs should include the changed module, the dataset or weight assumptions, the commands used for verification, and screenshots or CSV snippets when frontend or inference outputs change.

## Security & Configuration Tips
Do not commit datasets, model weights, camera credentials, runtime databases, or generated captures. Review `.gitignore` before adding new experiment folders. Keep environment-specific paths configurable when possible, especially in deployment scripts and model weight loading.
