# Contributing to CricketVision AI

Thank you for your interest in contributing to **CricketVision AI**! We welcome contributions from developers, researchers, and sports analytics enthusiasts.

By contributing to this project, you help build a highly precise, open-source tool for biomechanical cricket shot analysis.

---

## 📖 Code of Conduct

Please help maintain a professional, welcoming, and inclusive community space. Be respectful, supportive, and focused on constructive feedback.

---

## 🛠️ How to Contribute

### 1. Report Issues / Feature Requests
If you find a bug or have an idea for an enhancement:
1. Search the [Issue Tracker](https://github.com/himanshu-jadhav108/CricketVision_AI/issues) to ensure it hasn't been reported.
2. If not, open a new issue detailing:
   - Steps to reproduce (for bugs).
   - Expected vs. actual behavior.
   - Environment details (OS, Python version, package versions).

### 2. Submit Changes (Pull Requests)
We follow a standard fork-and-pull workflow:
1. **Fork** the repository to your own account.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CricketVision_AI.git
   cd CricketVision_AI
   ```
3. Create a **Feature Branch** naming it logically:
   ```bash
   git checkout -b feature/expand-pose-features
   ```
4. Install local development dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Implement your modifications following our [Coding Standards](#-coding-standards).
6. **Verify changes** by running the test suite:
   ```bash
   python -m unittest discover -s tests -v
   ```
7. Commit your updates with clear messages:
   ```bash
   git commit -m "feat(cv): add ankle flexion biomechanical index"
   ```
8. Push to your branch and submit a **Pull Request** to the `main` branch of the origin repository.

---

## ⚙️ Coding Standards

To maintain high code quality, please adhere to these standards:

### Python Guidelines
- **Type Hints:** Use explicit type hints for all public function inputs and outputs (e.g. `def predict(self, image_input: ImageInput) -> dict[str, Any]:`).
- **Docstrings:** All modules, classes, and public methods must contain descriptive docstrings using Google Style formatting.
- **PEP 8:** Follow standard PEP 8 naming and formatting conventions.
- **Imports:** Group imports logically (Standard Library -> Third-party dependencies -> Local modules). Remove any unused imports before committing.

### Code Abstractions
- **Stateless Inference:** Keep inference pipelines decoupled from serving frameworks (Gradio / FastAPI). Ensure the core `ShotPredictor` class operates independently.
- **Configuration Management:** Do not hardcode ports, paths, or thresholds. Decouple them into variables inside `config.py` and manage them via environment variables.

---

## 🧪 Testing Requirements

We enforce strict validation before merging any code:
- **Zero Breakages:** All existing unit tests must pass successfully.
- **New Coverage:** If you implement a new feature or algorithm, you should provide corresponding unit test coverage in the `tests/` directory.

To run tests locally:
```bash
python -m unittest discover -s tests -v
```

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
