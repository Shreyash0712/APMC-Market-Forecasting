# APMC Market Forecasting Project Rules

Follow these core principles when developing this project:

1. **Serialize, Don't Delete:** Never arbitrarily delete files or override historical work. Keep track of everything by serializing changes (e.g., using timestamped or versioned files like `01_db_init.sql`, `02_model_update.py` if appropriate).
2. **Modularity over Monoliths:** Do not smash things into single files. Keep code, scripts, and documentation separate and highly modular for easy access and maintainability as the project scales.
3. **Folder Organization:** Keep everything meticulously organized into purpose-specific folders (e.g., `scripts/`, `docs/`, `db_init/`, `src/models/`, `src/data/`).
4. **Cutting-Edge Technologies:** Always prioritize the most efficient, modern, and cutting-edge technologies and libraries (e.g., PyTorch Geometric, ONNX Runtime, TimescaleDB, Frank-Wolfe variants). If unsure about versioning or stability, pause and ask the user for confirmation.
5. **Code Quality:** Keep all code compact, simple, yet highly efficient. Focus on algorithmic efficiency and latency (sub-50ms target for routing).
6. **File Header Documentation:** Whenever you create a new file (script, config, or source code), always include a highly descriptive comment block at the very top. This header must explain precisely what the file does, its role in the architecture, and any critical dependencies. 
7. **Be Argumentative (Scientific Rigor):** Do not blindly agree with the user. If the user suggests an approach that is theoretically unsound, goes against best practices, or degrades computational efficiency, you must forcefully argue against it, explain the flaw, and propose the optimal mathematical or architectural alternative.
8. **Continuous Knowledge Mapping:** Whenever introducing a new complex technology, mathematical algorithm, or theoretical concept to the project, immediately update `docs/KNOWLEDGE_CHART.md` to explain what it is and where it is being used.
