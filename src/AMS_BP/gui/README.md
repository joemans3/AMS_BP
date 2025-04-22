# AMS-BP GUI

The AMS-BP GUI provides a user-friendly interface for constructing and validating configuration files for the AMS-BP fluorescence microscopy simulation framework. It integrates simulation execution, configuration design, log packaging, and data visualization in one cohesive window.

## Overview

This GUI supports:

- A template-based configuration builder for quick setup.
- A visual editor for each simulation parameter block (cells, molecules, fluorophores, lasers, etc.).
- Interactive help sections for each tab.
- Simulation execution using prebuilt configs.
- Napari integration for visualization of resulting microscopy data.
- Packaging of simulation logs for easy sharing.

## Launching the GUI

```bash
run_AMS_BP gui
```
## GUI Structure

### Main Window
The main window (MainWindow) acts as the launchpad for:

- Configuration Builder — Launches a template selector and opens a full editor.
- Run Simulation from Config — Lets you pick a .toml file and start the simulation.
- Visualize Microscopy Data — Opens TIFF and other images in a Napari viewer.
- Package Logs for Sharing — Archives a run_* folder into a .zip.

#### Configuration Editor
Once a template is selected, the ConfigEditor is launched. It contains:

- A dropdown navigation system replacing traditional tabs.
- A floating tab counter and preview/save buttons.
- A help button per section (reads *.md files from help_docs/).
- Live validation using internal config models (convertconfig.py).

##### Tabs & Widgets
Each configuration section is implemented as a modular PyQt widget, including:

- GlobalConfigWidget
- CellConfigWidget
- MoleculeConfigWidget
- FluorophoreConfigWidget
- CondensateConfigWidget
- LaserConfigWidget
- ExperimentConfigWidget
- ...and more

Each widget supports:

- get_data() → Extract validated dictionary data
- set_data(data: dict) → Load existing config into the UI
- validate() → Validate using backend logic
- get_help_path() → Load the corresponding markdown help page

####  Running a Simulation

Once you've completed the config file setup via the GUI:

- Click "Preview Configuration TOML" to confirm contents.
- Click "Ready to Save Configuration" and choose a .toml path.
- Return to the main window, click "Run Simulation from Config".
- Simulation will launch in a background thread and print logs to a live window.
- Once done, results will be saved in a run_*/ folder.

#### Viewing Results

- Click "Visualize Microscopy Data (Napari)"
- Select any .tif, .tiff, .nd2, or .zarr file

The data will be loaded into a new Napari viewer session

#### Packaging Logs

To share or archive a completed simulation:

- Click "Package Logs for Sharing".
- Select the run_* folder you want.
- Choose a destination for the .zip file.
