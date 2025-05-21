# Parts Grinding Code

This repository contains all code related to the Part C Grinding Parts Count group project.
The project focuses on real-time classification and visualization of grinding parts using machine learning techniques.

## 📁 Project Structure

```
parts-grinding-code/
├── CircularBuffer.py           # Implements a circular buffer for data handling
├── collect_and_train.py        # Script to collect data and train the model
├── real_time_classifier.py     # Real-time classification script
├── real_time_graphs.py         # Real-time data visualization
├── kernel_setup.txt            # Instructions for setting up file directories
├── README.md                   # Project documentation
└── LICENSE                     # MIT License
```

## 🚀 Getting Started

### Prerequisites

* Python 3.8 or higher
* Recommended: Use a virtual environment to manage dependencies

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Thead12/parts-grinding-code.git
   cd parts-grinding-code
   ```

2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**

   > Create a `requirements.txt` file listing all dependencies for easier installation.

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the file directories:**

   Refer to `kernel_setup.txt` for detailed instructions on setting up the necessary file directories.
   Proper configuration is essential for the scripts to function correctly.

## 🧪 Usage

### Collecting Data and Training the Model

Use `collect_and_train.py` to collect data and train your machine learning model.

```bash
python collect_and_train.py
```

> Ensure that your data is organized as specified in `kernel_setup.txt` before running this script.

### Real-Time Classification

Run `real_time_classifier.py` to perform real-time classification of grinding parts.

```bash
python real_time_classifier.py
```

### Real-Time Visualization

Use `real_time_graphs.py` to visualize data in real-time.

```bash
python real_time_graphs.py
```

## 🛠️ Configuration

The project utilizes `CircularBuffer.py` to manage data streams efficiently.
Ensure that all configurations are set as per the guidelines in `kernel_setup.txt`.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

*For any issues or questions, please open an issue in the repository.*

Python library for managing long running tasks, we can use this to initialise and close the program automatically I believe.
https://supervisord.org/

UPDATE AT THE END TO BE A PROFESSIONAL README, NEEDS USAGE INSTRUCTIONS AS THE TECH DOC REFERS THE USER HERE
Need to include instructions for setting up the file directories as I have them in the kernel_setup.txt file else the configs will break
