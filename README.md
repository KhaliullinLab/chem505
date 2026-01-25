# CHEM 505 - Computational Chemistry Tutorials

This repository contains tutorials and supplementary materials for CHEM 505.

---

## Getting the Tutorials

You have two options to get these files on your computer:

### Option 1: Download as ZIP (Simple but Limited)

1. Go to https://github.com/KhaliullinLab/chem505
2. Click the green **Code** button
3. Select **Download ZIP**
4. Extract the ZIP file to your desired location

**Disadvantage:** You won't be able to easily update your files when new tutorials are added or existing ones are updated. You would need to re-download the entire repository.

---

### Option 2: Clone with Git (Recommended)

Using Git allows you to easily update your local copy when new materials are added.

#### Step 1: Install Git

**macOS:**
- Open Terminal and run: `git --version`
- If not installed, you'll be prompted to install Xcode Command Line Tools
- Or install via Homebrew: `brew install git`
- Guide: https://git-scm.com/download/mac

**Windows:**
- Download from: https://git-scm.com/download/win
- Run the installer and follow the prompts (default options are fine)
- Guide: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install git
```

**Linux (Fedora):**
```bash
sudo dnf install git
```

#### Step 2: Clone the Repository

Open a terminal (or Git Bash on Windows) and run:

```bash
git clone https://github.com/KhaliullinLab/chem505.git
```

This creates a `chem505` folder with all the tutorials.

#### Step 3: Update Your Local Copy

When new tutorials are added, navigate to your `chem505` folder and run:

```bash
cd chem505
git pull
```

This downloads all new and updated files.

---

## Setting Up Your Environment

### Step 1: Install Python

Most tutorials require Python 3.8 or newer.

**macOS:**
- Download from: https://www.python.org/downloads/macos/
- Or install via Homebrew: `brew install python`
- Verify installation: `python3 --version`

**Windows:**
- Download from: https://www.python.org/downloads/windows/
- **Important:** Check "Add Python to PATH" during installation
- Verify installation: `python --version`

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

Verify installation: `python3 --version`

---

### Step 2: Install Cursor (Optional but Recommended)

[Cursor](https://cursor.com) is a modern code editor with built-in AI assistance. It's particularly useful for computational chemistry because:

- **AI-assisted coding**: Get help writing Python scripts for data analysis, parsing output files, and automating workflows
- **Intelligent autocomplete**: Suggestions based on context, including chemistry-specific libraries like RDKit, ASE, and cclib
- **Inline documentation**: Ask questions about code directly in the editor
- **Error debugging**: AI can help diagnose and fix errors in your scripts
- **Learning aid**: Ask the AI to explain unfamiliar code or concepts

**Cursor is free for students for 1 year** with a valid student email.

#### Installation:

1. Go to https://cursor.com
2. Download and install for your operating system
3. Sign up with your student email to get the free Pro plan

#### Opening the Repository in Cursor:

1. Open Cursor
2. Go to **File** → **Open Folder**
3. Navigate to and select your `chem505` folder
4. You now have access to all tutorials with AI assistance

---

## Repository Structure

```
chem505/
├── tutorials/
│   └── g16-basis-sets/      # Gaussian 16 basis set convergence tutorial
│       ├── README.md        # Tutorial instructions
│       ├── requirements.txt # Python dependencies
│       ├── inputs/          # Gaussian input files
│       └── python/          # Python scripts
└── README.md                # This file
```

---

## Working with Tutorials

Each tutorial folder contains its own `README.md` with specific instructions. General workflow:

1. Navigate to the tutorial folder
2. Read the tutorial's `README.md`
3. Install any required Python packages (usually via `pip install -r requirements.txt`)
4. Follow the instructions in the tutorial

### Example: Setting up a tutorial

```bash
cd chem505/tutorials/g16-basis-sets
python3 -m venv venv                    # Create virtual environment
source venv/bin/activate                # Activate it (Linux/macOS)
# On Windows: venv\Scripts\activate
pip install -r requirements.txt         # Install dependencies
```

---

## Need Help?

- Check the tutorial's `README.md` for specific instructions
- If using Cursor, use the AI assistant (Ctrl/Cmd + L) to ask questions
- Contact the instructor during office hours
