# Gesture Based Presentation

A Python project for controlling presentations using hand gestures. Interact with your slides in various ways—like flipping pages, cropping and placing images, and drawing directly on the screen—all through your webcam.

---

## ✨ Features & Gesture Guide

Here is a list of the core gestures you can use in this project.

### 👆 Pointer
-   **Right Index Finger**: Moves the on-screen pointer.

### 📄 Page Control
-   **Right Index + Middle Finger**: Skips forward 10 pages.
-   **Left Index + Middle Finger**: Skips back 10 pages.
-   **Right Fist with left or right gaze**: Navigates one page forward or backward.

### 🖼️ Image Control (Crop, Rotate, Zoom, Move)
-   **Insert Image(Crop)**: Select an area with both index fingers and hold to copy it to the note canvas.
-   **Select Menu**: Point at a menu icon with your left index finger and hold to select it.
-   **Zoom**: Use your right thumb and index finger to zoom the image from 0.5x to 3x.
-   **Rotate**: Use both index fingers to rotate the image.
-   **Move**: Use your right index finger to move the image.
-   **Confirm Placement**: After positioning the image, select the check (✔) icon from the menu to place it on the canvas.
    -   Placed images remain on the canvas and can be cropped multiple times.

### ✍️ Writing Mode
-   **Enter Mode**: Make and hold a fist with your left hand to start Writing Mode. (Default pen: black).
-   **Change Pen**: Select a different pen or color from the menu with your left index finger and hold.
-   **To Draw**:
    1.  Extend your **right index finger** to get ready to draw.
    2.  With your index finger extended, **fold your thumb** to draw and **extend it** to stop drawing.
-   **Clear Canvas**: Show your open left palm to the camera to erase all annotations.
-   **Exit Mode**: Release your left fist to exit Annotation Mode.

---

## 🛠️ Tech Stack

-   Python 3.11
-   OpenCV
-   MediaPipe
-   CVZONE

---

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

**1. Clone the Repository**
```bash
git clone https://github.com/spilak-stack/gesture_based_presentation/
cd gesture_based_presentation
```

**2. Create and Activate Virtual Environment**
> **Note**: This project has been tested in a Python 3.11 environment.

```bash
# Create a virtual environment using uv
uv venv -p python3.11

# Activate the virtual environment (Windows PowerShell)
.venv\Scripts\activate
# source .venv/bin/activate
```

**3. Install Dependencies**
Install all required packages using the `requirements.txt` file.
```bash
uv pip install -r requirements.txt
```

**4. Add Presentation Files**
Place your presentation files (e.g., image files) into the `source` folder.

**5. Run the Program**
```bash
python run.py
```
---

## 📄 License


This project is licensed under the MIT License.

