# Smart Till POC

A local proof-of-concept app to detect and count items at a supermarket checkout from a video stream or still image.

## Stack 🥞

  - **Python** 3.11+
  - **Streamlit** (for the UI)
  - **Ultralytics YOLOv8** (for object detection)
  - **Supervision** (for object tracking and annotation)
  - **OpenCV** (for video and image processing)
  - **Docker** (for containerization)

-----

## Run Locally (for Development) 🧑‍💻

This method is for running the app in a local Python virtual environment.

1.  **Activate the virtual environment:**

    ```bash
    source .venv/bin/activate
    ```

2.  **Run the Streamlit app:**

    ```bash
    streamlit run main.py
    ```

-----

## Run with Docker (for Colleagues) 🐳

This is the recommended way to run the application. It ensures the environment is identical for everyone and avoids dependency issues. Your colleague only needs to have **Docker Desktop** installed.

1.  **Build the Docker Image:**
    Open a terminal in the project's root directory and run the following command. This will read the `Dockerfile` and build a self-contained image with all necessary dependencies.

    ```bash
    docker build -t smart-till-app .
    ```

2.  **Run the Docker Container:**
    Once the build is complete, start the application with this command.

    ```bash
    docker run -p 8501:8501 smart-till-app
    ```

3.  **View the Application:**
    Open your web browser and go to the following address:
    **[http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)**