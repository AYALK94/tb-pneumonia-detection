from setuptools import setup, find_packages

setup(
    name="tb-pneumonia-detection",
    version="1.0.0",
    author="Ayalkibet Abriham",
    author_email="ayalkibetabriham94@gmail.com",
    description="YOLOv8-based chest X-ray classification for TB and Pneumonia detection",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi==0.111.0",
        "uvicorn==0.29.0",
        "ultralytics==8.2.0",
        "Pillow==10.3.0",
        "python-multipart==0.0.9",
        "numpy==1.26.4",
        "opencv-python-headless==4.9.0.80",
        "pytest==8.2.0",
        "httpx==0.27.0",
        "pyyaml==6.0.1",
        "starlette==0.37.2",
        "jinja2==3.1.4",
    ],
)