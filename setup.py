from setuptools import find_packages, setup

setup(
    name="videoswin",
    packages=find_packages(exclude=["notebooks", "assets", "data"]),
    version="2.0.0",
    license="Apache License 2.0",
    description="Video Swin Transformerss in Keras 3",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mohammed Innat",
    author_email="innat.dev@gmail.com",
    url="https://github.com/innat/VideoSwin",
    keywords=["deep learning", "video retrieval", "video recognition"],
    install_requires=[
        "opencv-python>=4.1.2",
        "keras==3.0.5",
        "tensorflow-datasets",
    ],
    extras_require={
        "tests": [
            "flake8",
            "isort",
            "black[jupyter]",
            "pytest",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
)
