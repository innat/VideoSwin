from setuptools import find_packages, setup

setup(
    name="videoswin",
    packages=find_packages(exclude=["notebooks", "assets"]),
    version="1.0.0",
    license="MIT",
    description="Video Swin Transformerss in Keras",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mohammed Innat",
    author_email="innat.dev@gmail.com",
    url="https://github.com/innat/VideoSwin",
    keywords=["deep learning", "image retrieval", "image recognition"],
    install_requires=[
        "opencv-python>=4.1.2",
        "keras-nightly",
    ],
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=["pytest"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)