import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nerf_workspace-panagiotis paschalidis",
    version="0.0.1",
    author="Paschalidis Panagioits, Despoina Paschalidou",
    author_email="paschalidispanagiotis.haf@gmail.com",
    description="neural radiance fields implementations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/panospaschalidis/nerf_workspace",
    project_urls={
        "nerf_workspace": "https://github.com/panospaschalidis/nerf_workspace",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "nerf_workspace"},
    packages=setuptools.find_packages(where="nerf_workspace"),
    # python_requires=">=3.6",
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "matplotlib",
        "simple-3dviz",
        "wandb",
    ]
)

