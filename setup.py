from setuptools import find_namespace_packages, setup

setup(
    name="EditSSC",
    version="0.1.0",
    packages=find_namespace_packages(
        include=[
            "dataset*",
            "diffusion*",
            "encoding*",
            "generation*",
            "utils*",
            "script*",
            "configs*",
        ]
    ),
)