from setuptools import setup, find_packages

setup(
    name="ecology_summarizer",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "litellm>=1.0.0",
        "instructor>=1.0.0",
        "pymupdf>=1.23.0",
        "faiss-cpu>=1.7.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "langchain-text-splitters>=0.2.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ]
    },
    author="Dor Apffel",
    description="A domain-specific agent for summarizing ecological research papers",
    python_requires=">=3.10",
)
