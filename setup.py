from setuptools import setup, find_packages

setup(
    name="ecology_summarizer",
    version="0.1.5",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "faiss-cpu>=1.7.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "tenacity>=8.0.0",
        "PyPDF2>=3.0.0",
        "pydantic>=2.0.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "python-magic>=0.4.27",
        "aiofiles>=23.1.0",
        "typing-extensions>=4.5.0",
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
    description="A domain-specific agent for summarizing ecological research papers using GPT",
    include_package_data=True,
    python_requires=">=3.8",
)