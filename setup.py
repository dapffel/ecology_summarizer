from setuptools import setup, find_packages

setup(
    name="ecology_summarizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai",
        "faiss-cpu",
        "numpy",
        "python-dotenv",
        "tenacity",
        "PyPDF2",
        "pydantic"
    ],
    author="Your Name",
    description="A domain-specific agent for summarizing ecological research papers using GPT",
    include_package_data=True,
    python_requires='>=3.8',
)