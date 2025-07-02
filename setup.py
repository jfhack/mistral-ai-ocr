#!/usr/bin/env python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setup(
  name="mistral-ai-ocr",
  version="1.4",
  packages=find_packages(),
  entry_points={
    'console_scripts': [
      'mistral-ai-ocr = mistral_ai_ocr.__main__:main',
    ],
  },
  url="https://github.com/jfhack/mistral-ai-ocr",
  install_requires=[
    'mistralai',
    'python-dotenv'
  ],
  long_description=long_description,
  long_description_content_type="text/markdown"
)
