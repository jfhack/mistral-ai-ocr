# Mistral AI OCR
This is a simple script that uses the Mistral AI OCR API to get the Markdown text from a PDF or image file

# Usage

## Install the Requirements

To install the necessary requirements, run the following command:

```sh
pip install mistral-ai-ocr
```

## Typical Usage

```sh
mistral-ai-ocr paper.pdf
mistral-ai-ocr paper.pdf --api-key jrWjJE5lFketfB2sA6vvhQK2SoHQ6R39
mistral-ai-ocr paper.pdf -o revision
mistral-ai-ocr paper.pdf -e
mistral-ai-ocr paper.pdf -m FULL
mistral-ai-ocr page74.jpg -e
mistral-ai-ocr -j paper.json
mistral-ai-ocr -j paper.json -m TEXT_NO_PAGES -n
```

## Arguments

| Argument || Description |
|-|-|-|
| | | input PDF or image file |
| -k API_KEY | --api-key API_KEY | Mistral API key, can be set via the **MISTRAL_API_KEY** environment variable |
| -o OUTPUT | --output OUTPUT | output directory path. If not set, a directory will be created in the current working directory using the same stem (filename without extension) as the input file |
| -j JSON_OCR_RESPONSE | --json-ocr-response JSON_OCR_RESPONSE | path from which to load a pre-existing JSON OCR response (any input file will be ignored) |
| -m MODE | --mode MODE | mode of operation: either the name or numerical value of the mode. _Defaults to FULL_NO_PAGES_ |
| -s PAGE_SEPARATOR | --page-separator PAGE_SEPARATOR | page separator to use when writing the Markdown file. _Defaults to `\n`_ |
| -n | --no-json | do not write the JSON OCR response to a file. By default, the response is written |
| -e | --load-dot-env | load the .env file from the current directory using [`python-dotenv`](https://pypi.org/project/python-dotenv/), to retrieve the Mistral API key |
| -E | --load-path-dot-env | load the .env file from the specified path using [`python-dotenv`](https://pypi.org/project/python-dotenv/), to retrieve the Mistral API key. Defaults to ~/.mistral_ai_ocr.env |

## Modes

| Value | Name |
|-|-|
| 0 | FULL |
| 1 | FULL_ALT |
| 2 | FULL_NO_DIR |
| 3 | FULL_NO_PAGES |
| 4 | TEXT |
| 5 | TEXT_NO_PAGES |

Given the input file `paper.pdf`, the directory structure for each mode is shown below:

### 0 - `FULL`

Structure
```
paper
├── full
│   ├── image1.png
│   ├── image2.png
│   ├── image3.png
│   └── paper.md
├── page_0
│   ├── image1.png
│   └── paper.md
├── page_1
│   ├── image2.png
│   └── paper.md
└── page_2
    ├── image3.png
    └── paper.md
```

### 1 - `FULL_ALT`

Structure
```
paper
├── image1.png
├── image2.png
├── image3.png
├── paper.md
├── page_0
│   ├── image1.png
│   └── paper.md
├── page_1
│   ├── image2.png
│   └── paper.md
└── page_2
    ├── image3.png
    └── paper.md
```

### 2 - `FULL_NO_DIR`

Structure
```
paper
├── image1.png
├── image2.png
├── image3.png
├── paper.md
├── paper0.md
├── paper1.md
└── paper2.md
```

### 3 - `FULL_NO_PAGES` *default*

Structure
```
paper
├── image1.png
├── image2.png
├── image3.png
└── paper.md
```

### 4 - `TEXT`

Structure
```
paper
├── paper.md
├── paper0.md
├── paper1.md
└── paper2.md
```

### 5 - `TEXT_NO_PAGES`

Structure
```
paper
└── paper.md
```

By default, the JSON response from the Mistral AI OCR API is saved in the output directory. To disable JSON output, use the `-n` or `--no-json` argument. To experiment with a different **mode** without using additional API calls, reuse an existing JSON response instead of the original input file

### Mistral AI API Key

To obtain an API key, you need a [Mistral AI](https://auth.mistral.ai/ui/registration) account. Then visit [https://admin.mistral.ai/organization/api-keys](https://admin.mistral.ai/organization/api-keys) and click the **Create new key** button

To avoid using `-e` to load the `.env` file, you can create one at `$HOME/.mistral_ai_ocr.env` (where `$HOME` is your home directory). It will then be automatically loaded at the start of the script

For example, for an user called `vavilov`, the path would look like this:

* **Linux**
  ```
  /home/vavilov/.mistral_ai_ocr.env  
  ```

* **macOS**
  ```
  /Users/vavilov/.mistral_ai_ocr.env  
  ```

* **Windows**
  ```
  C:\Users\vavilov\.mistral_ai_ocr.env  
  ```

and the content will be something like this:

```
MISTRAL_API_KEY=jrWjJE5lFketfB2sA6vvhQK2SoHQ6R39
```
