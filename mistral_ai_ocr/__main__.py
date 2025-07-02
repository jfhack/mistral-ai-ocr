#!/usr/bin/env python
from pathlib import Path
from . import Modes, construct_from_mode, get_mode_from_string
import argparse
import codecs
from os import getenv
from dotenv import load_dotenv

_default_path_dot_env = Path.home() / ".mistral_ai_ocr.env"

_mode_choices = [mode.name for mode in Modes] + [str(mode.value) for mode in Modes]

def _unescape(s: str) -> str:
  return codecs.decode(s, 'unicode_escape')

def main():
  example_text = (
    'examples:\n\n'
    '%(prog)s paper.pdf\n'
    '%(prog)s paper.pdf --api-key jrWjJE5lFketfB2sA6vvhQK2SoHQ6R39\n'
    '%(prog)s paper.pdf -o revision\n'
    '%(prog)s paper.pdf -e\n'
    '%(prog)s paper.pdf -m FULL\n'
    '%(prog)s -j paper.json\n'
    '%(prog)s -j paper.json -m TEXT_NO_PAGES -n\n'
  )
  parser = argparse.ArgumentParser(
    description="A simple script that uses the Mistral AI OCR API to get the Markdown text from a PDF or image file.",
    epilog=example_text,
    formatter_class=argparse.RawDescriptionHelpFormatter
  )

  parser.add_argument("input", type=Path, nargs="?", help="input PDF or image file", default=None)
  parser.add_argument("-k", "--api-key", help="Mistral API key, can be set via the MISTRAL_API_KEY environment variable", default=None)
  parser.add_argument("-o", "--output", type=Path, help="output directory path. If not set, a directory will be created in the current working directory using the same stem (filename without extension) as the input file", default=None)
  parser.add_argument("-j", "--json-ocr-response", type=Path, help="path from which to load a pre-existing JSON OCR response (any input file will be ignored)", default=None)
  parser.add_argument("-m", "--mode", type=str, choices=_mode_choices, default="FULL_NO_PAGES",
                      help="mode of operation: either the name or numerical value of the mode. Defaults to FULL_NO_PAGES")
  parser.add_argument("-s", "--page-separator", type=str, default="\n",
                      help="page separator to use when writing the Markdown file. Defaults to '\\n'")
  parser.add_argument("-n", "--no-json", action="store_false", dest="write_json",
                      help="do not write the JSON OCR response to a file. By default, the response is written")
  parser.add_argument("-e", "--load-dot-env", action="store_true",
                      help="load the .env file from the current directory using python-dotenv, to retrieve the Mistral API key")
  parser.add_argument("-E", "--load-path-dot-env", type=Path, default=_default_path_dot_env,
                      help="load the .env file from the specified path using python-dotenv, to retrieve the Mistral API key. Defaults to ~/.mistral_ai_ocr.env")
  args = parser.parse_args()

  if args.load_dot_env:
    load_dotenv()
    load_dotenv(".env")

  if args.load_path_dot_env is not None:
    if args.load_path_dot_env.exists():
      load_dotenv(args.load_path_dot_env)

  if args.api_key is None:
    args.api_key = getenv("MISTRAL_API_KEY")
    if args.api_key is None:
      parser.error("API key is required. Set it with --api-key, via the MISTRAL_API_KEY environment variable, or load it from a .env file with -e/--load-dot-env or -E/--load-path-dot-env")

  try:
    construct_from_mode(
      document_path=args.input,
      api_key=args.api_key,
      output_directory=args.output,
      json_ocr_response_path=args.json_ocr_response,
      page_separator=_unescape(args.page_separator),
      write_json=args.write_json,
      mode=get_mode_from_string(args.mode)
    ).process()
  except FileNotFoundError as e:
    parser.error(e)
  except ValueError as e:
    parser.error(e)

if __name__ == "__main__":
  main()
