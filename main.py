from grid_maker.grid_maker import GridMaker
import argparse

import os

import os

from grid_maker.image_reader import read_image

from grid_maker.grid_maker import GridMaker
import argparse
import os
import cv2

from grid_solver.word_finder import load_words, WordFinder

def edit_grid(grid):
    while True:
        print("\nCurrent grid:")
        for r, row in enumerate(grid):
            print(r, " ".join(c if c else "." for c in row))

        cmd = input("\nEdit? (format: row col letter) or ENTER to continue: ").strip()

        if cmd == "":
            break

        try:
            r, c, val = cmd.split()
            r, c = int(r), int(c)
            val = val.upper()

            if val == ".":
                grid[r][c] = None
            elif len(val) == 1 and val.isalpha():
                grid[r][c] = val
            else:
                print("Invalid letter")

        except Exception:
            print("Invalid input. Use: row col letter (e.g. 1 2 A)")

def resolve_image_name(name: str) -> str:
    if os.path.isfile(name):
        return name

    if "." not in name:
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join("pictures", name + ext)
            if os.path.isfile(candidate):
                return candidate

    raise FileNotFoundError(
        f"Image '{name}' not found in current directory or pictures/"
    )


def main(path:str = None):
    if path is None:
        parser = argparse.ArgumentParser(description="Process an image and extract grid.")
        parser.add_argument("path", type=str, help="Path or image name")

        args = parser.parse_args()
        path = args.path
    path = resolve_image_name(path)
    print(path)
    img = read_image(path)
    gm = GridMaker()
    grid = gm.get_grid(img)

    edit_grid(grid)
    for row in grid:
        print(" ".join(c if c else "." for c in row))

    words = load_words()
    finder = WordFinder(grid, words)
    found = finder.find_words()

    print(sorted(found))
if __name__ == '__main__':
    main("16-05-2026")