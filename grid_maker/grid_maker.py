import os

import cv2
import numpy as np
import pytesseract
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt


class GridMaker:
    def __init__(self, template_folder="letters"):
        self.templates = self._load_templates(template_folder)

    # -------------------------------------------------------------------------
    # PUBLIC
    # -------------------------------------------------------------------------

    def get_grid(self, img):
        gray = self._to_gray(img)
        binary = self._binarize(gray)

        tiles = self._detect_tiles(binary, gray)
        if not tiles:
            raise ValueError("No tiles detected — check image quality or grid structure.")

        self._debug_draw_tiles(img, tiles)

        rows = self._group_into_rows(tiles)
        cells = self._extract_cells(gray, rows)

        self._debug_show_all_cells(cells)

        return self._build_grid(cells)

    # -------------------------------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------------------------------

    def _to_gray(self, img):
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _binarize(self, gray):
        """Adaptive threshold handles uneven lighting better than global Otsu."""
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=4,
        )
        return binary

    # -------------------------------------------------------------------------
    # TILE DETECTION — morphological grid-line extraction
    # -------------------------------------------------------------------------

    def _detect_tiles(self, binary, gray):
        """
        Extract individual grid cells using morphological line detection.

        Strategy:
          1. Isolate horizontal lines with a wide horizontal kernel.
          2. Isolate vertical lines with a tall vertical kernel.
          3. Combine into a clean grid skeleton.
          4. Invert the skeleton so cell interiors become solid blobs.
          5. Find contours of those blobs — each one is a cell.

        This is fundamentally more reliable than Canny + RETR_EXTERNAL, which
        only finds the outer boundary of the whole grid.
        """
        h, w = binary.shape

        # --- horizontal lines ---
        h_len = max(w // 20, 20)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)

        # --- vertical lines ---
        v_len = max(h // 20, 20)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

        # --- combine and close tiny gaps at intersections ---
        grid_mask = cv2.add(horizontal, vertical)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grid_mask = cv2.dilate(grid_mask, close_kernel, iterations=1)

        # --- cell interiors are the holes in the grid mask ---
        cell_mask = cv2.bitwise_not(grid_mask)
        contours, _ = cv2.findContours(
            cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        tiles = self._filter_tile_contours(contours)

        # If morphological approach found too little, fall back
        if len(tiles) < 4:
            print("Morphological detection found too few tiles — trying fallback.")
            tiles = self._fallback_detect_tiles(binary)

        print(f"Detected {len(tiles)} tiles.")
        return sorted(tiles, key=lambda t: (t[1], t[0]))

    def _filter_tile_contours(self, contours):
        """
        Keep only square-ish, consistently-sized blobs that look like grid cells.
        Uses median tile size as the reference so one oddly-sized element can't
        skew the filter.
        """
        if not contours:
            return []

        candidates = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 10 or h < 10:
                continue
            aspect = w / float(h)
            if 0.5 < aspect < 2.0:
                candidates.append((x, y, w, h))

        if not candidates:
            return []

        median_w = np.median([w for _, _, w, _ in candidates])
        median_h = np.median([h for _, _, _, h in candidates])

        return [
            (x, y, w, h) for x, y, w, h in candidates
            if 0.5 * median_w < w < 1.5 * median_w
            and 0.5 * median_h < h < 1.5 * median_h
        ]

    def _fallback_detect_tiles(self, binary):
        """
        Fallback for gridless layouts: find all contours in the binary image
        and filter by size consistency.
        """
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return self._filter_tile_contours(contours)

    # -------------------------------------------------------------------------
    # ROW GROUPING — proportional Y-centroid threshold
    # -------------------------------------------------------------------------

    def _group_into_rows(self, tiles):
        """
        Group tiles into rows by comparing Y centroids.
        Threshold is 40% of the median tile height — scales automatically
        with tile size and survives mild perspective distortion.
        """
        if not tiles:
            return []

        median_h = np.median([h for _, _, _, h in tiles])
        y_threshold = median_h * 0.4

        rows = []
        current_row = []
        last_y_center = None

        for tile in tiles:
            x, y, w, h = tile
            y_center = y + h / 2

            if last_y_center is None or abs(y_center - last_y_center) < y_threshold:
                current_row.append(tile)
            else:
                rows.append(sorted(current_row, key=lambda t: t[0]))
                current_row = [tile]

            last_y_center = y_center

        if current_row:
            rows.append(sorted(current_row, key=lambda t: t[0]))

        return rows

    # -------------------------------------------------------------------------
    # CELL EXTRACTION
    # -------------------------------------------------------------------------

    def _extract_cells(self, gray, rows):
        cells = []
        for row in rows:
            cell_row = []
            for x, y, w, h in row:
                # Trim ~10% on each side to remove grid-line pixels
                pad_x = max(int(w * 0.10), 2)
                pad_y = max(int(h * 0.10), 2)

                x1 = max(x + pad_x, 0)
                y1 = max(y + pad_y, 0)
                x2 = min(x + w - pad_x, gray.shape[1])
                y2 = min(y + h - pad_y, gray.shape[0])

                cell = gray[y1:y2, x1:x2]
                cell_row.append(cell)
            cells.append(cell_row)
        return cells

    # -------------------------------------------------------------------------
    # GRID BUILDING
    # -------------------------------------------------------------------------

    def _build_grid(self, cells):
        grid = []
        for row in cells:
            grid_row = []
            for cell in row:
                if self._is_empty(cell):
                    grid_row.append(None)
                else:
                    grid_row.append(self._recognize(cell))
            grid.append(grid_row)
        return grid

    # -------------------------------------------------------------------------
    # EMPTY CHECK — std-dev based
    # -------------------------------------------------------------------------

    def _is_empty(self, cell):
        """
        A blank cell is visually uniform → low standard deviation.
        A cell containing a letter has dark strokes on a light background → high std dev.
        This is far more robust than a fixed pixel-ratio threshold.
        """
        if cell is None or cell.size == 0:
            return True
        return float(np.std(cell)) < 10.0

    # -------------------------------------------------------------------------
    # OCR — Tesseract PSM 10 primary, template matching fallback
    # -------------------------------------------------------------------------

    def _recognize(self, cell):
        """
        Two-stage recognition:
          1. Tesseract --psm 10 (single-character mode) with an A-Z whitelist.
             Purpose-built for exactly this use case — vastly outperforms
             general-purpose OCR engines (EasyOCR) on isolated grid letters.
          2. Template matching fallback for cases Tesseract misses.
        """
        prepped = self._prep_for_ocr(cell)

        letter = self._tesseract_recognize(prepped)
        if letter:
            return letter

        if self.templates:
            letter = self._template_recognize(prepped)
            if letter:
                return letter

        return None

    def _prep_for_ocr(self, cell, target_size=128):
        """
        Resize to a fixed square, binarize with Otsu, then add white border padding.
        Tesseract needs clean, high-contrast input and dislikes content flush to edges.
        """
        cell = cv2.resize(cell, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cell = cv2.copyMakeBorder(cell, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        return cell

    def _tesseract_recognize(self, cell):
        config = (
            "--psm 10 "           # single character
            "--oem 3 "            # LSTM engine
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

        # First attempt: normal image
        raw = pytesseract.image_to_string(cell, config=config).strip().upper()
        if len(raw) == 1 and raw.isalpha():
            print(f"Tesseract: {raw}")
            return raw

        # Second attempt: inverted (dark bg sometimes produces better results)
        raw = pytesseract.image_to_string(
            cv2.bitwise_not(cell), config=config
        ).strip().upper()
        if len(raw) == 1 and raw.isalpha():
            print(f"Tesseract (inv): {raw}")
            return raw

        return None

    def _template_recognize(self, cell, size=(128, 128)):
        """
        Normalised cross-correlation against pre-loaded letter templates.
        Uses result.max() instead of result[0][0] to handle any kernel size.
        Only accepts matches above a confidence threshold of 0.4.
        """
        cell = cv2.resize(cell, size)
        _, cell = cv2.threshold(
            cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        best_char = None
        best_score = -1

        for char, template in self.templates.items():
            result = cv2.matchTemplate(cell, template, cv2.TM_CCOEFF_NORMED)
            score = float(result.max())
            if score > best_score:
                best_score = score
                best_char = char

        if best_score > 0.4:
            print(f"Template: {best_char} (score={best_score:.3f})")
            return best_char

        return None

    # -------------------------------------------------------------------------
    # TEMPLATE LOADING
    # -------------------------------------------------------------------------

    def _load_templates(self, folder="letters", size=(128, 128)):
        templates = {}
        if not os.path.isdir(folder):
            print(f"Template folder '{folder}' not found — template matching disabled.")
            return templates

        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            path = os.path.join(folder, f"{char}.png")
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Crop tight to the letter pixels before resizing
            coords = cv2.findNonZero(img)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                img = img[y:y + h, x:x + w]

            img = cv2.resize(img, size)
            _, img = cv2.threshold(
                img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            templates[char] = img

        print(f"Loaded {len(templates)} letter templates.")
        return templates

    # -------------------------------------------------------------------------
    # DEBUG
    # -------------------------------------------------------------------------

    def _debug_draw_tiles(self, img, tiles):
        debug = (
            img.copy() if len(img.shape) == 3
            else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        )
        for x, y, w, h in tiles:
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Tiles ({len(tiles)})")
        plt.axis('off')
        plt.show()

    def _debug_show_all_cells(self, cells):
        if not cells or not cells[0]:
            return

        cell_size = 64
        rows_vis = []

        for r, row in enumerate(cells):
            row_imgs = []
            for c, cell in enumerate(row):
                cell_resized = cv2.resize(cell, (cell_size, cell_size))
                cell_vis = cv2.cvtColor(cell_resized, cv2.COLOR_GRAY2BGR)
                cv2.putText(
                    cell_vis, f"{r},{c}", (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1
                )
                row_imgs.append(cell_vis)
            rows_vis.append(np.hstack(row_imgs))

        grid_img = np.vstack(rows_vis)
        plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
        plt.title("All Cells")
        plt.axis('off')
        plt.show()