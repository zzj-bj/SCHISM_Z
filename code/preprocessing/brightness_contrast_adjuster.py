# -*- coding: utf-8 -*-
"""
 This script automatically adjusts the brightness and contrast of 8-bit grayscale images
 in a specified directory by ignoring extreme histogram ends and rescaling
 pixel values to the 0–255 range. It processes each image, applies the
 adjustment, and saves the adjusted images to an output directory.

Author: Ozan.YAZAR
"""

# Standard library
import os
from typing import Tuple, List, Optional

# Third-party
from PIL import Image
from tqdm import tqdm

# Local application imports
from tools.constants import IMAGE_EXTENSIONS, TQDM_NCOLS, DISPLAY_COLORS
from tools import display_color as dc
from tools import utils as ut

# =============================================================================
# pylint: disable=too-few-public-methods


class BrightnessContrastAdjuster:
    """
    This class allows automatic brightness and contrast adjustment for a group of images.
    Z: Core idea: ignore large background and noise with few pixels, compute hmin/hmax from either a
    reference image or per-image histogram, build a linear LUT from that range, apply it to each image,
    and save the adjusted copy with an "_adjusted_" suffix.
    """

    # Factor controlling how aggressively extreme histogram values are ignored
    # Z: total pixel count / AUTO_THRESHOLD = threshold to ignore, > keep, < ignore
    AUTO_THRESHOLD = 5000

    # This range must be a float to ensure float division.
    IMAGE_RANGE = 256.0

    # The max value in an 8-bit image is 255 and min is 0.
    LUT_MIN_VAL = 0
    LUT_MAX_VAL = 255

    # The default hmin/hmax values
    DEFAULT_HMIN = 0
    DEFAULT_HMAX = 255

    # Fraction of total pixels used to define histogram limit
    # Z: used to ignore large background
    LIMIT_FRACTION = 10.0

    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.display = dc.DisplayColor()

    def calculate_hmin_hmax(self, image_path: str) -> Tuple[int, int]:
        """Calculate and return hmin and hmax for the given image.
        Z: will ignore large background and noise with few pixels
        """

        image = Image.open(image_path)
        # Z: convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')

        hist = image.histogram()
        pixel_count = image.width * image.height

        # Ignore very high bins in the histogram (limit = 10% of total pixels)
        # Z: equivalent to ignoring large background
        limit = pixel_count / self.LIMIT_FRACTION

        # Threshold to ignore extreme histogram bins with very few pixels
        # Z: equivalent to ignoring noise with few pixels
        threshold = pixel_count / self.AUTO_THRESHOLD

        hmin, hmax = self.DEFAULT_HMIN, self.DEFAULT_HMAX

        # Find hmin
        for i, count in enumerate(hist):
            if count > limit:
                count = 0
            if count > threshold:
                hmin = i
                break

        # Find hmax
        for i in reversed(range(len(hist))):
            count = hist[i]
            if count > limit:
                count = 0
            if count > threshold:
                hmax = i
                break
        # Z: hmin and hmax are indices in histogram
        return hmin, hmax

    def build_lut(self, hmin: int, hmax: int) -> List[int]:
        """Build lookup table based on the given hmin and hmax."""
        lut = []
        for i in range(self.LUT_MAX_VAL + 1):
            if i < hmin:
                lut.append(self.LUT_MIN_VAL)
            elif i > hmax:
                lut.append(self.LUT_MAX_VAL)
            else:
                # Z: scale linearly the intermediate values between hmin and hmax
                scaled_lut = int(
                    ((i - hmin) * self.IMAGE_RANGE / (hmax - hmin))
                )
                lut.append(min(scaled_lut, self.LUT_MAX_VAL))
        # Z: lut is a list of 256 values mapping input pixel values to adjusted pixel values
        return lut

    def apply_lut(self, image_path: str, lut: List[int]) -> Image.Image:
        """Apply the given precomputed LUT to an image."""

        image = Image.open(image_path)
        if image.mode != 'L':
            image = image.convert('L')

        # Apply the LUT
        # Z: replace each pixel value with corresponding value from LUT
        adjusted_image = image.point(lut)

        return adjusted_image

    def auto_adjust_brightness_contrast(
        self, hmin_hmax_mode: str, ref_img_path: str
    ) -> None:
        """
        Automatically adjust the brightness/contrast of images in the input folder and
        save the results.
        Each image’s pixel values are rescaled based on histogram
        minimum/maximum (hmin/hmax).

        Two modes are supported:
            - "ref_image": Use a single reference image to compute hmin/hmax.
            - "per_image": Compute hmin/hmax separately for each image.
        """

        # Collect all images in the input folder that match valid extensions
        input_image_names = [
            f for f in os.listdir(self.input_path)
            if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
        ]

        # List to store (filename, error_message) for failed images
        errors: List[Tuple[str, str]] = []

        # Progress bar setup
        image_progress = tqdm(
            input_image_names,
            ncols=TQDM_NCOLS,
            bar_format="{desc} {n_fmt}/{total_fmt} |{bar}| "
                       "{percentage:5.1f}%"
        )

        lut = None
        # Precompute LUT (lookup table) if using "ref_image" mode
        if hmin_hmax_mode == "ref_image":
            lut = self._compute_ref_image_lut(ref_img_path)
            if lut is None:
                return

        # Iterate over images in the input folder
        for image_name in image_progress:

            # Short folder path (last two folders) for progress bar display
            short_input_path = os.path.basename(
                os.path.dirname(self.input_path)
            )

            # Update progress bar description
            image_progress.set_description(
                rf"{short_input_path} → {image_name}"
            )

            # Define full input and output paths
            input_image_path = os.path.join(self.input_path, image_name)

            # Z: define output image path with _adjusted_ suffix
            name, extension = os.path.splitext(image_name)
            split_name_parts = ut.split_string(name)
            if split_name_parts:
                new_name = f"{split_name_parts[0]}_adjusted_{split_name_parts[1]}{extension}"
            else:
                new_name = image_name
            output_image_path = os.path.join(self.output_path, new_name)


            # If "per_image" mode → compute LUT separately for each image
            if hmin_hmax_mode == "per_image":
                lut = self._compute_per_image_lut(input_image_path, errors)
                if lut is None:
                    continue

            try:
                if lut is None:
                    errors.append((image_name, "LUT is None, cannot adjust image"))
                    continue
                adjusted_image = self.apply_lut(input_image_path, lut)
                adjusted_image.save(output_image_path)
            except (IOError, ValueError) as e:
                errors.append((image_name, str(e)))

        if errors:
            # Summary of failed images
            self.display.print(
                f"{len(errors)} image(s) failed to adjust. "
                "See details below:",
                DISPLAY_COLORS["error"]
            )

            # List each failed image and reason
            for fname, reason in errors:
                self.display.print(
                    f"- {fname}: {reason}", DISPLAY_COLORS["error"]
                )
        else:
            # Success message
            self.display.print(
                "All images in the folder have been successfully adjusted",
                DISPLAY_COLORS["ok"]
            )

    def _compute_ref_image_lut(
        self, ref_img_path: str
    ) -> Optional[List[int]]:
        """Compute the LUT (Look-Up Table) based on a reference image."""

        try:
            # Calculate histogram min/max from the reference image
            hmin, hmax = self.calculate_hmin_hmax(ref_img_path)
            if hmax <= hmin:  # Sanity check
                self.display.print(
                    "Failed to adjust brightness/contrast for the image sequence. "
                    "See details below:",
                    DISPLAY_COLORS["error"]
                )
                self.display.print(
                    f"- {os.path.basename(ref_img_path)}: Invalid "
                    "brightness/contrast range (hmin/hmax) for "
                    "the reference image",
                    DISPLAY_COLORS["error"]
                )
                return None

            return self.build_lut(hmin, hmax)

        except Exception as e:
            self.display.print(
                "Failed to compute LUT from the reference image. "
                "See details below:",
                DISPLAY_COLORS["error"]
            )
            self.display.print(
                f"- {os.path.basename(ref_img_path)}: {e}", DISPLAY_COLORS["error"]
            )
            return None

    def _compute_per_image_lut(
        self, input_image_path: str, errors: List[Tuple[str, str]]
    ) -> Optional[List[int]]:
        """Compute the LUT (Look-Up Table) for a single image."""

        try:
            hmin, hmax = self.calculate_hmin_hmax(input_image_path)
            if hmax <= hmin:  # Sanity check
                errors.append(
                    (
                        input_image_path,
                        "Invalid brightness/contrast range "
                        "(hmin/hmax) for the image"
                    )
                )
                return None

            return self.build_lut(hmin, hmax)

        except Exception as e:
            errors.append((input_image_path, str(e)))
            return None
