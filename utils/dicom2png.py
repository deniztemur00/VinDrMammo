import matplotlib.pyplot as plt
import pydicom
import numpy as np
from pathlib import Path
from PIL import Image


class DICOMConverter:
    def convert_to_png(
        self,
        dicom_path: Path,
        png_path: Path,
        delete_dicom: bool = False,
        voi_lut: bool = True,
    ) -> Path:
        """Converts a DICOM file to a PNG image with optional VOI LUT, windowing, and rescaling.

        Args:
            dicom_relative_path (str): The relative path to the DICOM file.
            delete_dicom (bool): Whether to delete the original DICOM file after conversion.
            voi_lut (bool): Whether to apply the Value of Interest (VOI) Lookup Table.
        """

        if not dicom_path.exists():
            print(f"Missing DICOM file: {dicom_path}")
            return None

        try:
            dicom_data = pydicom.read_file(dicom_path)
            pixel_array = dicom_data.pixel_array

            if voi_lut:
                pixel_array = self.apply_voi_lut(pixel_array, dicom_data)

            if (
                "PhotometricInterpretation" in dicom_data
                and dicom_data.PhotometricInterpretation == "MONOCHROME1"
            ):
                pixel_array = np.amax(pixel_array) - pixel_array

            pixel_array = self.normalize_pixel_values(pixel_array)

            # if 'RescaleSlope' in dicom_data and 'RescaleIntercept' in dicom_data:
            #     pixel_array = self.apply_rescale(pixel_array, float(dicom_data.RescaleSlope), float(dicom_data.RescaleIntercept))

            # if 'WindowCenter' in dicom_data and 'WindowWidth' in dicom_data:
            #     window_center = self.get_first_value(dicom_data.WindowCenter)
            #     window_width = self.get_first_value(dicom_data.WindowWidth)
            #     pixel_array = self.apply_windowing(pixel_array, window_center, window_width)

            # pixel_array = self.apply_padding(pixel_array, dicom_data)

            image = Image.fromarray(pixel_array.astype(np.uint8))
            image.save(str(png_path))

            if delete_dicom:
                dicom_path.unlink()

            return png_path
        except Exception as e:
            print(f"Error converting {dicom_path}: {str(e)}")
            return None

    def apply_voi_lut(self, pixel_array, dicom_data) -> np.ndarray:
        """Apply Value of Interest (VOI) Lookup Table if available."""
        try:
            pixel_array = pydicom.pixel_data_handlers.util.apply_voi_lut(
                pixel_array, dicom_data
            )
        except Exception as e:
            print(f"VOI LUT failed: {e}, defaulting to raw pixel data.")
        return pixel_array

    def normalize_pixel_values(self, pixel_array) -> np.ndarray:
        """Normalize pixel values to the 0-255 range."""
        pixel_array = pixel_array - np.min(pixel_array)
        max_val = np.max(pixel_array)
        if max_val > 0:
            pixel_array = (pixel_array / max_val) * 255
        return pixel_array

    def apply_rescale(self, pixel_array, slope, intercept):
        """Apply rescale operation based on DICOM headers."""
        return pixel_array * slope + intercept

    def apply_windowing(self, pixel_array, center, width):
        """Apply window leveling to enhance contrast."""
        lower_bound = center - width / 2
        upper_bound = center + width / 2
        return np.clip(
            (pixel_array - lower_bound) / (upper_bound - lower_bound) * 255, 0, 255
        )

    def apply_padding(self, pixel_array, data):
        """Handle padding values, setting them to 0."""
        try:
            pad_val = data.get("PixelPaddingValue", 0)
            pad_limit = data.get("PixelPaddingRangeLimit", -99999)
            if pad_limit == -99999:
                mask_pad = pixel_array == pad_val
            else:
                photometricInterpretation = data.PhotometricInterpretation
                if photometricInterpretation == "MONOCHROME2":
                    mask_pad = (pixel_array >= pad_val) & (pixel_array <= pad_limit)
                else:
                    mask_pad = (pixel_array >= pad_limit) & (pixel_array <= pad_val)
            pixel_array[mask_pad] = 0
        except Exception:
            pixel_array = pixel_array.astype(np.int)
            pixels, counts = np.unique(pixel_array, return_counts=True)
            most_common = np.argsort(counts)[-2:]
            if counts[most_common[0]] > counts[most_common[1]] * 10:
                pixel_array[pixel_array == pixels[most_common[0]]] = 0
        return pixel_array

    def get_first_value(self, dicom_attribute):
        """Extract the first value from a MultiValue DICOM attribute."""
        if isinstance(dicom_attribute, pydicom.multival.MultiValue):
            return dicom_attribute[0]


def convert_dicom_to_png(dicom_file: str) -> np.ndarray:
    """
    dicom_file: path to the dicom fife

    return
        gray scale image with pixel intensity in the range [0,255]
        None if cannot convert

    """
    data = pydicom.read_file(dicom_file)
    if (
        ("WindowCenter" not in data)
        or ("WindowWidth" not in data)
        or ("PhotometricInterpretation" not in data)
        or ("RescaleSlope" not in data)
        or ("PresentationIntentType" not in data)
        or ("RescaleIntercept" not in data)
    ):

        print(f"{dicom_file} DICOM file does not have required fields")
        return

    intentType = data.data_element("PresentationIntentType").value
    if str(intentType).split(" ")[-1] == "PROCESSING":
        print(f"{dicom_file} got processing file")
        return

    c = data.data_element("WindowCenter").value  # data[0x0028, 0x1050].value
    w = data.data_element("WindowWidth").value  # data[0x0028, 0x1051].value
    if type(c) == pydicom.multival.MultiValue:
        c = c[0]
        w = w[0]

    photometricInterpretation = data.data_element("PhotometricInterpretation").value

    try:
        a = data.pixel_array
    except:
        print(f"{dicom_file} Cannot get get pixel_array!")
        return

    slope = data.data_element("RescaleSlope").value
    intercept = data.data_element("RescaleIntercept").value
    a = a * slope + intercept

    try:
        pad_val = data.get("PixelPaddingValue")
        pad_limit = data.get("PixelPaddingRangeLimit", -99999)
        if pad_limit == -99999:
            mask_pad = a == pad_val
        else:
            if str(photometricInterpretation) == "MONOCHROME2":
                mask_pad = (a >= pad_val) & (a <= pad_limit)
            else:
                mask_pad = (a >= pad_limit) & (a <= pad_val)
    except:
        # Manually create padding mask
        # this is based on the assumption that padding values take majority of the histogram
        print(f"{dicom_file} has no PixelPaddingValue")
        a = a.astype(np.int)
        pixels, pixel_counts = np.unique(a, return_counts=True)
        sorted_idxs = np.argsort(pixel_counts)[::-1]
        sorted_pixel_counts = pixel_counts[sorted_idxs]
        sorted_pixels = pixels[sorted_idxs]
        mask_pad = a == sorted_pixels[0]
        try:
            # if the second most frequent value (if any) is significantly more frequent than the third then
            # it is also considered padding value
            if sorted_pixel_counts[1] > sorted_pixel_counts[2] * 10:
                mask_pad = np.logical_or(mask_pad, a == sorted_pixels[1])
                print(
                    f"{dicom_file} most frequent pixel values: {sorted_pixels[0]}; {sorted_pixels[1]}"
                )
        except:
            print(f"{dicom_file} most frequent pixel value {sorted_pixels[0]}")

    # apply window
    mm = c - 0.5 - (w - 1) / 2
    MM = c - 0.5 + (w - 1) / 2
    a[a < mm] = 0
    a[a > MM] = 255
    mask = (a >= mm) & (a <= MM)
    a[mask] = ((a[mask] - (c - 0.5)) / (w - 1) + 0.5) * 255

    if str(photometricInterpretation) == "MONOCHROME1":
        a = 255 - a

    a[mask_pad] = 0
    return a


if __name__ == "__main__":

    dicom_path = "VinDr-Mammo/images/0a0c5108270e814818c1ad002482ce74/0a6a90bdc088e0cc62df8d2d58d14840.dicom"
    png_img = convert_dicom_to_png(dicom_path)
    plt.imshow(png_img, cmap="gray")
    plt.show()
