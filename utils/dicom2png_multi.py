from dicom2png import DICOMConverter
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser
import time

class DicomToPngConverter:
    def __init__(self, input_dir, output_dir, delete_dicom=False, max_workers=16, file_pattern="**/*.dicom"):
        """
        Initialize the DICOM to PNG converter.
        
        Args:
            input_dir (str or Path): Directory containing DICOM files
            output_dir (str or Path): Directory where PNG files will be saved
            delete_dicom (bool): Whether to delete DICOM files after conversion
            max_workers (int): Maximum number of parallel processes
            file_pattern (str): Glob pattern to search for DICOM files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.delete_dicom = delete_dicom
        self.max_workers = max_workers
        self.file_pattern = file_pattern
        
    @staticmethod
    def _convert_single_entry(dicom_filepath, output_dir, delete_dicom):
        """
        Convert a single DICOM file to PNG.
        
        Args:
            dicom_filepath (Path): Path to DICOM file
            output_dir (Path): Output directory
            delete_dicom (bool): Whether to delete the DICOM file after conversion
            
        Returns:
            Path: Path to the created PNG file or None if conversion failed
        """
        try:
            # Create the corresponding output path
            relative_path = dicom_filepath.relative_to(dicom_filepath.parents[1])
            png_path = output_dir / relative_path.with_suffix(".png")
            png_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert the DICOM file to PNG
            converter = DICOMConverter()  # Create a new converter for each process
            converter.convert_to_png(dicom_filepath, png_path, delete_dicom)
            return png_path
        except Exception as e:
            print(f"Failed to convert {dicom_filepath}: {e}")
            return None
            
    def convert(self):
        """
        Convert all DICOM files to PNG using multiprocessing.
        
        Returns:
            tuple: (success_count, total_count, execution_time_seconds)
        """
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all DICOM files
        dicom_filepaths = list(self.input_dir.glob(self.file_pattern))
        
        if not dicom_filepaths:
            print(f"No DICOM files found in {self.input_dir} using pattern '{self.file_pattern}'")
            return 0, 0, 0
            
        print(f"Found {len(dicom_filepaths)} DICOM files to convert")
        start_time = time.time()
        
        successful_conversions = 0
        
        # Create a ProcessPoolExecutor to manage multiprocessing
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks to the executor
            futures = [
                executor.submit(
                    self._convert_single_entry,
                    dicom_filepath,
                    self.output_dir,
                    self.delete_dicom,
                )
                for dicom_filepath in dicom_filepaths
            ]
            
            # Process results as they complete
            for future in tqdm(
                as_completed(futures), 
                total=len(futures), 
                desc="Converting DICOMs to PNGs"
            ):
                if future.result() is not None:
                    successful_conversions += 1
        
        execution_time = time.time() - start_time
        print(f"Conversion completed: {successful_conversions}/{len(dicom_filepaths)} files in {execution_time:.2f} seconds.")
        
        return successful_conversions, len(dicom_filepaths), execution_time


def main():
    """
    Command line interface for the DICOM to PNG converter.
    """
    parser = ArgumentParser(description="Convert DICOM files to PNG using multiprocessing.")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Base path where DICOM files are stored.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Base path where PNG files will be saved.",
    )
    parser.add_argument(
        "--delete_dicom",
        action="store_true",
        help="Delete DICOM files after conversion. Defaults to False.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Maximum number of parallel worker processes. Default: 16",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.dicom",
        help="File pattern to search for DICOM files. Default: **/*.dicom",
    )

    args = parser.parse_args()

    if args.delete_dicom:
        input("WARNING: You are about to delete DICOM files. Press Enter to continue or Ctrl+C to exit.")

    converter = DicomToPngConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        delete_dicom=args.delete_dicom,
        max_workers=args.max_workers,
        file_pattern=args.pattern
    )
    
    converter.convert()


if __name__ == "__main__":
    main()