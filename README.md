Here’s an updated version of your README with **logos** added to make it visually appealing and align with the tools used (OpenCV and NumPy):

---

# Image Filter Application

![OpenCV Logo](https://opencv.org/assets/images/opencv-logo-small.png) ![NumPy Logo](https://numpy.org/images/logo.svg)

This Python script allows you to apply various filters (e.g., sharpen, invert) to images in a specified directory and save the filtered images to an output directory. It is a simple yet powerful tool for batch-processing images with custom filters.

---

## Features
- **Batch Processing**: Apply filters to all images in a directory at once.
- **Custom Filters**: Includes built-in filters like sharpen and invert, with the ability to add more.
- **User-Friendly**: Prompts for source and destination directories at runtime.
- **Error Handling**: Checks for valid directories and skips unreadable images.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ImageFilterApp.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ImageFilterApp
   ```
3. Install the required dependencies:
   ```bash
   pip install opencv-python numpy
   ```

---

## Usage
1. Run the script:
   ```bash
   python image_filter.py
   ```
2. Enter the path to the source directory containing your images.
3. Enter the path to the destination directory where filtered images will be saved.
4. The script will process all `.png`, `.jpg`, and `.jpeg` files in the source directory and save the filtered images in subfolders within the destination directory.

---

## Example
### Input Directory Structure
```
source/
├── image1.jpg
├── image2.png
```

### Output Directory Structure
```
destination/
├── image1/
│   ├── sharpen.png
│   ├── invert.png
├── image2/
│   ├── sharpen.png
│   ├── invert.png
```

---

## Available Filters
- **Sharpen**: Enhances the edges of the image.
- **Invert**: Inverts the colors of the image.

You can easily add more filters by defining new functions and adding them to the `filter_names` and `filters` lists in the `apply_filters` function.

---

## Contributing
Contributions are welcome! If you'd like to add new filters or improve the code, feel free to open a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Built with [OpenCV](https://opencv.org/) and [NumPy](https://numpy.org/).
- Inspired by the need for simple batch image processing tools.

---

Enjoy filtering your images! 🎨

---

### Logos Used:
1. **OpenCV Logo**: Represents the core library used for image processing.
2. **NumPy Logo**: Represents the library used for numerical operations and kernel-based filtering.
