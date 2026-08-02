Bonus Exercise: HDR from JPG using Numpy + Camera Response Curve Estimation
-------------------------------------------------------------------------------

Author: Saranya Bhattacharjee
Course: Computer Vision Project - Exercise 4 (Bonus)
File: hdr_from_jpg_bonus.py

--------------------------------------------------------------------------------
HOW TO RUN

1. Place the script (hdr_from_jpg_bonus.py) in your exercise4/exercise4/ directory.
2. Make sure the bonus data is in:
       ../ex4_additional_exercise_data/ex4_additional_exercise_data/
   with 12 JPG images.
3. Double-check that the order of images matches the order in the exposure_times array.
   (The script prints out the filenames and their order—check this the first time.)
4. Run:
       python hdr_from_jpg_bonus.py
5. After it runs, you’ll find the result saved as:
       ../Exercise 4\exercise4\bonus_result\hdr_jpg_bonus_result.png

--------------------------------------------------------------------------------
WHAT THIS CODE DOES

- Loads the provided 12 JPG images at different exposures.
- Estimates the camera’s response curve (CRF) using the Debevec & Malik method (slide 44 from lecture).
- Linearizes all input images using the estimated CRF and the given exposure times.
- Merges them into an HDR image using a weighted average (log domain).
- Tone maps the HDR to a displayable PNG using a simple log compression.

Everything is implemented using only numpy (plus PIL for image I/O), as per assignment instructions.

--------------------------------------------------------------------------------
NOTES / LIMITATIONS

- The exposure_times array **must** match the order the files are loaded (alphabetically by default).
- The code samples random (non-saturated) pixels for CRF estimation. For my image set, I used 30 samples—this is a safe value. If you run into errors about “too few valid pixels”, lower num_samples, or relax the sampling criteria.
- This method is not meant to produce perfect “Photoshop” HDR images. It demonstrates the principle and pipeline only!
- The tone mapping is basic (logarithmic). Real HDR workflows may use more advanced operators for better visualization.
- JPG input means colors may not be linear—ideally, HDR is done from RAW data, but we use JPGs here because that’s what’s provided.

--------------------------------------------------------------------------------
KNOWN ISSUES

- If you get an error about “Number of images and exposure times must match”, check the folder and exposure_times array.
- If the CRF fitting step fails due to not enough valid samples, lower the num_samples value.
- For very large images, increasing num_samples makes the system slower and isn’t really needed for this assignment.

--------------------------------------------------------------------------------
WHY THE RESULT LOOKS LIKE THIS

- With a small number of images and generic weighting/tone-mapping, the HDR will look “flat” or “greyish” but that’s expected for this kind of assignment.
- There will always be some limitations because we’re using compressed 8-bit JPGs and not RAW images.
- My implementation is intended for reproducibility and learning the pipeline, not for competition with Lightroom or Photoshop HDR.

--------------------------------------------------------------------------------
Contact me if you have any questions :))
