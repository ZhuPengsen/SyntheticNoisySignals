## Synthetic Noisy Signals

### :white_check_mark: Welcome to this repository. :smiley:
- This repository provides a method for generating noisy ship radiated noise datasets, enabling the creation of paired clean-noisy datasets for training and evaluating denoising models. The noise addition approach follows the method implemented in [FullsubNet-plus](https://arxiv.org/abs/2203.12188), with modifications based on its publicly available [code](https://github.com/RookieJunChen/FullSubNet-plus) to suit the specific requirements of our work.

### :white_check_mark: Noise Data
- All noise data is stored in the `noise` folder. Among the three categories of noise, we name the collected noise using the format `Type_net_k`. For noise from the ShipsEar dataset, we retain its original file names. However, we only provide access to the data we collected. Since we do not hold the copyright for the ShipsEar dataset, we cannot share its data directly. Instead, we use a placeholder file, `sample.wav`, for all samples related to ShipsEar. Researchers can download the original data directly from the ShipsEar website and perform the necessary preprocessing.

###  :white_check_mark: Constructing Rain, Wind, and Wave Noisy Data
- After completing data preprocessing, the first step is to create a folder named `Noisy_*`, where `*` represents the type of noise to be added. Inside this folder, create two subfolders: `clean` and `noisy`. Place the data to be processed in the `clean` folder. Next, in the `Add_Noise_Rain_Wave_Wind.py` script, set `clean_dir` to the path of the `Noisy_*` folder, `noise_dir` to the path of the noise data, and `snr_range` to the desired range of SNR. After running the script, the corresponding noisy data will be generated and saved in the `noisy` folder.

### :white_check_mark: Constructing Mixed Noisy Data
- For the mixed environment, where three types of noise are combined, the process differs from adding a single type of noise. The corresponding script is `Add_Noise_Mixed.py`. In this case, there is no need to specify a single noise type, set `noise_dir` to the folder containing all noise categories. Additionally, configure `silence_length` to define the intervals between different noise types (set to 0.1 seconds in our work) and set `snr_range` to the desired SNR range. After configuring these parameters, run the script to generate the noisy data for the mixed noise environment.

### :white_check_mark: Acknowledgements
- `audio_zen` originates directly from [FullsubNet-plus](https://github.com/RookieJunChen/FullSubNet-plus), which we used as a reference for this implementation.

###  :white_check_mark: At Last
- I hope this dataset splitting method will be helpful to everyone :smiley:.
