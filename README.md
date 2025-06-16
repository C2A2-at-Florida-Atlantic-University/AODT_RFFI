# Mobintel RFFI Project

This repository contains the source code for a research paper: "RFFI for Mobility Intelligence and MAC Address Derandomization".

Authors:
* [Stephan Mazokha](https://scholar.google.com/citations?user=tCc9M3EAAAAJ&hl=en&oi=ao)
* [Fanchen Bao](https://scholar.google.com/citations?user=LAAw4LMAAAAJ&hl=en)
* [George Sklivanitis](https://scholar.google.com/citations?user=kNTCxpEAAAAJ&hl=en)
* [Jason O. Hallstrom](https://www.fau.edu/engineering/directory/faculty/hallstrom/)

For further details, reach out to [I-SENSE at Florida Atlantic University](https://isense.fau.edu/).

## What Can You Find Here

This repository contains the following directories:

* [orbit-capture](orbit-capture): this directory contains the code and instructions for automating data capture in the Orbit testbed facility;
* [preprocessor](preprocessor): this directory contains the code and instructions for converting raw IQ samples from the previous step into input data ready for training or testing the fingerprinting method;
* [fingerprinting](fingerprinting): this directory contains our code for training the fingerprint extractor model, testing its performance and producing figures which we included in our paper.

## Requirements

For this project, you will need both Python and Matlab environments on your machine. The latest Matlab (R2024a at the time of writing) can be downloaded [here](https://www.mathworks.com/help/install/ug/install-products-with-internet-connection.html). As for Python, consider using [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

Specific requirements w.r.t. particular modules of this project can be found in corresponding subdirectories.

## That's all, folks!

Thank you for exploring our project! If you found this useful, please consider citing our paper. This would mean the world to our team.

![](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExeG1zM3RhM3ozY3J1MDNrM2xwcHRxYm9xZDhzNWU1eGcwdDNoeWphbiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/1rFPUD5ZFpc7bVYWk6/giphy.gif)

```bibtex
@article{mazokha2025mobrffi,
  title={MobRFFI: Non-cooperative Device Re-identification for Mobility Intelligence},
  author={Mazokha, Stepan and Bao, Fanchen and Sklivanitis, George and Hallstrom, Jason O},
  journal={arXiv preprint arXiv:2503.02156},
  year={2025}
}
